# Adapted from: https://github.com/maitrix-org/llm-reasoners/blob/main/examples/RAP/gsm8k/search_config.py

from typing import NamedTuple

import regex
import random
import numpy as np
from tqdm import trange, tqdm
from string import punctuation

# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import deepspeed
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import GenerationConfig, PreTrainedTokenizerBase, AutoModelForCausalLM
from transformers.tokenization_utils import TruncationStrategy

from mcts_rl.algorithms.mcts.mcts.base import SearchConfig
from mcts_rl.algorithms.mcts.mcts.world_model import StepLMState, StepLMAction
from mcts_rl.algorithms.mcts.mcts.embedding_retrieve import get_embs_masks, check_match
from mcts_rl.utils import extract_answer, math_equal, csr_equal
from mcts_rl.configs import (
    PROMPT_BEGIN, PROMPT_ASSISTANT, PROMPT_ASSISTANT_MCQ,
    EVAL_PROMPT_USER, EVAL_PROMPT_ASSISTANT,
    ANSWER_HINT, HINTED_EVAL_PROMPT, REWARD_EVAL_PROMPT, HH_EVAL_PROMPT,
    LLAMA3_PROMPT_ASSISTANT, LLAMA3_PROMPT_ASSISTANT_MCQ,
    LLAMA3_HINTED_EVAL_PROMPT, LLAMA3_EVAL_PROMPT_ASSISTANT,
)


class SearchArgs(NamedTuple):
    ref_policy_model: deepspeed.DeepSpeedEngine
    base_tokenizer: PreTrainedTokenizerBase
    generation_config: GenerationConfig = None
    n_actions: int = 16
    n_init_actions: int = 16
    reward_alpha: float = 0.5
    reward_confidence_default: float = 0.8
    depth_limit: int = 32
    force_terminating_on_depth_limit: bool = False
    breadth_limit: int = 16
    similarity_threshold: float = .99
    negative_gen: bool = False
    kl_coeff: float = 0.02
    disable_tqdm: bool = True
    no_self_eval: bool = False
    reward_model: deepspeed.DeepSpeedEngine = None
    reward_tokenizer: PreTrainedTokenizerBase = None
    use_code: bool = False
    use_mcq: bool = False
    eval_mode: bool = False
    init_temperature: float = 1.0
    temperature: float = 1.0
    get_tp_zero: bool = False
    model_type: str = 'mistral'
    include_gt: bool = True
    verbose: bool = False


class StepLMConfig(SearchConfig):
    def __init__(self, args: SearchArgs) -> None:
        super().__init__()
        self.example = None
        self.double_actions = False
        self.n_actions = args.n_actions
        self.n_init_actions = args.n_init_actions
        self.force_terminating_on_depth_limit = args.force_terminating_on_depth_limit
        self.depth_limit = args.depth_limit
        self.reward_alpha = args.reward_alpha
        self.reward_confidence_default = args.reward_confidence_default
        self.action_size = args.breadth_limit
        self.similarity_threshold = args.similarity_threshold
        self.negative_gen = args.negative_gen
        
        self.ref_policy_model = args.ref_policy_model
        
        self.base_tokenizer = args.base_tokenizer
        self.generation_config = args.generation_config        
        self.kl_coeff = args.kl_coeff
        self.disable_tqdm = args.disable_tqdm
        
        self.no_self_eval = args.no_self_eval
        self.reward_model = args.reward_model
        self.reward_tokenizer = args.reward_tokenizer
        
        self.use_code = args.use_code
        self.use_mcq = args.use_mcq
        self.eval_mode = args.eval_mode
        
        self.init_temperature = args.init_temperature
        self.temperature = args.temperature
        
        self.prompt_assistant = PROMPT_ASSISTANT_MCQ if self.use_mcq else PROMPT_ASSISTANT
        
        self.get_tp_zero = args.get_tp_zero
        self.model_type = args.model_type
        if self.model_type == 'llama3':
            self.prompt_assistant = LLAMA3_PROMPT_ASSISTANT_MCQ if self.use_mcq else LLAMA3_PROMPT_ASSISTANT
            
        self.include_gt = args.include_gt
        self.verbose = args.verbose

    def _gather_log_probabilities(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Gather log probabilities of the given labels from the logits."""
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(dim=-1))
        return log_probs_labels.squeeze(dim=-1)
    
    def _filter_via_similarity(self, raw_results: list):
        seq_ids, seq_embs = [], []
        available_indexes, idx = [], -1
        for ids, _, embs in raw_results:
            idx += 1
            if len(seq_ids) >= self.action_size or ids.tolist() in seq_ids:
                continue
            if not len(seq_ids):
                seq_ids.append(ids.tolist())
                seq_embs.append(embs)
                available_indexes.append(idx)
                continue
            key_embs, key_masks, key_idfs = get_embs_masks(seq_ids, seq_embs)
            _, _, F_scores = check_match(key_embs, key_masks, key_idfs, ids, embs)
            if F_scores.max().item() > self.similarity_threshold:
                continue
            seq_ids.append(ids.tolist())
            seq_embs.append(embs)
            available_indexes.append(idx)
        
        return [raw_results[idx] for idx in available_indexes]

    @torch.no_grad()
    def _get_logits(self, _input_ids: torch.LongTensor, 
                    attention_mask: torch.BoolTensor, 
                    model: AutoModelForCausalLM = None):
        return_hidden_states = model is not None
        model = self.ref_policy_model.module if model is None else model
        
        _outputs = model(
            _input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )
        logits = _outputs.logits[0]
        hidden_states = _outputs.hidden_states[-1][0] if return_hidden_states else None
        return logits, hidden_states
    
    @torch.no_grad()
    def get_actions(self, policy_model, state: StepLMState, add_kl: bool = False) -> list[StepLMAction]:
        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 > self.depth_limit
        n_actions = self.n_init_actions if not len(state) else self.n_actions
        if self.use_mcq:
            n_actions = 2 if at_depth_limit and self.n_actions > 1 else n_actions  # set a larger basic action space for MCQ
        else:
            n_actions = 1 if at_depth_limit else n_actions
        
        assert self.example['input_ids'].dim() == 1, "Input IDs should be a 1-dim sequence for a single example"
        assert self.generation_config.num_return_sequences == 1, "Otherwise will get stuck"
        
        input_ids, attention_mask = self._get_sequence_ids_in_path(state)
        prompt = self.base_tokenizer.decode(input_ids, 
                                            skip_special_tokens=self.model_type != 'llama3', 
                                            clean_up_tokenization_spaces=False)
        terminators = [self.base_tokenizer.eos_token_id]
        terminators += [self.base_tokenizer.convert_tokens_to_ids("<|eot_id|>")] if self.model_type == 'llama3' else []
        unique_text_list, sequences_list = [], []
        for _ in trange(n_actions, disable=self.disable_tqdm, desc='Expand: action generation', leave=False):
            cur_max_new_tokens = self.generation_config.max_new_tokens + (0 if self.use_mcq else 16)
            ## sample candidate steps to construct action space from LLMs with some randomness (temperature >= 0)
            if (not self.get_tp_zero) or unique_text_list or prompt.startswith(PROMPT_BEGIN):
                cur_max_new_tokens = (self.generation_config.max_length - input_ids.size(-1)) if n_actions == 1 else cur_max_new_tokens
                cur_max_new_tokens = max(1, min(cur_max_new_tokens, self.generation_config.max_length - input_ids.size(-1)))
                sequences = policy_model.module.generate(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    temperature=self.init_temperature if not len(state) else (self.temperature if random.random() < .75 else 1.0),
                    max_new_tokens=cur_max_new_tokens,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    bos_token_id=self.base_tokenizer.bos_token_id,
                    pad_token_id=self.base_tokenizer.eos_token_id,
                    do_sample=True,
                    eos_token_id=terminators,
                    synced_gpus=True,
                )
            else:
                cur_max_new_tokens = (self.generation_config.max_length - input_ids.size(-1)) if n_actions == 1 else cur_max_new_tokens
                cur_max_new_tokens = max(1, min(cur_max_new_tokens, self.generation_config.max_length - input_ids.size(-1)))
                sequences = policy_model.module.generate(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    max_new_tokens=cur_max_new_tokens,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    bos_token_id=self.base_tokenizer.bos_token_id,
                    pad_token_id=self.base_tokenizer.eos_token_id,
                    do_sample=False,
                    eos_token_id=terminators,
                    synced_gpus=True,
                )
            
            for seq in sequences:
                ## get full generated text for current step
                full_generated = self.base_tokenizer.decode(seq, 
                                                            skip_special_tokens=self.model_type != 'llama3', 
                                                            clean_up_tokenization_spaces=False)
                full_generated = full_generated[len(prompt):] if full_generated.startswith(prompt) else \
                                        self.base_tokenizer.decode(seq[input_ids.size(-1):], 
                                                                   skip_special_tokens=self.model_type != 'llama3', 
                                                                   clean_up_tokenization_spaces=False)
                raw_full_generated = full_generated
                if self.model_type != 'llama3':
                    full_generated = full_generated.split(EVAL_PROMPT_USER.split(':')[0])[0]
                    if raw_full_generated != full_generated:
                        full_generated = full_generated.rstrip() + self.base_tokenizer.eos_token
                ## split text sequence into granular steps
                newline_flag = True
                raw_sentences = regex.split(r'[\n]+', full_generated)
                if len(raw_sentences) <= 1 and not self.use_code:
                    raw_sentences = sent_tokenize(full_generated)
                    newline_flag = False
                sentences, sent, subcnt = [], '', 0
                for i, raw_sent in enumerate(raw_sentences):
                    sent += (f'\n{raw_sent}' if newline_flag else f' {raw_sent}') if subcnt else raw_sent
                    subcnt += 1
                    if i == len(raw_sentences) - 1:
                        sentences.append(sent)
                        sent, subcnt = '', 0
                    elif len(sent) > 3:    # Sentences cannot be too short
                        if not self.use_code or (any(x.strip() and not x.strip().startswith('#') for x in sent.split('\n')) \
                            and all(not sent.strip().endswith(x) for x in [':', '(', '['])):
                            sentences.append(sent)
                            sent, subcnt = '', 0
                for i, raw_sent in enumerate(sentences):
                    ## identify end of sentence
                    if ' answer is' in raw_sent and not raw_sent.endswith(' answer is'):
                        if self.model_type == 'llama3':
                            if "<|eot_id|>" not in raw_sent:
                                sentences[i] += "<|eot_id|>"
                        elif self.base_tokenizer.eos_token not in raw_sent:
                            sentences[i] += self.base_tokenizer.eos_token
                        sentences = sentences[:i + 1]
                        break
                
                ## collect generation as steps
                sents = []
                if not len(sentences): continue
                if self.n_actions > 1 and not len(state) and len(sentences) > 1 and \
                    any(x in ''.join(sentences) for x in ["<|eot_id|>", self.base_tokenizer.eos_token]):
                    # further break down single-step output if possible
                    sents = sentences[:-1]
                elif sentences[-1].rstrip().endswith('.') or len(sentences) < 2:
                    # fullstop / cannot break down --> accept as a valid step
                    sents = sentences
                elif len(state) and ((seq.size(-1) - input_ids.size(-1) < cur_max_new_tokens) or 
                    any(x in ''.join(sentences) for x in ["<|eot_id|>", self.base_tokenizer.eos_token])):
                    # reach a terminate state
                    sents = sentences
                elif self.n_actions > 1 and len(self.base_tokenizer.encode(' '.join(sentences[:-1]))) >= cur_max_new_tokens * 7/8:
                    sents = sentences[:-1]
                else:
                    sents = sentences
                step = '\n'.join(sents) if newline_flag else ' '.join(sents)
                
                text = step
                if len(sents) == len(sentences):
                    text = full_generated
                elif len(sents) < len(sentences) and newline_flag:
                    text += '\n'
                if text in unique_text_list or not text.strip():
                    continue
                
                gen_ids = self.base_tokenizer(
                    prompt + text,
                    add_special_tokens=self.model_type != 'llama3',
                    truncation=TruncationStrategy.LONGEST_FIRST,
                    return_tensors='pt',
                )['input_ids'][0].to(seq.device) if len(sents) < len(sentences) or raw_full_generated != full_generated else seq
                
                gen_ids = gen_ids[input_ids.size(-1):]
                if not gen_ids.size(-1):
                    continue
                sequences_list.append(gen_ids)
                unique_text_list.append(text)
        
        ## integrate G.T. guidance (ground-truth solutions used in SFT tuning)
        if self.include_gt and not self.use_mcq and self.example.get('reasoning', ''):
            pre_gen = prompt.split(LLAMA3_EVAL_PROMPT_ASSISTANT)[-1] if self.model_type == 'llama3' else prompt.split(EVAL_PROMPT_ASSISTANT)[-1]
            newline_flag = True
            _solution_steps = regex.split(r'[\n]+', self.example['reasoning'].strip())
            if len(_solution_steps) < 2 and not self.use_code:
                _solution_steps = sent_tokenize(self.example['reasoning'].strip())
                newline_flag = False
            if not self.use_code and ' answer is' not in _solution_steps[-1]:
                _solution_steps.append("The answer is {}{}".format(self.example['answer'], "<|eot_id|>" if self.model_type == 'llama3' else self.base_tokenizer.eos_token))
            solution_steps = []
            for i, x in enumerate(_solution_steps):
                if newline_flag and i < len(_solution_steps) - 1:
                    solution_steps.append(f'{x}\n')
                elif not newline_flag and i > 0:
                    solution_steps.append(f' {x}')
                else:
                    solution_steps.append(x)
            cur_step = ''
            for i, step in enumerate(solution_steps):
                if pre_gen.lstrip() == cur_step.lstrip():
                    j = len(solution_steps)
                    text = ''.join(solution_steps[i:j])
                    while n_actions > 1 and j > i + 1 and (len(self.base_tokenizer.encode(text)) > (self.generation_config.max_new_tokens * 9/8) or
                                                           (j == len(solution_steps) and i == 0)):
                        j -= 1
                        text = ''.join(solution_steps[i:j])
                    if text not in unique_text_list:
                        if self.model_type != 'llama3' and not pre_gen.strip() and not text.startswith('\n'):
                            text = f' {text}'
                        gen_ids = self.base_tokenizer(
                            prompt + text,
                            add_special_tokens=self.model_type != 'llama3',
                            truncation=TruncationStrategy.LONGEST_FIRST,
                            return_tensors='pt',
                        )['input_ids'][0].to(seq.device)
                        sequences_list = [gen_ids[input_ids.size(-1):]] + sequences_list
                        unique_text_list = [text] + unique_text_list
                    break
                cur_step += step
        
        ## add eos token
        if not len(sequences_list):
            if self.model_type == 'llama3':
                sequences_list.append(torch.tensor([self.base_tokenizer.convert_tokens_to_ids("<|eot_id|>")]).to(input_ids.device))
                unique_text_list.append("<|eot_id|>")
            else:
                sequences_list.append(torch.tensor([self.base_tokenizer.eos_token_id]).to(input_ids.device))
                unique_text_list.append(self.base_tokenizer.eos_token)
        
        ## gather result candidate steps (text, embeddings, logits, logprobs)
        results = []
        for gen_ids in sequences_list:
            seq_input_ids = torch.cat((input_ids, gen_ids.to(input_ids.device)), dim=-1).unsqueeze(0)
            seq_attention_mask = torch.logical_and(
                seq_input_ids.not_equal(self.base_tokenizer.pad_token_id),
                seq_input_ids.not_equal(self.base_tokenizer.unk_token_id),
            )
            logits, hidden_states = self._get_logits(seq_input_ids, attention_mask=seq_attention_mask, model=policy_model.module)
            log_probs = self._gather_log_probabilities(logits[input_ids.size(-1)-1:-1, :], gen_ids.to(logits.device))
            embs = hidden_states[input_ids.size(-1):]
            if add_kl:
                ref_logits, _ = self._get_logits(seq_input_ids, attention_mask=seq_attention_mask)
                ref_log_probs = self._gather_log_probabilities(ref_logits[input_ids.size(-1)-1:-1, :], gen_ids.to(ref_logits.device))
            else:
                ref_log_probs = None
            results.append((gen_ids, (log_probs, ref_log_probs), embs))
        return self._filter_via_similarity(results)

    def _append_action(self, input_ids: torch.LongTensor, action: list[int]) -> torch.LongTensor:
        return torch.cat((
            input_ids,
            torch.tensor(action).to(input_ids.device),
        ))

    def _get_sequence_ids_in_path(self, state: StepLMState, hint: bool=False) -> tuple[torch.LongTensor, torch.BoolTensor]:
        length = self.example['attention_mask'].nonzero()[-1].item() + 1
        input_ids = self.example['input_ids'][:length]
        if hint:
            input_txt = self.base_tokenizer.decode(input_ids, skip_special_tokens=True)
            gt_ans = f"({self.example['answer']}) {self.example['answer_content']}" \
                if self.example['answer'] != self.example['answer_content'] else f"{self.example['answer']}"
            input_prompt = input_txt.split(self.prompt_assistant)[0].rstrip() + '\n' + ANSWER_HINT.format(answer=gt_ans, prompt=self.prompt_assistant)
            input_ids = self.base_tokenizer(
                input_prompt,
                add_special_tokens=True,
                truncation=TruncationStrategy.LONGEST_FIRST,
                return_tensors='pt',
            )['input_ids'][0].to(input_ids.device)
        existed_state = [s.next_step_ids for s in state]
        input_ids = torch.cat([input_ids] + existed_state, dim=-1)
        
        attention_mask = torch.logical_and(
            input_ids.not_equal(self.base_tokenizer.pad_token_id),
            input_ids.not_equal(self.base_tokenizer.unk_token_id),
        )
        return input_ids, attention_mask

    def get_values(
        self,
        policy_model,
        state: StepLMState,
        action_batch: list[StepLMAction],
        log_probs_batch: list[torch.Tensor],
        ref_log_probs_batch: list[torch.Tensor],
        correct_token_ids: list[int] = [319],
        add_kl: bool = False,
        parent_depth: int = 0,
        parent_value: float = 0.0,
    ) -> list[tuple[float, bool]]:
        outputs = []
        input_ids, attention_mask = self._get_sequence_ids_in_path(state)
        prompt = self.base_tokenizer.decode(input_ids, skip_special_tokens=True)
        try:
            correct_token_ids = [self.base_tokenizer.encode(tok)[1] for tok in ['B', 'correct', 'Correct']]
        except:
            correct_token_ids = [self.base_tokenizer.encode(tok)[-1] for tok in ['B', 'correct', 'Correct']]
        correct_token_ids += [self.base_tokenizer.encode(tok)[-1] for tok in ['(B', ' B', ' correct']]
        if len(self.base_tokenizer.encode('Correct')) < 3:
            correct_token_ids += [self.base_tokenizer.encode(tok)[-1] for tok in [' Correct']]
        correct_token_ids = list(set(correct_token_ids))
        option_token_ids = {
            x: [self.base_tokenizer.encode(tok)[-1] for i, tok in enumerate([x, f'({x}'])]
            for x in 'ABCD'
        }
        
        @torch.no_grad()
        def _eval(eval_prompt, generated, gt_ans, device, self_eval=(not self.no_self_eval)):
            conf, correct_score = 1.0, 0.0
            
            if self_eval:
                if self.reward_model is not None:
                    ## if there is external reward model available for evaluation
                    response = ''
                    eval_inputs = self.reward_tokenizer(
                        eval_prompt,
                        add_special_tokens=True,
                        truncation=TruncationStrategy.LONGEST_FIRST,
                        return_tensors='pt',
                    )
                    conf = torch.sigmoid(self.reward_model(
                        eval_inputs['input_ids'].to(device), 
                        attention_mask=eval_inputs['attention_mask'].to(device)
                    ).end_scores.squeeze(dim=-1)[0]).detach().item()
                else:
                    ## self-evaluation
                    eval_inputs = self.base_tokenizer(
                        eval_prompt,
                        add_special_tokens=True,
                        truncation=TruncationStrategy.LONGEST_FIRST,
                        return_tensors='pt',
                    )
                    
                    if eval_inputs['input_ids'].size(-1) >= self.generation_config.max_length:
                        seq = []
                        response = 'N/A'
                    else:
                        sequences = policy_model.module.generate(
                            input_ids=eval_inputs['input_ids'].to(device),
                            attention_mask=eval_inputs['attention_mask'].to(device),
                            max_new_tokens=4,
                            do_sample=False,
                            output_scores=True,
                            synced_gpus=True,
                            return_dict_in_generate=True,
                            pad_token_id=self.base_tokenizer.eos_token_id,
                        )
                        
                        sequences, scores = sequences.sequences.cpu(), sequences.scores
                        seq = sequences[0][eval_inputs['input_ids'].size(-1):]
                        response = self.base_tokenizer.decode(seq, skip_special_tokens=True)
            else:
                response = ''
            
            ## calculate confidence scores
            
            if gt_ans is None:
                if self_eval:
                    conf = 0.0
                    for idx, _id in enumerate(seq):
                        if self.base_tokenizer.decode(_id).strip() in 'ABCD':
                            logprobs = F.log_softmax(scores[idx][0], dim=-1)
                            confs = {k: sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in v) for k, v in option_token_ids.items()}
                            conf = 1.0 * confs['D'] + .0 * confs['C'] + (-1.0) * confs['B'] + (-2.0) * confs['A']
                return response, conf, False
            
            if self_eval and self.reward_model is None:
                conf = 0.0
                for idx, _id in enumerate(seq):
                    if self.base_tokenizer.decode(_id).strip() in ['A', 'B', 'correct', 'wrong', 'incorrect']:
                        logprobs = F.log_softmax(scores[idx][0], dim=-1)
                        conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                        break
                if conf == 0.0:
                    for idx, _id in enumerate(seq):
                        if self.base_tokenizer.decode(_id).strip() in ['Cor', 'In', 'A', 'B', 'correct', 'wrong', 'incorrect']:
                            logprobs = F.log_softmax(scores[idx][0], dim=-1)
                            conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                            break
            
            if isinstance(gt_ans, str):
                pred = extract_answer(generated, use_code=self.use_code)
                correct = math_equal(pred, gt_ans)
                correct_score = 1 if correct else -1
            else:
                correct_score = csr_equal(generated, gt_ans)
            return response, conf, correct_score
        
        for action, log_probs, ref_log_probs in zip(action_batch, log_probs_batch, ref_log_probs_batch):
            if add_kl:
                kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)
                base_rewards = kl_divergence_estimate  # size = (L,)
            else:
                base_rewards = None
            
            step = self.base_tokenizer.decode(action, skip_special_tokens=False)
            is_terminal = step.endswith(self.base_tokenizer.eos_token) or step.endswith('<|eot_id|>')
            if is_terminal:
                step = step[:-len('<|eot_id|>' if self.model_type == 'llama3' else self.base_tokenizer.eos_token)]
            if self.model_type == 'llama3':
                texts = prompt.split('user\n\n')[-1].split('assistant\n\n')
                input_txt = 'assistant\n\n'.join(texts[:-1])
            else:
                texts = prompt.split(self.prompt_assistant)
                input_txt = self.prompt_assistant.join(texts[:-1]) # + PROMPT_ASSISTANT
            init_answer = texts[-1] if len(texts) > 1 else ''
            
            if input_txt.startswith(PROMPT_BEGIN):
                ## for HH-RLHF
                eval_prompt = HH_EVAL_PROMPT.format(input=input_txt, prompt=self.prompt_assistant + texts[-1] + step)
                # eval_prompt = QA_EVAL_PROMPT.format(input=input_txt, prompt=self.prompt_assistant + texts[-1] + step)
                eval_result, eval_conf, eval_correct_score = _eval(eval_prompt, prompt.split(self.prompt_assistant)[-1] + ' ' + step, None, action.device)
            else:
                if self.example['reasoning'] and self.example['reasoning'] != self.example['answer_content']:
                    ## for cases where the intermediate steps are available in G.T. solutions
                    newline_flag = True
                    _solution_steps = regex.split(r'[\n]+', self.example['reasoning'].strip())
                    if len(_solution_steps) < 2 and not self.use_code:
                        _solution_steps = sent_tokenize(self.example['reasoning'].strip())
                        newline_flag = False
                    if not self.use_code and ' answer is' not in _solution_steps[-1]:
                        _solution_steps.append("The answer is {}{}".format(self.example['answer'], "<|eot_id|>" if self.model_type == 'llama3' else self.base_tokenizer.eos_token))
                    solution_steps = []
                    for i, x in enumerate(_solution_steps):
                        if newline_flag and i < len(_solution_steps) - 1:
                            solution_steps.append(f'{x}\n')
                        elif not newline_flag and i > 0:
                            solution_steps.append(f' {x}')
                        else:
                            solution_steps.append(x)
                    solution = ''.join(solution_steps)
                    gt_ans = self.example["answer"]
                else:
                    ## for cases (MCQ) where only final answers are available
                    gt_ans = [f"({self.example['answer']})", self.example['answer_content']] \
                        if self.example['answer'] != self.example['answer_content'] else [f"({self.example['answer']})"]
                    solution = f'The answer is {gt_ans[0]} {gt_ans[1]}'
                
                if self.reward_model is not None:
                    eval_prompt = REWARD_EVAL_PROMPT.format(input=input_txt, prompt=init_answer + step, 
                                                            eos_token=self.reward_tokenizer.eos_token)
                else:
                    if self.model_type == 'llama3':
                        eval_prompt_user = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
                        input_txt = eval_prompt_user + input_txt.split(f'{eval_prompt_user}')[-1].lstrip()
                        eval_prompt = LLAMA3_HINTED_EVAL_PROMPT.format(input=input_txt, solution=solution, 
                                                                       prompt=init_answer + step.replace('<|eot_id|>', ''))
                    else:
                        eval_prompt_user = 'QUESTION:'
                        input_txt = eval_prompt_user + input_txt.split(f'{eval_prompt_user}')[-1].lstrip()
                        eval_prompt = HINTED_EVAL_PROMPT.format(input=input_txt, solution=solution, 
                                                                prompt=init_answer + step)
                
                eval_result, eval_conf, eval_correct_score = \
                    _eval(eval_prompt.lstrip(), init_answer + step.replace('<|eot_id|>', ''), gt_ans, action.device)
                if self.example['reasoning'] and not self.use_mcq and eval_correct_score >= 1:
                    if not solution.strip().startswith((init_answer + step).strip()):
                        eval_correct_score *= 5/4
            
            if self.verbose:
                print(f'\n======\n{eval_prompt} {eval_result} ({eval_conf})')
            if self.use_code and is_terminal:
                pred = extract_answer(prompt.split(self.prompt_assistant)[-1] + step, use_code=self.use_code)
                if self.verbose:
                    print('\nPredicted answer is: {} | Ground-truth is: {}'.format(pred, gt_ans))
            score = eval_conf / max(parent_depth - 3, 1)    # Penalize generations that are too long
            if score == 0: score = parent_value
            if is_terminal and not input_txt.startswith(PROMPT_BEGIN) and not self.eval_mode:
                if self.n_actions < 2 or parent_depth > 0 or eval_correct_score <= 0 or not self.use_mcq:
                    score += eval_correct_score if parent_depth > 1 or not self.use_mcq or eval_correct_score <= 0 else eval_correct_score * 0.5
            outputs.append((score, base_rewards, is_terminal))
        return outputs

    def calculate_reward(self, logscore, logprob=None):
        if logprob is None:
            logprob = np.log(self.reward_confidence_default)
        if self.negative_gen:
            return self.reward_alpha * np.log(1 - np.exp(logscore)) + (1 - self.reward_alpha) * logprob, \
                {'logscore': logscore, 'logprob': logprob}
        return self.reward_alpha * logscore + (1 - self.reward_alpha) * logprob, \
            {'logscore': logscore, 'logprob': logprob}

    def reward(self, state: StepLMState, action: StepLMAction,
               logscore: float = None, logprob: float = None) -> tuple[float, dict]:
        assert logscore is not None, "score from RM is required to calculate reward in this search config, consider passing it in fast_reward"
        assert logprob is not None, "confidence from LM is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(logscore, logprob)
