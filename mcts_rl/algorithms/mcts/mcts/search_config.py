from typing import Optional, NamedTuple

import regex
import numpy as np
from tqdm import trange, tqdm
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import deepspeed
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from transformers import GenerationConfig, PreTrainedTokenizerBase, AutoModelForCausalLM
from transformers.tokenization_utils import TruncationStrategy

from mcts_rl.algorithms.mcts.mcts.base import SearchConfig
from mcts_rl.algorithms.mcts.mcts.world_model import StepLMState, StepLMAction
from mcts_rl.algorithms.mcts.mcts.embedding_retrieve import get_embs_masks, check_match
from mcts_rl.utils import extract_answer, math_equal, csr_equal
from mcts_rl.configs import (
    PROMPT_BEGIN, PROMPT_USER, PROMPT_ASSISTANT, 
    SELF_EVAL_I, SELF_EVAL_E, HH_EVAL_PROMPT, QA_EVAL_PROMPT,
    EVAL_PROMPT_USER, EVAL_PROMPT_ASSISTANT, EVAL_PROMPT, SAFE_EVAL_PROMPT,
    ANSWER_HINT, HINTED_PROMPT, HINTED_EVAL_PROMPT, HINTED_EVAL_PROMPT_I, 
)


class SearchArgs(NamedTuple):
    ref_policy_model: deepspeed.DeepSpeedEngine
    base_tokenizer: PreTrainedTokenizerBase
    generation_config: GenerationConfig = None
    n_actions: int = 16
    reward_alpha: float = 0.5
    reward_confidence_default: float = 0.8
    depth_limit: int = 32
    force_terminating_on_depth_limit: bool = True
    breadth_limit: int = 16
    similarity_threshold: float = .9
    negative_gen: bool = False
    kl_coeff: float = 0.02
    disable_tqdm: bool = True


class StepLMConfig(SearchConfig):
    def __init__(self,
                 args: SearchArgs) -> None:
        super().__init__()
        self.example = None
        self.double_actions = False
        self.n_actions = args.n_actions
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
        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        n_actions = 2 * self.n_actions if not len(state) else self.n_actions
        n_actions = 1 if at_depth_limit else n_actions
        
        assert self.example['input_ids'].dim() == 1, "Input IDs should be a 1-dim sequence for a single example"
        assert self.generation_config.num_return_sequences == 1, "Otherwise will get stuck"
        
        input_ids, attention_mask = self._get_sequence_ids_in_path(state)
        unique_text_list, sequences_list = [], []
        prompt = self.base_tokenizer.decode(input_ids, skip_special_tokens=True)
        for _ in trange(n_actions, disable=self.disable_tqdm, desc='Expand: action generation', leave=False):
            if unique_text_list or prompt.startswith(PROMPT_BEGIN):
                sequences = policy_model.module.generate(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    generation_config=self.generation_config,
                    synced_gpus=True,
                    do_sample=True,
                )
            else:
                sequences = policy_model.module.generate(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    max_new_tokens=self.generation_config.max_new_tokens,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    bos_token_id=self.base_tokenizer.bos_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id,
                    pad_token_id=self.base_tokenizer.pad_token_id,
                    synced_gpus=True,
                    do_sample=False,
                )
            
            for seq in sequences:
                full_generated = ' ' + self.base_tokenizer.decode(seq[input_ids.size(-1):], skip_special_tokens=True)
                
                raw_sentences = sent_tokenize(full_generated)
                sentences, sent = [], ''
                for i, raw_sent in enumerate(raw_sentences):
                    sent += f' {raw_sent}' if len(sent) else raw_sent
                    if len(sent) > 3 or i == len(raw_sentences) - 1:    # Sentences cannot be too short
                        sentences.append(sent)
                        sent = ''
                if len(sentences) == 1:
                    sentences = regex.split(r'\n[\n]+', full_generated)
                
                sents = []
                for sid, sent in enumerate(sentences):
                    sents.append(sent)
                    if len(' '.join(sents).strip()) and sid >= len(sentences) - 2:
                        break
                step = ' '.join(sents)
                
                text = step if len(sents) < len(sentences) else self.base_tokenizer.decode(seq[input_ids.size(-1):])
                if text in unique_text_list or not text.strip():
                    continue
                
                gen_ids = self.base_tokenizer(
                    prompt + step,
                    add_special_tokens=True,
                    truncation=TruncationStrategy.LONGEST_FIRST,
                    return_tensors='pt',
                )['input_ids'][0].to(seq.device) if len(sents) < len(sentences) else seq
                
                gen_ids = gen_ids[input_ids.size(-1):]
                if not gen_ids.size(-1):
                    continue
                sequences_list.append(gen_ids)
                unique_text_list.append(text)
        
        if not len(sequences_list):
            sequences_list.append(torch.tensor([self.base_tokenizer.eos_token_id]).to(input_ids.device))
            unique_text_list.append(self.base_tokenizer.eos_token)
        
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
            input_prompt = input_txt.split(PROMPT_ASSISTANT)[0].rstrip() + '\n' + ANSWER_HINT.format(answer=gt_ans, prompt=PROMPT_ASSISTANT)
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

    @torch.no_grad()
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
    ) -> list[tuple[float, bool]]:
        outputs = []
        input_ids, attention_mask = self._get_sequence_ids_in_path(state)
        prompt = self.base_tokenizer.decode(input_ids, skip_special_tokens=True)
        correct_token_ids = [self.base_tokenizer.encode(tok)[1] for tok in ['B', 'correct', 'Correct']]
        correct_token_ids += [self.base_tokenizer.encode(tok)[2] for tok in ['(B']]
        correct_token_ids = list(set(correct_token_ids))
        option_token_ids = {
            x: [self.base_tokenizer.encode(tok)[-1] for i, tok in enumerate([x, f'({x}'])]
            for x in 'ABCD'
        }
        
        @torch.no_grad()
        def _eval(eval_prompt, generated, gt_ans, device):
            eval_inputs = self.base_tokenizer(
                eval_prompt,
                add_special_tokens=True,
                truncation=TruncationStrategy.LONGEST_FIRST,
                return_tensors='pt',
            )
            sequences = policy_model.module.generate(
                input_ids=eval_inputs['input_ids'].to(device),
                attention_mask=eval_inputs['attention_mask'].to(device),
                max_new_tokens=4,
                do_sample=False,
                output_scores=True,
                synced_gpus=True,
                return_dict_in_generate=True,
                # eos_token_id=self.base_tokenizer.encode('\n')[-1],
            )
            
            sequences, scores = sequences.sequences.cpu(), sequences.scores
            seq = sequences[0][eval_inputs['input_ids'].size(-1):]
            response = self.base_tokenizer.decode(seq, skip_special_tokens=True)
            
            if gt_ans is None:
                conf = 0.0
                for idx, _id in enumerate(seq):
                    if self.base_tokenizer.decode(_id).strip() in 'ABCD':
                        logprobs = F.log_softmax(scores[idx][0], dim=-1)
                        confs = {k: sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in v) for k, v in option_token_ids.items()}
                        conf = 1.0 * confs['D'] + .0 * confs['C'] + (-1.0) * confs['B'] + (-2.0) * confs['A']
                return response, conf, False
            
            conf, correct_score = 0.0, 0.0
            for idx, _id in enumerate(seq):
                if self.base_tokenizer.decode(_id).strip() in ['A', 'B', 'correct', 'wrong', 'incorrect']:
                    logprobs = F.log_softmax(scores[idx][0], dim=-1)
                    conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                    break
            if conf == 0:
                for idx, _id in enumerate(seq):
                    if self.base_tokenizer.decode(_id).strip() in ['Cor', 'In', 'A', 'B', 'correct', 'wrong', 'incorrect']:
                        logprobs = F.log_softmax(scores[idx][0], dim=-1)
                        conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                        break
            
            if isinstance(gt_ans, str):
                correct = math_equal(extract_answer(generated), gt_ans)
                correct_score = 1 if correct else -.5
            else:
                correct_score = csr_equal(generated, gt_ans)
            return response, conf, correct_score
        
        for action, log_probs, ref_log_probs in zip(action_batch, log_probs_batch, ref_log_probs_batch):
            if add_kl:
                kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)
                base_rewards = kl_divergence_estimate  # size = (L,)
            else:
                base_rewards = None
            
            step = self.base_tokenizer.decode(action)
            is_terminal = step.endswith(self.base_tokenizer.eos_token)
            if is_terminal:
                step = step[:-len(self.base_tokenizer.eos_token)]
            texts = prompt.split(PROMPT_ASSISTANT)
            input_txt = PROMPT_ASSISTANT.join(texts[:-1])# + PROMPT_ASSISTANT
            init_answer = texts[-1] if len(texts) > 1 else ''
            
            if input_txt.startswith(PROMPT_BEGIN):
                eval_prompt = HH_EVAL_PROMPT.format(input=input_txt, prompt=PROMPT_ASSISTANT + texts[-1] + step)
                # eval_prompt = QA_EVAL_PROMPT.format(input=input_txt, prompt=PROMPT_ASSISTANT + texts[-1] + step)
                eval_result, eval_conf, eval_correct_score = _eval(eval_prompt, prompt.split(PROMPT_ASSISTANT)[-1] + ' ' + step, None, action.device)
            else:
                if self.example['reasoning'] and self.example['reasoning'] != self.example['answer_content']:
                    solution = f'{self.example["reasoning"]}\nThe answer is {self.example["answer"]}'
                    gt_ans = self.example["answer"]
                else:
                    gt_ans = [f"({self.example['answer']})", self.example['answer_content']] \
                        if self.example['answer'] != self.example['answer_content'] else [f"({self.example['answer']})"]
                    solution = f'The answer is {gt_ans[0]} {gt_ans[1]}'
                eval_prompt = HINTED_EVAL_PROMPT.format(input=input_txt, solution=solution, prompt=init_answer + step)
                eval_result, eval_conf, eval_correct_score = _eval(eval_prompt, prompt.split(PROMPT_ASSISTANT)[-1] + ' ' + step, gt_ans, action.device)
            
            print(eval_prompt + eval_result + f' ({eval_conf})')
            score = eval_conf / max(parent_depth - 3, 1)    # Penalize generations that are too long
            if is_terminal and not input_txt.startswith(PROMPT_BEGIN):
                score += eval_correct_score

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
               logscore: float = None,
               logprob: float = None) -> tuple[float, dict]:
        assert logscore is not None, "score from RM is required to calculate reward in this search config, consider passing it in fast_reward"
        assert logprob is not None, "confidence from LM is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(logscore, logprob)
