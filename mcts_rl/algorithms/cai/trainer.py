# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.nn.functional as F
from transformers.tokenization_utils import TruncationStrategy
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
)

from mcts_rl.datasets import (
    PromptOnlyBatch, PromptOnlyPostBatch,
)
from mcts_rl.trainers import TSRLTrainer
from mcts_rl.utils import (
    gather_log_probabilities,
    get_all_reduce_mean,
    extract_answer,
    math_equal,
    csr_equal,
)
from mcts_rl.algorithms.mcts.mcts.embedding_retrieve import get_embs_masks, check_match
from mcts_rl.configs.constants import (
    PROMPT_ASSISTANT, 
    HINTED_EVAL_PROMPT,
)


class CAITrainer(TSRLTrainer):
    TRAINING_TYPE = 'cai'
    
    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.global_step = 0

        self.init_models()
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        
        self.generation_config = GenerationConfig(
            max_length=self.args.max_length,
            max_new_tokens=self.args.max_new_tokens,
            num_return_sequences=self.args.num_return_sequences,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        dist.barrier()
        self.init_logger()

        # Those value can be changed
        self.kl_coeff = self.args.kl_coeff
        self.clip_range_ratio = self.args.clip_range_ratio
        self.clip_range_score = self.args.clip_range_score
        self.clip_range_value = self.args.clip_range_value
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.scale_coeff = self.args.scale_coeff

    def init_mcts_searcher(self) -> None:
        return
    
    def _gather_log_probabilities(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Gather log probabilities of the given labels from the logits."""
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(dim=-1))
        return log_probs_labels.squeeze(dim=-1)
    
    @torch.no_grad()
    def _get_logits(
        self, 
        _input_ids: torch.LongTensor, 
        attention_mask: torch.BoolTensor, 
        model: AutoModelForCausalLM = None,
        return_hidden_states: bool = False,
    ):        
        _outputs = model(
            _input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states,
        )
        logits = _outputs.logits[0]
        hidden_states = _outputs.hidden_states[-1][0] if return_hidden_states else None
        return logits, hidden_states
    
    @torch.no_grad()
    def tree_constructor(self, prompt_only_batch: PromptOnlyBatch | PromptOnlyPostBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""        
        input_ids = prompt_only_batch['input_ids']
        attention_mask = prompt_only_batch['attention_mask']
        assert input_ids.size(0) == 1, '''Only support one instance per device.'''
        gt_answer, solution = prompt_only_batch['answer'][0], prompt_only_batch['reasoning'][0]
        
        sequences_list, unique_text_list = [], []
        for _ in range(3):  # TODO: magic number
            seq = self.actor_model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.args.max_length,
                synced_gpus=True,
                do_sample=True,
                num_return_sequences=1,
            )
            seq = seq[0][input_ids.size(-1):]
            text = self.tokenizer.decode(seq)
            if text in unique_text_list or not text.strip():
                continue
            sequences_list.append(seq)
            unique_text_list.append(text)
        
        raw_results = []
        for gen_ids in sequences_list:
            seq_input_ids = torch.cat((input_ids[0], gen_ids), dim=-1).unsqueeze(0)
            seq_attention_mask = torch.logical_and(
                seq_input_ids.not_equal(self.tokenizer.pad_token_id),
                seq_input_ids.not_equal(self.tokenizer.unk_token_id),
            )
            
            ref_logits, _ = self._get_logits(
                seq_input_ids,
                attention_mask=seq_attention_mask,
                model=self.actor_reference_model.module,
            )
            ref_log_probs = self._gather_log_probabilities(ref_logits[input_ids.size(-1)-1:-1, :], gen_ids)            
            logits, hidden_states = self._get_logits(
                seq_input_ids,
                attention_mask=seq_attention_mask,
                model=self.actor_model.module,
                return_hidden_states=True,
            )
            log_probs = self._gather_log_probabilities(logits[input_ids.size(-1)-1:-1, :], gen_ids)
            embs = hidden_states[input_ids.size(-1):]
            
            raw_results.append((gen_ids, (log_probs, ref_log_probs), embs))
        
        seq_ids, seq_embs = [], []
        available_indexes, idx = [], -1
        for ids, _, embs in raw_results:
            idx += 1
            if ids.tolist() in seq_ids:
                continue
            if not len(seq_ids):
                seq_ids.append(ids.tolist())
                seq_embs.append(embs)
                available_indexes.append(idx)
                continue
            key_embs, key_masks, key_idfs = get_embs_masks(seq_ids, seq_embs)
            _, _, F_scores = check_match(key_embs, key_masks, key_idfs, ids, embs)
            if F_scores.max().item() > .9:  # TODO: magic number
                continue
            seq_ids.append(ids.tolist())
            seq_embs.append(embs)
            available_indexes.append(idx)
        results = [raw_results[idx] for idx in available_indexes]
        
        def _eval(eval_prompt, generated):
            correct_token_ids = [self.tokenizer.encode(tok)[1] for tok in ['A', 'correct', 'Correct']]
            correct_token_ids += [self.tokenizer.encode(tok)[2] for tok in ['(A']]
            correct_token_ids = list(set(correct_token_ids))
            eval_inputs = self.tokenizer(
                eval_prompt,
                add_special_tokens=True,
                truncation=TruncationStrategy.LONGEST_FIRST,
                return_tensors='pt',
            )
            eval_sequences = self.actor_model.module.generate(
                input_ids=eval_inputs['input_ids'].to(input_ids.device),
                attention_mask=eval_inputs['attention_mask'].to(input_ids.device),
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                output_scores=True,
                synced_gpus=True,
                return_dict_in_generate=True,
            )
            eval_sequences, scores = eval_sequences.sequences.cpu(), eval_sequences.scores
            seq = eval_sequences[0][eval_inputs['input_ids'].size(-1):]
            response = self.tokenizer.decode(seq, skip_special_tokens=True)
            conf, correct = 0.0, False
            for idx, _id in enumerate(seq):
                if self.tokenizer.decode(_id).strip() in ['A', 'B', 'correct', 'wrong', 'incorrect']:
                    logprobs = F.log_softmax(scores[idx][0], dim=-1)
                    conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                    break
            if conf == 0:
                for idx, _id in enumerate(seq):
                    if self.tokenizer.decode(_id).strip() in ['Cor', 'In', 'A', 'B', 'correct', 'wrong', 'incorrect']:
                        logprobs = F.log_softmax(scores[idx][0], dim=-1)
                        conf = sum(torch.exp(logprobs[tok_id]).detach().item() for tok_id in correct_token_ids)
                        break
            # correct = math_equal(extract_answer(generated), gt_answer)
            correct = csr_equal(generated, (f'({gt_answer})', ''))
            # return self.tokenizer.decode(seq, skip_special_tokens=True), conf, correct
            return '', conf, correct
        
        prompt = self.tokenizer.decode(input_ids[0])
        eval_results = []
        for gen_ids, (log_probs, ref_log_probs), _ in results:
            # kl_divergence_estimate = -self.kl_coeff * (log_probs - ref_log_probs)
            step = self.tokenizer.decode(gen_ids)
            input_txt = prompt.split(PROMPT_ASSISTANT)[0]
            gt_solution = f'The answer is ({gt_answer})'
            # eval_prompt = HINTED_EVAL_PROMPT.format(input=input_txt, solution=gt_solution, prompt=' ' + step)
            eval_prompt = HINTED_EVAL_PROMPT.format(input=input_txt, solution=solution, prompt=' ' + step)
            eval_result, eval_conf, eval_correct = _eval(eval_prompt, step)
            score = eval_conf + eval_correct
            eval_results.append((gen_ids, score, None))
        # eval_results.sort(key=lambda x: x[2].mean().detach().item() + x[1])
        eval_results.sort(key=lambda x: x[1])
        
        # if len(eval_results) < 1 or min(x[1] for x in eval_results) >= 1:
        #     return [{}]
        if len(eval_results) < 2:
            return [{}]
        eval_results = eval_results[:1] + eval_results[-1:]
        
        # if max(x[1] for x in eval_results) < 1 or len(eval_results) < 2:
        #     soln_ids = self.tokenizer(
        #         solution, 
        #         add_special_tokens=True, 
        #         truncation=TruncationStrategy.LONGEST_FIRST,
        #         return_tensors='pt',
        #     )['input_ids'][0][1:].to(input_ids.device)
        #     soln_ids = torch.cat([input_ids[0], soln_ids], dim=-1)
        #     soln_ids = torch.cat([soln_ids, torch.LongTensor([self.tokenizer.eos_token_id]).to(soln_ids.device)], dim=-1)
        #     gen_ids = torch.cat([input_ids[0], eval_results[-1][0]], dim=-1)
        #     input_ids_list = pad_sequence(
        #         [gen_ids, soln_ids], 
        #         batch_first=True, 
        #         padding_value=self.tokenizer.pad_token_id
        #     )
        # else:
        #     input_ids_list = pad_sequence(
        #         [torch.cat((input_ids[0], x[0]), dim=-1) for x in eval_results], 
        #         batch_first=True, 
        #         padding_value=self.tokenizer.pad_token_id,
        #     )
        input_ids_list = pad_sequence(
            [torch.cat((input_ids[0], x[0]), dim=-1) for x in eval_results], 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id,
        )
        prompts = torch.stack([input_ids[0] for _ in input_ids_list], dim=0)
        
        return [
            self.post_tree_construct(
                prompts=prompts, input_ids=input_ids_list,
                max_score=eval_results[-1][1],
            )
        ]
    
    @torch.no_grad()
    def post_tree_construct(
        self,
        prompts: torch.Tensor,
        input_ids: torch.Tensor,
        max_score: float = 0.0,
    ) -> dict[str, Any]:
        attention_mask = torch.logical_and(
            input_ids.not_equal(self.tokenizer.pad_token_id),
            input_ids.not_equal(self.tokenizer.unk_token_id),
        )
        
        return {
            'prompt': prompts,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_score': max_score,
        }

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
    
    def tsrl_step(
        self, 
        prompt: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_score: float = 0.0,
    ) -> dict[str, Any]:
        torch.cuda.empty_cache()
        sequence_log_probs = self.compute_log_probs(
            self.actor_model.module,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        torch.cuda.empty_cache()            
        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(
                self.actor_reference_model.module,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        better_input_ids, worse_input_ids = input_ids[-1], input_ids[0]
        better_attention_mask, worse_attention_mask = attention_mask[-1], attention_mask[0]
        better_sequence_log_probs, worse_sequence_log_probs = sequence_log_probs[-1], sequence_log_probs[0]
        ref_better_sequence_log_probs, ref_worse_sequence_log_probs = ref_sequence_log_probs[-1], ref_sequence_log_probs[0]
        
        better_end_index = better_attention_mask.nonzero()[-1]
        worse_end_index = worse_attention_mask.nonzero()[-1]
        diverge_index = (better_input_ids != worse_input_ids).nonzero()[0]
        assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
        assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'
        
        better_seq_slice = slice(diverge_index, better_end_index + 1)
        worse_seq_slice = slice(diverge_index, worse_end_index + 1)
        
        better_log_probs = better_sequence_log_probs[better_seq_slice].sum(dim=-1)
        worse_log_probs = worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
        ref_better_log_probs = ref_better_sequence_log_probs[better_seq_slice].sum(dim=-1)
        ref_worse_log_probs = ref_worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
        better_log_ratio = better_log_probs - ref_better_log_probs
        worse_log_ratio = worse_log_probs - ref_worse_log_probs
        
        loss = -F.logsigmoid(self.scale_coeff * (better_log_ratio - worse_log_ratio))
        better_sample_rewards = self.scale_coeff * better_log_ratio.detach()
        worse_sample_rewards = self.scale_coeff * worse_log_ratio.detach()
        
        rewards_accuracy = (
            (better_sample_rewards > worse_sample_rewards).float().mean()
        )  # size = ()
        better_sample_rewards = better_sample_rewards.mean()  # size = ()
        worse_sample_rewards = worse_sample_rewards.mean()  # size = ()
        rewards = better_sample_rewards + worse_sample_rewards  # size = ()
        rewards_margin = better_sample_rewards - worse_sample_rewards  # size = ()
        
        torch.cuda.empty_cache()
        self.actor_model.backward(loss)
        self.actor_model.step()
        
        loss = get_all_reduce_mean(loss)
        rewards = get_all_reduce_mean(rewards)
        better_sample_rewards = get_all_reduce_mean(better_sample_rewards)
        worse_sample_rewards = get_all_reduce_mean(worse_sample_rewards)
        rewards_accuracy = get_all_reduce_mean(rewards_accuracy)
        rewards_margin = get_all_reduce_mean(rewards_margin)
        
        return {
            'train/loss': loss.item(),
            'train/rewards': rewards.item(),
            'train/better_sample_rewards': better_sample_rewards.item(),
            'train/worse_sample_rewards': worse_sample_rewards.item(),
            'train/rewards_accuracy': rewards_accuracy.item(),
            'train/rewards_margin': rewards_margin.item(),
            'train/lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/max_length': input_ids.size(-1),
            'train/max_score': max_score,
        }
    