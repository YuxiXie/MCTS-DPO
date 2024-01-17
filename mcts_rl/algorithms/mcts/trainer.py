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

from typing import Any

import random
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils import TruncationStrategy

from mcts_rl.datasets import (
    PromptOnlyBatch, PromptOnlyPostBatch,
)
from mcts_rl.trainers import TSRLTrainer
from mcts_rl.utils import (
    batch_retokenize,
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    math_equal,
    extract_answer,
    csr_equal,
)
from mcts_rl.configs.constants import PROMPT_ASSISTANT, PROMPT_BEGIN
from mcts_rl.algorithms.mcts.mcts import (
    StepLMWorldModel, 
    StepLMConfig, 
    SearchArgs, 
    MCTS,
    MCTSNode,
    MCTSConfig, 
    TreeConstructor,
)


class MCTSTrainer(TSRLTrainer):
    TRAINING_TYPE = 'mcts'

    def init_mcts_searcher(self) -> None:
        world_model = StepLMWorldModel(
            max_length=self.generation_config.max_length,
            base_tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )
        search_cfg = StepLMConfig(SearchArgs(
            ref_policy_model=self.actor_reference_model,
            base_tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            n_actions=self.args.n_actions,
            n_init_actions=self.args.n_init_actions,
            breadth_limit=self.args.breadth_limit,
            depth_limit=self.args.depth_limit,
            force_terminating_on_depth_limit=self.args.force_terminating_on_depth_limit,
            kl_coeff=self.args.kl_coeff,
            disable_tqdm=False,
            no_self_eval=self.args.no_self_eval,
            reward_model=self.reward_model if self.use_reward_model else None,
            reward_tokenizer=self.reward_tokenizer if self.use_reward_model else None,
            use_code=self.args.use_code,
            use_mcq=self.args.use_mcq,
            eval_mode=self.args.eval_mode,
        ))
        mcts_algo = MCTS(MCTSConfig(
            w_exp=self.args.w_exp,
            depth_limit=self.args.depth_limit,
            breadth_limit=self.args.breadth_limit,
            n_iters=self.args.n_iters,
            temperature=self.args.mcts_temperature,
            temperature_decay_ratio=self.args.mcts_temperature_decay_ratio,
            consider_diversity=(not self.args.no_consider_diversity),
        ))
        self.mcts_searcher = TreeConstructor(
            world_model=world_model, 
            search_config=search_cfg, 
            search_algo=mcts_algo,
        )
    
    @torch.no_grad()
    def tree_constructor(self, prompt_only_batch: PromptOnlyBatch | PromptOnlyPostBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch['input_ids']
        attention_mask = prompt_only_batch['attention_mask']
        answer = prompt_only_batch['answer']
        assert input_ids.size(0) == 1, '''Only support one instance per device.'''
        seq, attn_msk = input_ids[0], attention_mask[0]
        gt_answer, solution = answer[0], prompt_only_batch['reasoning'][0]
        
        self.mcts_searcher.search_algo.policy_model = self.actor_reference_model if self.args.offline else self.actor_model
        target_probs, Q_values, r_values, base_values, select_indexes = [], [], [], [], []
        cur_node = None
        while cur_node is None or not cur_node.is_terminal:
            if cur_node is not None and self.tokenizer.eos_token_id in cur_node.action:
                cur_node.is_terminal = True
                break
            # MCTS for next step
            mcts_rst = self.mcts_searcher({
                'input_ids': seq, 'attention_mask': attn_msk, 
                'answer': gt_answer, 'reasoning': solution,
                'answer_content': prompt_only_batch['answer_content'][0],
            }, node=cur_node)
            pi, cur_node = mcts_rst.next_action_pi, mcts_rst.tree_state
            target_probs.append(pi)
            Q_values.append([child.Q for child in cur_node.children])
            r_values.append([child.r for child in cur_node.children])
            base_values.append([child.value for child in cur_node.children])
            
            cur_node = cur_node.children[mcts_rst.next_action_idx]
            select_indexes.append(mcts_rst.next_action_idx)
            
            if self.args.n_actions == 1: break
        
        dist.barrier()
        
        return [
            self.post_tree_construct(
                prompt=input_ids[idx],
                target_probs=target_probs,
                Q_values=Q_values,
                r_values=r_values,
                base_values=base_values,
                select_indexes=select_indexes,
                cur_node=mcts_rst.tree_state,
                solution=(solution, gt_answer,),
            )
            for idx in range(input_ids.size(0))
        ]
    
    @torch.no_grad()
    def post_tree_construct(
        self,
        prompt: torch.Tensor,
        target_probs: list[torch.Tensor],
        Q_values: list[list[float]],
        r_values: list[list[float]],
        base_values: list[list[float]],
        select_indexes: list[int],
        cur_node: MCTSNode,
        solution: tuple = None,
    ) -> dict[str, Any]:
        exec('''import pickle\nwith open('mcts_rst_code.pkl', 'wb') as f: \n    pickle.dump(cur_node, f)''')
        
        while cur_node.depth:
            cur_node = cur_node.parent
        
        prompts, candidates, init_value_list, step_id = [], [], [], 0
        while cur_node.children:
            next_completions = []
            for child in cur_node.children:
                cur_child = child
                next_completion = [cur_child.action]
                while cur_child.children and len(cur_child.children) == 1:  # no other candidate(s)
                    cur_child = cur_child.children[0]
                    next_completion.append(cur_child.action)
                next_completions.append(torch.cat(next_completion, dim=-1))
            
            # record the scores: \pi (visiting count), Q values, advantages (relative values), base/init (absolute) values
            scores = [(q, s, r, bv) for s, q, r, bv in zip(target_probs[step_id], Q_values[step_id], r_values[step_id], base_values[step_id],)]
            _candidates = [[x[1], scores[x[0]]] for x in sorted(enumerate(next_completions), key=lambda x: scores[x[0]])]
            init_values = [x[1][1] for x in _candidates]
            _candidates = [x[0] for x in _candidates]
            prompts.append(prompt)
            candidates.append(_candidates)
            init_value_list.append(init_values)
            
            cur_node = cur_node.children[select_indexes[step_id]]
            prompt = torch.cat([prompt, cur_node.action], dim=-1)
            step_id += 1
            while cur_node.children and len(cur_node.children) == 1:  # no other candidate(s)
                cur_node = cur_node.children[0]
                prompt = torch.cat([prompt, cur_node.action], dim=-1)
                step_id += 1
        
        mini_batches = {k:[] for k in ['prompts_list', 'input_ids_list', 'attention_mask_list', 'init_value_list']}
        for prompt, next_completions, init_values in zip(prompts, candidates, init_value_list):
            prompt = torch.stack([prompt for _ in next_completions], dim=0)
            next_completions = pad_sequence(next_completions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            mini_batches['prompts_list'].append(prompt)
            input_ids = torch.cat((prompt, next_completions), dim=-1)
            mini_batches['input_ids_list'].append(input_ids)
            mini_batches['attention_mask_list'].append(torch.logical_and(
                input_ids.not_equal(self.tokenizer.pad_token_id),
                input_ids.not_equal(self.tokenizer.unk_token_id),
            ))
            mini_batches['init_value_list'].append(init_values)
        
        r = max(r_values[-1])
        is_correct = False
        if len(mini_batches['input_ids_list']):
            text = self.tokenizer.decode(input_ids[-1], skip_special_tokens=True)
            if not text.startswith(PROMPT_BEGIN):
                prediction = text.split(PROMPT_ASSISTANT)[-1]
                if self.args.use_code:
                    is_correct = math_equal(extract_answer(prediction, use_code=self.args.use_code), solution[1])
                elif not solution[0].strip():
                    is_correct = csr_equal(prediction, ('(' + solution[1].strip() + ')', ''))
                else:
                    is_correct = math_equal(extract_answer(prediction), extract_answer(f'{solution[0]}\nThe answer is {solution[1]}'))
        
        mini_batches['prediction'] = (r, is_correct,)
        return mini_batches

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
        prompts_list: list[torch.Tensor], 
        input_ids_list: list[torch.Tensor],
        attention_mask_list: list[torch.Tensor],
        prediction: tuple = None,
        init_value_list: list[float] = None,
        max_n_sample: int = 8,
    ) -> dict[str, Any]:
        losses, better_sample_rewards, worse_sample_rewards, max_lengths = [], [], [], []
        n_sample = len(input_ids_list)
        start = prompts_list[0].size(-1) - 1
        better_idx = -1
        worse_idx = 0 if self.args.choose_worst else -2
        for sample_id in range(n_sample):
            input_ids = input_ids_list[sample_id]
            attention_mask = attention_mask_list[sample_id]
            
            n_output = input_ids.size(0)
            if n_output < 2: continue
            
            if self.args.choose_random:
                worse_idx = random.choice(range(n_output - 1))
            better_input_ids, worse_input_ids = input_ids[better_idx], input_ids[worse_idx]
            better_attention_mask, worse_attention_mask = attention_mask[better_idx], attention_mask[worse_idx]
            input_ids = torch.stack([better_input_ids, worse_input_ids], dim=0)
            attention_mask = torch.stack([better_attention_mask, worse_attention_mask], dim=0)
            
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
            
            better_sequence_log_probs, worse_sequence_log_probs = sequence_log_probs[0], sequence_log_probs[-1]
            ref_better_sequence_log_probs, ref_worse_sequence_log_probs = ref_sequence_log_probs[0], ref_sequence_log_probs[-1]
            
            better_end_index = better_attention_mask.nonzero()[-1]
            worse_end_index = worse_attention_mask.nonzero()[-1]
            try:
                diverge_index = (better_input_ids != worse_input_ids).nonzero()[0]
                assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
                assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'
            except:
                continue
            
            better_seq_slice = slice(diverge_index, better_end_index + 1)
            worse_seq_slice = slice(diverge_index, worse_end_index + 1)
            
            better_log_probs = better_sequence_log_probs[better_seq_slice].sum(dim=-1)
            worse_log_probs = worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
            ref_better_log_probs = ref_better_sequence_log_probs[better_seq_slice].sum(dim=-1)
            ref_worse_log_probs = ref_worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
            better_log_ratio = better_log_probs - ref_better_log_probs
            worse_log_ratio = worse_log_probs - ref_worse_log_probs
            
            losses.append(-F.logsigmoid(self.scale_coeff * (better_log_ratio - worse_log_ratio)))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())
            
            max_lengths.append(better_attention_mask[start:].float().sum())
            max_lengths.append(worse_attention_mask[start:].float().sum())
        
        if not len(losses): return {}
        
        loss = torch.stack(losses).mean()
        max_generated_length = torch.stack(max_lengths).max()
        better_sample_rewards = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_rewards = torch.stack(worse_sample_rewards)  # size = (B,)
        rewards_accuracy = (
            (better_sample_rewards > worse_sample_rewards).float().mean()
        )  # size = ()
        better_sample_rewards = better_sample_rewards.mean()  # size = ()
        worse_sample_rewards = worse_sample_rewards.mean()  # size = ()
        rewards = better_sample_rewards + worse_sample_rewards  # size = ()
        rewards_margin = better_sample_rewards - worse_sample_rewards  # size = ()
        
        torch.cuda.empty_cache()
        flag_loss = True
        try:
            self.actor_model.backward(loss)
            self.actor_model.step()
        except Exception as e:
            print('\n{}'.format(str(e)))
            flag_loss = False
        
        loss = get_all_reduce_mean(loss)
        rewards = get_all_reduce_mean(rewards)
        better_sample_rewards = get_all_reduce_mean(better_sample_rewards)
        worse_sample_rewards = get_all_reduce_mean(worse_sample_rewards)
        rewards_accuracy = get_all_reduce_mean(rewards_accuracy)
        rewards_margin = get_all_reduce_mean(rewards_margin)
        max_generated_length = get_all_reduce_max(max_generated_length)
        
        return {
            'train/loss': loss.item(),
            'train/rewards': rewards.item(),
            'train/better_sample_rewards': better_sample_rewards.item(),
            'train/worse_sample_rewards': worse_sample_rewards.item(),
            'train/rewards_accuracy': rewards_accuracy.item(),
            'train/rewards_margin': rewards_margin.item(),
            'train/lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/r_scores': float(prediction[0]),
            'train/correct': float(prediction[1]),
            'train/n_sample': n_sample,
            'train/max_generated_length': max_generated_length.item(),
        } if flag_loss else None
    
