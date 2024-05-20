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

from typing import Callable, Hashable
from typing_extensions import TypedDict  # Python 3.10+

import torch
from torch.utils.data import Dataset, Subset

from mcts_rl.datasets.base import CollatorBase, RawSample, RawSamplePost, TokenizedDataset
from mcts_rl.datasets.utils import format_prompt, left_padding


__all__ = [
    'PromptOnlyDataset',
    'PromptOnlyCollator',
    'PromptOnlySample',
    'PromptOnlyBatch',
    'PromptOnlyPostDataset',
    'PromptOnlyPostCollator',
    'PromptOnlyPostSample',
    'PromptOnlyPostBatch',
]


class PromptOnlySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)


class PromptOnlyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


class PromptOnlyDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PromptOnlySample:
        try:
            prompt = format_prompt(input=raw_sample['input'], eos_token=self.tokenizer.eos_token, 
                                   use_mcq=self.use_mcq, few_shot=self.few_shot, model_type=self.model_type)
        except:
            import ipdb; ipdb.set_trace()
        input_ids = self.tokenize(prompt)
        return {
            'input_ids': input_ids,  # size = (L,)
            'answer': raw_sample.get('final_answer', ''),     # str
            'reasoning': raw_sample.get('answer', ''),
            'answer_content': raw_sample.get('final_answer_content', raw_sample['final_answer'] if 'final_answer' in raw_sample else '')
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyCollator(self.tokenizer.pad_token_id)

    def _merge_raw_datasets(self, seed: int | None = None) -> Dataset[RawSample]:
        """Merge multiple raw datasets into one dataset and remove duplicates."""

        def to_hashable(raw_sample: RawSample) -> Hashable:
            input = raw_sample['input']  # pylint: disable=redefined-builtin
            return input if isinstance(input, str) else tuple(input)

        merged = super()._merge_raw_datasets(seed)
        inputs = {to_hashable(merged[i]): i for i in range(len(merged)) if isinstance(merged[i]['input'], str) or len(merged[i]['input']) == 1}
        return Subset(merged, sorted(inputs.values()))


class PromptOnlyCollator(CollatorBase):
    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        input_ids = [sample['input_ids'] for sample in samples]
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]

        input_ids = left_padding(input_ids, padding_value=self.pad_token_id)
        attention_mask = left_padding(attention_mask, padding_value=0)
        return {
            'input_ids': input_ids,  # size = (B, L)
            'attention_mask': attention_mask,  # size = (B, L)
            'answer': [sample['answer'] for sample in samples], 
            'reasoning': [sample['reasoning'] for sample in samples], 
            'answer_content': [sample['answer_content'] for sample in samples], 
        }


class PromptOnlyPostSample(TypedDict, total=True):
    prompts_list: list[torch.LongTensor]
    input_ids_list: list[torch.LongTensor]
    answer: str
    answer_content: str


class PromptOnlyPostBatch(TypedDict, total=True):
    prompts_list: list[torch.LongTensor]
    input_ids_list: list[torch.LongTensor]
    answer: list[str]
    answer_content: list[str]


class PromptOnlyPostDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSamplePost) -> PromptOnlyPostSample:
        return {
            'prompts_list': [raw_sample['prompt']],
            'input_ids_list': raw_sample['input_ids_list'],
            'answer': raw_sample['final_answer'],     # str
            'answer_content': raw_sample.get('final_answer_content', raw_sample['final_answer'])
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyPostCollator(self.tokenizer.pad_token_id)

    def _merge_raw_datasets(self, seed: int | None = None) -> Dataset[RawSamplePost]:
        """Merge multiple raw datasets into one dataset and remove duplicates."""

        def to_hashable(raw_sample: RawSamplePost) -> Hashable:
            input = raw_sample['prompt']  # pylint: disable=redefined-builtin
            return input if isinstance(input, str) else tuple(input.tolist())

        merged = super()._merge_raw_datasets(seed)
        inputs = {to_hashable(merged[i]): i for i in range(len(merged))}
        return Subset(merged, sorted(inputs.values()))


class PromptOnlyPostCollator(CollatorBase):
    def __call__(self, samples: list[PromptOnlyPostSample]) -> PromptOnlyPostBatch:
        prompts_list = [sample['prompts_list'] for sample in samples]
        input_ids_list = [sample['input_ids_list'] for sample in samples]
        attention_mask_list = [[
            input_ids.not_equal(self.pad_token_id) for input_ids in sample['input_ids_list']
        ] for sample in samples]
        
        return {
            'prompts_list': prompts_list,
            'input_ids_list': input_ids_list,
            'attention_mask_list': attention_mask_list,
            'answer': [sample['answer'] for sample in samples], 
            'answer_content': [sample['answer_content'] for sample in samples], 
        }
