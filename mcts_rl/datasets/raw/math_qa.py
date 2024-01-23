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
"""MATH datasets."""

from __future__ import annotations

import os
import regex
from typing import ClassVar

from datasets import load_dataset
from mcts_rl.utils import random, get_math_data
from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'MathQADataset',
    'MathQATrainDataset',
    'MathQATestDataset',
    'MathQACodeTrainDataset',
    'MathQACodeTestDataset',
]

DATA_DIR = "/home/users/nus/e0672129/scratch"


class MathQADataset(RawDataset):
    SPLIT: ClassVar[str]
    TYPE: ClassVar[str]

    def __init__(self) -> None:
        if self.TYPE == 'pot':
            gsm8k = jsonlines_load(os.path.join(DATA_DIR, f'gsm8k/gsm8k_{self.SPLIT}.jsonl'))
            raw_arithmo = jsonlines_load(os.path.join(DATA_DIR, f'arithmo/arithmo_code_{self.SPLIT}.jsonl'))
            arithmo = []
            for dt in raw_arithmo:
                prompt = dt['question'] if 'question' in dt else dt['problem']
                if all(not prompt.strip().startswith(x['question'].strip()) for x in gsm8k):
                    arithmo.append(dt)
            for i, dt in enumerate(gsm8k):
                gsm8k[i]['question'] = dt['question'] + ' Write a Python program to solve this.'
            self.data = gsm8k #+ list(random.sample(arithmo, min(len(gsm8k), len(arithmo))))
        else:
            gsm8k = jsonlines_load(os.path.join(DATA_DIR, f'gsm8k/gsm8k_{self.SPLIT}.jsonl'))
            math = jsonlines_load(os.path.join(DATA_DIR, f'math/math_{self.SPLIT}.jsonl'))
            arithmo = get_math_data(load_dataset('akjindal53244/Arithmo-Data', split=self.SPLIT))
            self.data = gsm8k + math #+ list(random.sample(arithmo, min(len(gsm8k + math), len(arithmo))))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['question'] if 'question' in data else data['problem']
        return RawSample(
            input=prompt,
            answer=data['solution'],
            final_answer=data.get('answer', None),
            final_answer_content=data.get('answer_content', data.get('answer', None)),
        )

    def __len__(self) -> int:
        return len(self.data)


class MathQATrainDataset(MathQADataset):
    NAME: str = 'MathQA/train'
    SPLIT: str = 'train'
    TYPE: str = 'cot'


class MathQATestDataset(MathQADataset):
    NAME: str = 'MathQA/test'
    SPLIT: str = 'test'
    TYPE: str = 'cot'


class MathQACodeTrainDataset(MathQADataset):
    NAME: str = 'MathQACode/train'
    SPLIT: str = 'train'
    TYPE: str = 'pot'


class MathQACodeTestDataset(MathQADataset):
    NAME: str = 'MathQACode/test'
    SPLIT: str = 'test'
    TYPE: str = 'pot'