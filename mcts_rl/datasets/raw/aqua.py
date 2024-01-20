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
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'AQuADataset',
    'AQuATestDataset',
    'AQuAPoTTestDataset',
]

DATA_DIR = "/home/users/nus/e0672129/scratch/aqua"


class AQuADataset(RawDataset):
    SPLIT: ClassVar[str]
    PTYPE: ClassVar[str]

    def __init__(self) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'aqua_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question'] + '\nAnswer Choices: (' + ' ('.join(data['options'])
        prompt = question + '\nWrite a Python program to solve this.' if self.PTYPE == 'pot' else question
        return RawSample(
            input=prompt,
            answer=data['rationale'],
            final_answer=data.get('correct', None),
            final_answer_content=data.get('correct', None),
        )

    def __len__(self) -> int:
        return len(self.data)


class AQuATestDataset(AQuADataset):
    NAME: str = 'AQuA/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'


class AQuAPoTTestDataset(AQuADataset):
    NAME: str = 'AQuACode/test'
    SPLIT: str = 'test'
    PTYPE: str = 'pot'