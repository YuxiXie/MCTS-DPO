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
    'MathQADataset',
    'MathQATrainDataset',
    'MathQATestDataset',
]

DATA_DIR = "/home/users/nus/e0672129/scratch"


class MathQADataset(RawDataset):
    SPLIT: ClassVar[str]

    def __init__(self) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'gsm8k/gsm8k_{self.SPLIT}.jsonl')) \
            + jsonlines_load(os.path.join(DATA_DIR, f'math/math_{self.SPLIT}.jsonl')) \

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['question'] if 'question' in data else data['problem']
        return RawSample(
            input=prompt,
            answer=data['solution'],
            final_answer=data.get('answer', None),
            final_answer_content=data.get('answer', None),
        )

    def __len__(self) -> int:
        return len(self.data)


class MathQATrainDataset(MathQADataset):
    NAME: str = 'MathQA/train'
    SPLIT: str = 'train'


class MathQATestDataset(MathQADataset):
    NAME: str = 'MathQA/test'
    SPLIT: str = 'test'
