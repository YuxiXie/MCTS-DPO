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
    'MATHDataset',
    'MATHTrainDataset',
    'MATHTestDataset',
]

DATA_DIR = "/mnt/data/yuxi/math/MATH"


class MATHDataset(RawDataset):
    SPLIT: ClassVar[str]

    def __init__(self) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'math_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['problem'],
            answer=data['solution'],
            final_answer=data.get('answer', None),
            final_answer_content=data.get('answer', None),
        )

    def __len__(self) -> int:
        return len(self.data)


class MATHTrainDataset(MATHDataset):
    NAME: str = 'MATH/train'
    SPLIT: str = 'train'


class MATHTestDataset(MATHDataset):
    NAME: str = 'MATH/test'
    SPLIT: str = 'test'
