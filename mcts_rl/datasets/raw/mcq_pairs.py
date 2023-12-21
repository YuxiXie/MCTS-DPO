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
"""Safe-RLHF preference datasets."""

from __future__ import annotations

import os
from typing import ClassVar

from datasets import load_dataset
from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'MCQPreferenceDataset',
]

DATA_DIR = "/mnt/data/yuxi/reward-model"


class MCQPreferenceDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'{self.DTYPE}_pairs_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'].replace('QUESTION: ', ''),
            answer=f"\n{data['response_0']}",
            other_answer=f"\n{data['response_1']}",
            better=True,
            is_safe=bool(data['is_response_0_correct']),
            is_other_safe=bool(data['is_response_1_correct']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SQAPreferenceTrainDataset(MCQPreferenceDataset):
    NAME: str = 'SQAPreference/train'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class SQAPreferenceTestDataset(MCQPreferenceDataset):
    NAME: str = 'SQAPreference/test'
    DTYPE: str = 'sqa_all'
    SPLIT: str = 'train'


class CSRPreferenceTrainDataset(MCQPreferenceDataset):
    NAME: str = 'CSRPreference/train'
    DTYPE: str = 'csr'
    SPLIT: str = 'train'


class CSRPreferenceTestDataset(MCQPreferenceDataset):
    NAME: str = 'CSRPreference/test'
    DTYPE: str = 'csr'
    SPLIT: str = 'test'
