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
"""CSR Datasets"""

from __future__ import annotations

import os
import random
from string import punctuation
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load
from mcts_rl.configs.constants import COT_INSTRUCTIONS, ANSWER_HINT_COT


__all__ = ['MCQDataset']

DATA_DIR = "/mnt/data/yuxi/CSR"


class MCQDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]
    
    def __init__(self, path: str | None = None) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'mcq_{self.DTYPE}_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        return RawSample(input=question, final_answer=data['answer'], 
                         final_answer_content=data.get('answer_content', data['answer']))

    def __len__(self) -> int:
        return len(self.data)


class SQATrainDataset(MCQDataset):
    NAME: str = 'SQA/train'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class CSRTrainDataset(MCQDataset):
    NAME: str = 'CSR/train'
    DTYPE: str = 'csr'
    SPLIT: str = 'train'


class SQATestDataset(MCQDataset):
    NAME: str = 'SQA/test'
    DTYPE: str = 'sqa'
    SPLIT: str = 'test'


class CSRTestDataset(MCQDataset):
    NAME: str = 'CSR/test'
    DTYPE: str = 'csr'
    SPLIT: str = 'test'
