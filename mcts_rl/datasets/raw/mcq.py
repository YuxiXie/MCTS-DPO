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

DATA_DIR = "/home/users/nus/e0672129/scratch/csr"


class MCQDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]
    
    def __init__(self, path: str | None = None) -> None:
        if self.DTYPE == 'all':
            # self.data = jsonlines_load(os.path.join(DATA_DIR, f'mcq_sqa_{self.SPLIT}.jsonl')) + \
            #     jsonlines_load(os.path.join(DATA_DIR, f'mcq_csr_{self.SPLIT}.jsonl'))
            self.data = jsonlines_load(os.path.join(DATA_DIR, f'mcq_{self.SPLIT}.jsonl'))
        else:
            self.data = jsonlines_load(os.path.join(DATA_DIR, f'mcq_{self.DTYPE}_{self.SPLIT}.jsonl'))
        
        data = []
        if self.SPLIT.count('test'):
            for x in self.data:
                if self.DTYPE == 'all':
                    if x['label'] not in ['arc_hard', 'ai2s_mid', 'sciq']: continue
                # if x['label'] not in ['arc_hard', 'sciq', 'csqa']: continue
                data.append(x)
            self.data = data
            if self.DTYPE == 'csqa':
                self.data = data[:len(data) // 2]
        # else:
        #     for x in self.data:
        #         if x['label'] not in ['openbook', 'csqa']: continue
        #         data.append(x)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        # if self.DTYPE == 'all':
        #     answer = f'The answer is ({data["answer"]}) {data["answer_content"]}'
        #     return RawSample(input=question, answer=answer)
        return RawSample(input=question, final_answer=data['answer'], 
                         final_answer_content=data.get('answer_content', data['answer']))

    def __len__(self) -> int:
        return len(self.data)


class MCQTrainDataset(MCQDataset):
    NAME: str = 'MCQ/train'
    DTYPE: str = 'all'
    SPLIT: str = 'train'


class MCQTestDataset(MCQDataset):
    NAME: str = 'MCQ/test'
    DTYPE: str = 'all'
    SPLIT: str = 'test'


class SQATrainDataset(MCQDataset):
    NAME: str = 'SQA/train'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class CSRTrainDataset(MCQDataset):
    NAME: str = 'CSR/train'
    DTYPE: str = 'csqa'
    SPLIT: str = 'train'


class SQATestDataset(MCQDataset):
    NAME: str = 'SQA/test'
    DTYPE: str = 'sqa'
    SPLIT: str = 'fulltest'


class CSRTestDataset(MCQDataset):
    NAME: str = 'CSR/test'
    DTYPE: str = 'csqa'
    SPLIT: str = 'test'


class SciQTestDataset(MCQDataset):
    NAME: str = 'SciQ/test'
    DTYPE: str = 'sciq'
    SPLIT: str = 'test'


class NLITestDataset(MCQDataset):
    NAME: str = 'NLI/test'
    DTYPE: str = 'nli'
    SPLIT: str = 'test'
