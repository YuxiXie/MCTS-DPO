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
from mcts_rl.configs.constants import HINTED_EVAL_PROMPT


__all__ = [
    'MCQEvalDataset',
]

DATA_DIR = "/home/users/nus/e0672129/scratch/MCTS-DPO/rm-traindata/reward-model"


class MCQEvalDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        data = jsonlines_load(os.path.join(DATA_DIR, f'{self.DTYPE}_pairs_{self.SPLIT}.jsonl'))
        self.data = []
        for dt in data[:1086]:
            prompt = dt['prompt'].replace('QUESTION: ', '')
            for i in range(2):
                response = f'\n{dt[f"response_{i}"]}'
                eval_prompt = HINTED_EVAL_PROMPT.format(
                    input=f'{prompt}\n\n', 
                    solution=f"The answer is ({dt['answer'][0]}) {dt['answer'][1]}", 
                    prompt=response,
                ).replace('\n\nANSWER: The answer is', '').strip()
                
                self.data.append({
                    'question': eval_prompt,
                    'answer': ' ' + ('B' if dt[f'is_response_{i}_correct'] else 'A'),
                })

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['question'],
            final_answer=data['answer'],
        )

    def __len__(self) -> int:
        return len(self.data)


class SQAEvalTrainDataset(MCQEvalDataset):
    NAME: str = 'SQAEval/train'
    DTYPE: str = 'sqa_all'
    SPLIT: str = 'train'


class SQAEvalTestDataset(MCQEvalDataset):
    NAME: str = 'SQAEval/test'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class CSREvalTrainDataset(MCQEvalDataset):
    NAME: str = 'CSREval/train'
    DTYPE: str = 'csr_all'
    SPLIT: str = 'train'


class CSREvalTestDataset(MCQEvalDataset):
    NAME: str = 'CSREval/test'
    DTYPE: str = 'csr'
    SPLIT: str = 'test'


class GSMEvalTrainDataset(MCQEvalDataset):
    NAME: str = 'GSMEval/train'
    DTYPE: str = 'gsm_all'
    SPLIT: str = 'train'


class GSMEvalTestDataset(MCQEvalDataset):
    NAME: str = 'GSMEval/test'
    DTYPE: str = 'gsm'
    SPLIT: str = 'train'