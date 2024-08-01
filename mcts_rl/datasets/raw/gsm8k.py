"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'GSM8KDataset',
    'GSM8KTrainDataset',
    'GSM8KTestDataset',
    'GSM8KPoTTrainDataset',
    'GSM8KPoTTestDataset',
]

DATA_DIR = "path_to_dataset_folder"


class GSM8KDataset(RawDataset):
    SPLIT: ClassVar[str]
    PTYPE: ClassVar[str]

    def __init__(self) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'gsm8k/gsm8k_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['question'] + ' Write a Python program to solve this.' if self.PTYPE == 'pot' else data['question']
        return RawSample(
            input=prompt,
            answer=data['solution'],
            final_answer=data.get('answer', None),
            final_answer_content=data.get('answer', None),
        )

    def __len__(self) -> int:
        return len(self.data)


class GSM8KTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8K/train'
    SPLIT: str = 'train'
    PTYPE: str = 'cot'


class GSM8KTestDataset(GSM8KDataset):
    NAME: str = 'GSM8K/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'


class GSM8KPoTTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8KCode/train'
    SPLIT: str = 'train'
    PTYPE: str = 'pot'


class GSM8KPoTTestDataset(GSM8KDataset):
    NAME: str = 'GSM8KCode/test'
    SPLIT: str = 'test'
    PTYPE: str = 'pot'