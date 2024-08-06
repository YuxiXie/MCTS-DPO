"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar
from datasets import load_dataset

from mcts_rl.datasets.base import RawDataset, RawSample
from mcts_rl.utils import extract_answer


__all__ = [
    'GSM8KDataset',
    'GSM8KTrainDataset',
    'GSM8KTestDataset',
    'GSM8KPoTTrainDataset',
    'GSM8KPoTTestDataset',
]


class GSM8KDataset(RawDataset):
    SPLIT: ClassVar[str]
    PTYPE: ClassVar[str]

    def __init__(self) -> None:
        if self.PTYPE != 'pot':
            self.data = load_dataset('openai/gsm8k', 'main', split=self.SPLIT, trust_remote_code=True)
        else:
            raise ValueError('Do not Support PoT for now.')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['question'] + ' Write a Python program to solve this.' if self.PTYPE == 'pot' else data['question']
        answer = extract_answer(data['answer'])
        solution = f'{data["answer"]}\nThe answer is {answer}'
        return RawSample(
            input=prompt,
            answer=solution,
            final_answer=answer,
            final_answer_content=answer,
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