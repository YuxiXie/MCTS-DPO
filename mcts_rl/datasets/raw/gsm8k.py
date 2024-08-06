"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar
from datasets import load_dataset

from mcts_rl.datasets.base import RawDataset, RawSample
from mcts_rl.utils import extract_answer, list_to_dict, get_math_data


__all__ = [
    'GSM8KDataset',
    'GSM8KTrainDataset',
    'GSM8KTestDataset',
    'GSM8KPoTTrainDataset',
    'GSM8KPoTTestDataset',
    'GSM8KSFTTrainDataset',
    'GSM8KSFTTestDataset',
]


class GSM8KDataset(RawDataset):
    SPLIT: ClassVar[str]
    PTYPE: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self) -> None:
        if self.PTYPE != 'pot':
            self.data = load_dataset('openai/gsm8k', 'main', split=self.SPLIT, trust_remote_code=True)
        else:
            raise ValueError('Do not Support PoT for now.')
        if self.DTYPE == 'arithmo':
            gsm_dict = list_to_dict(self.data)
            arithmo_dict = list_to_dict(get_math_data(load_dataset('akjindal53244/Arithmo-Data', split=self.SPLIT)))
            arithmo = {k:v for k, v in arithmo_dict.items() if k in gsm_dict}
            self.data = [vv for v in arithmo.values() for vv in v]
            # self.data = get_arithmo_data(arithmo)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['problem'] if 'problem' in data else data['question']
        prompt = prompt + '\nWrite a Python program to solve this.' if self.PTYPE == 'pot' else prompt
        solution = data['solution'] if 'solution' in data else data['answer']
        answer = extract_answer(solution)
        if self.DTYPE == 'default':
            solution = f'{solution}\nThe answer is {answer}'
        return RawSample(
            input=prompt,
            answer=solution,
            final_answer=answer,
            final_answer_content=answer,
        )

    def __len__(self) -> int:
        return len(self.data)


class GSM8KSFTTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8KSFT/train'
    SPLIT: str = 'train'
    PTYPE: str = 'cot'
    DTYPE: str = 'arithmo'


class GSM8KSFTTestDataset(GSM8KDataset):
    NAME: str = 'GSM8KSFT/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'
    DTYPE: str = 'arithmo'


class GSM8KTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8K/train'
    SPLIT: str = 'train'
    PTYPE: str = 'cot'
    DTYPE: str = 'default'


class GSM8KTestDataset(GSM8KDataset):
    NAME: str = 'GSM8K/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'
    DTYPE: str = 'default'


class GSM8KPoTTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8KCode/train'
    SPLIT: str = 'train'
    PTYPE: str = 'pot'
    DTYPE: str = 'default'


class GSM8KPoTTestDataset(GSM8KDataset):
    NAME: str = 'GSM8KCode/test'
    SPLIT: str = 'test'
    PTYPE: str = 'pot'
    DTYPE: str = 'default'