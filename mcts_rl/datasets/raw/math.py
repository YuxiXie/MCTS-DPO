"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar
from datasets import load_dataset

from mcts_rl.datasets.base import RawDataset, RawSample
from mcts_rl.utils import extract_answer, get_math_data, list_to_dict, get_arithmo_data


__all__ = [
    'MATHDataset',
    'MATHTrainDataset',
    'MATHTestDataset',
    'MATHSFTTrainDataset',
    'MATHSFTTestDataset',
]


class MATHDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self) -> None:
        self.data = load_dataset('hendrycks/competition_math', split=self.SPLIT, trust_remote_code=True)
        if self.DTYPE == 'arithmo':
            math_dict = list_to_dict(self.data)
            arithmo_dict = list_to_dict(get_math_data(load_dataset('akjindal53244/Arithmo-Data', split=self.SPLIT)))
            arithmo = {k:v for k, v in arithmo_dict.items() if k in math_dict}
            self.data = [vv for v in arithmo.values() for vv in v]
            # self.data = get_arithmo_data(arithmo)


    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        solution = data['solution']
        answer = extract_answer(solution)
        if self.DTYPE == 'default':
            solution = f'{solution}\nThe answer is {answer}'
        return RawSample(
            input=data['problem'] if 'problem' in data else data['question'],
            answer=solution,
            final_answer=answer,
            final_answer_content=answer,
        )

    def __len__(self) -> int:
        return len(self.data)


class MATHTrainDataset(MATHDataset):
    NAME: str = 'MATH/train'
    SPLIT: str = 'train'
    DTYPE: str = 'default'


class MATHTestDataset(MATHDataset):
    NAME: str = 'MATH/test'
    SPLIT: str = 'test'
    DTYPE: str = 'default'


class MATHSFTTrainDataset(MATHDataset):
    NAME: str = 'MATHSFT/train'
    SPLIT: str = 'train'
    DTYPE: str = 'arithmo'


class MATHSFTTestDataset(MATHDataset):
    NAME: str = 'MATHSFT/test'
    SPLIT: str = 'test'
    DTYPE: str = 'arithmo'
