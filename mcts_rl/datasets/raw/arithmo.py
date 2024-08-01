
from __future__ import annotations

import os
import regex
from typing import ClassVar

from datasets import load_dataset
from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'ArithmoDataset',
    'ArithmoTrainDataset',
    'ArithmoTestDataset',
    'ArithmoMATHTrainDataset',
    'ArithmoMCQTrainDataset',
    'ArithmoCodeTrainDataset',
]

DATA_DIR = "path_to_dataset_folder"

class ArithmoDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]
    TYPE: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        try:
            self.data = load_dataset(path or self.PATH, split=self.SPLIT)
        except:
            self.data = jsonlines_load(os.path.join(DATA_DIR, f'arithmo/{self.SPLIT}.jsonl'))
        if self.TYPE == 'math':
            self.data = [dt for dt in self.data if ' answer is' in dt['answer']]
        elif self.TYPE == 'mcq':
            self.data = [
                dt for dt in self.data if ' answer is' in dt['answer'] and not dt['answer'].startswith('The answer is') \
                    and 'answer choices' in dt['question'].lower()
            ]
        elif self.TYPE == 'code':
            self.data = [dt for dt in self.data if regex.search(r'print\(.+\)', dt['answer'])]

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['question'], 
            answer=data['answer'],
        )

    def __len__(self) -> int:
        return len(self.data)


class ArithmoTrainDataset(ArithmoDataset):
    NAME: str = 'Arithmo/train'
    ALIASES: tuple[str, ...] = ('akjindal53244/Arithmo-Data/train',)
    PATH: str = 'akjindal53244/Arithmo-Data'
    SPLIT: str = 'train'
    TYPE: str = 'all'


class ArithmoTestDataset(ArithmoDataset):
    NAME: str = 'Arithmo/test'
    ALIASES: tuple[str, ...] = ('akjindal53244/Arithmo-Data/test',)
    PATH: str = 'akjindal53244/Arithmo-Data'
    SPLIT: str = 'test'
    TYPE: str = 'all'


class ArithmoMATHTrainDataset(ArithmoDataset):
    NAME: str = 'ArithmoMATH/train'
    ALIASES: tuple[str, ...] = ('akjindal53244/Arithmo-Data/train/mathqa',)
    PATH: str = 'akjindal53244/Arithmo-Data'
    SPLIT: str = 'train'
    TYPE: str = 'math'


class ArithmoMCQTrainDataset(ArithmoDataset):
    NAME: str = 'ArithmoMCQ/train'
    ALIASES: tuple[str, ...] = ('akjindal53244/Arithmo-Data/train/mcq',)
    PATH: str = 'akjindal53244/Arithmo-Data'
    SPLIT: str = 'train'
    TYPE: str = 'mcq'


class ArithmoCodeTrainDataset(ArithmoDataset):
    NAME: str = 'ArithmoCode/train'
    ALIASES: tuple[str, ...] = ('akjindal53244/Arithmo-Data/train/code',)
    PATH: str = 'akjindal53244/Arithmo-Data'
    SPLIT: str = 'train'
    TYPE: str = 'code'