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
"""Stanford Alpaca dataset for supervised instruction fine-tuning."""

from __future__ import annotations

import regex
from typing import ClassVar

from datasets import load_dataset
from mcts_rl.datasets.base import RawDataset, RawSample


__all__ = [
    'ArithmoDataset',
    'ArithmoTrainDataset',
    'ArithmoTestDataset',
    'ArithmoMATHTrainDataset',
    'ArithmoMCQTrainDataset',
    'ArithmoCodeTrainDataset',
]


class ArithmoDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]
    TYPE: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)
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