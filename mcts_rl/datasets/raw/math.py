"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'MATHDataset',
    'MATHTrainDataset',
    'MATHTestDataset',
]

DATA_DIR = "path_to_dataset_folder"


class MATHDataset(RawDataset):
    SPLIT: ClassVar[str]

    def __init__(self) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'math/math_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['problem'],
            answer=data['solution'],
            final_answer=data.get('answer', None),
            final_answer_content=data.get('answer', None),
        )

    def __len__(self) -> int:
        return len(self.data)


class MATHTrainDataset(MATHDataset):
    NAME: str = 'MATH/train'
    SPLIT: str = 'train'


class MATHTestDataset(MATHDataset):
    NAME: str = 'MATH/test'
    SPLIT: str = 'test'
