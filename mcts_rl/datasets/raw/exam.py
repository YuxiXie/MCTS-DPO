
from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from mcts_rl.datasets.base import RawDataset, RawSample


__all__ = [
    'ExamDataset',
]

class ExamDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['Question'], 
            answer=None,
        )

    def __len__(self) -> int:
        return len(self.data)


class ExamTestDataset(ExamDataset):
    NAME: str = 'Exam/test'
    PATH: str = 'keirp/hungarian_national_hs_finals_exam'
    SPLIT: str = 'test'