
from __future__ import annotations

import os
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'MCQPreferenceDataset',
]

DATA_DIR = "path_to_dataset_folder"


class MCQPreferenceDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'{self.DTYPE}_pairs_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'].replace('QUESTION: ', ''),
            answer=f"\n{data['response_0']}",
            other_answer=f"\n{data['response_1']}",
            better=True,
            is_safe=bool(data['is_response_0_correct']),
            is_other_safe=bool(data['is_response_1_correct']),
        )

    def __len__(self) -> int:
        return len(self.data)


class SQAPreferenceTrainDataset(MCQPreferenceDataset):
    NAME: str = 'SQAPreference/train'
    DTYPE: str = 'sqa_all'
    SPLIT: str = 'train'


class SQAPreferenceTestDataset(MCQPreferenceDataset):
    NAME: str = 'SQAPreference/test'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class CSRPreferenceTrainDataset(MCQPreferenceDataset):
    NAME: str = 'CSRPreference/train'
    DTYPE: str = 'csr_all'
    SPLIT: str = 'train'


class CSRPreferenceTestDataset(MCQPreferenceDataset):
    NAME: str = 'CSRPreference/test'
    DTYPE: str = 'csr'
    SPLIT: str = 'test'


class GSMPreferenceTrainDataset(MCQPreferenceDataset):
    NAME: str = 'GSMPreference/train'
    DTYPE: str = 'gsm'
    SPLIT: str = 'train'


class GSMPreferenceTestDataset(MCQPreferenceDataset):
    NAME: str = 'GSMPreference/test'
    DTYPE: str = 'gsm'
    SPLIT: str = 'test'