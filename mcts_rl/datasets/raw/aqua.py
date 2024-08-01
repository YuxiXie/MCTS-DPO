"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'AQuADataset',
    'AQuATestDataset',
    'AQuAPoTTestDataset',
]

DATA_DIR = "path_to_dataset_folder"


class AQuADataset(RawDataset):
    SPLIT: ClassVar[str]
    PTYPE: ClassVar[str]

    def __init__(self) -> None:
        self.data = jsonlines_load(os.path.join(DATA_DIR, f'auqa/aqua_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question'] + '\nAnswer Choices: (' + ' ('.join(data['options'])
        prompt = question + '\nWrite a Python program to solve this.' if self.PTYPE == 'pot' else question
        return RawSample(
            input=prompt,
            answer=data['rationale'],
            final_answer=data.get('correct', None),
            final_answer_content=data.get('correct', None),
        )

    def __len__(self) -> int:
        return len(self.data)


class AQuATestDataset(AQuADataset):
    NAME: str = 'AQuA/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'


class AQuAPoTTestDataset(AQuADataset):
    NAME: str = 'AQuACode/test'
    SPLIT: str = 'test'
    PTYPE: str = 'pot'