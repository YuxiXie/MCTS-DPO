"""CSR Datasets"""

from __future__ import annotations

import os
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = ['MCQDataset']

DATA_DIR = "path_to_dataset_folder"


class MCQDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]
    
    def __init__(self, path: str | None = None) -> None:
        if self.DTYPE == 'all':
            self.data = jsonlines_load(os.path.join(DATA_DIR, f'csr/mcq_{self.SPLIT}.jsonl'))
        else:
            self.data = jsonlines_load(os.path.join(DATA_DIR, f'csr/mcq_{self.DTYPE}_{self.SPLIT}.jsonl'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        return RawSample(input=question, final_answer=data['answer'], 
                         final_answer_content=data.get('answer_content', data['answer']))

    def __len__(self) -> int:
        return len(self.data)


class MCQTrainDataset(MCQDataset):
    NAME: str = 'MCQ/train'
    DTYPE: str = 'all'
    SPLIT: str = 'train'


class MCQTestDataset(MCQDataset):
    NAME: str = 'MCQ/test'
    DTYPE: str = 'all'
    SPLIT: str = 'test'


class SQATrainDataset(MCQDataset):
    NAME: str = 'SQA/train'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class CSRTrainDataset(MCQDataset):
    NAME: str = 'CSR/train'
    DTYPE: str = 'csqa'
    SPLIT: str = 'train'


class SQATestDataset(MCQDataset):
    NAME: str = 'SQA/test'
    DTYPE: str = 'sqa'
    SPLIT: str = 'fulltest'


class CSRTestDataset(MCQDataset):
    NAME: str = 'CSR/test'
    DTYPE: str = 'csqa'
    SPLIT: str = 'test'


class SciQTestDataset(MCQDataset):
    NAME: str = 'SciQ/test'
    DTYPE: str = 'sciq'
    SPLIT: str = 'test'


class NLITestDataset(MCQDataset):
    NAME: str = 'NLI/test'
    DTYPE: str = 'nli'
    SPLIT: str = 'test'
