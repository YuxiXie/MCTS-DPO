
from __future__ import annotations

import os
from typing import ClassVar

from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load
from mcts_rl.configs.constants import HINTED_EVAL_PROMPT


__all__ = [
    'MCQEvalDataset',
]

DATA_DIR = "path_to_dataset_folder"


class MCQEvalDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        data = jsonlines_load(os.path.join(DATA_DIR, 'arithmo/stepwise_generations.jsonl'))
        self.data = []
        for dt in data:
            prompt = dt['question']
            solution = dt['solution']
            eval_prompt = HINTED_EVAL_PROMPT.format(
                input=f'{prompt}\n\n', 
                solution=dt['answer'], 
                prompt=solution,
            ).replace('\n\nANSWER: The answer is', '').strip()
            self.data.append({
                'question': eval_prompt,
                'answer': '',
            })
            

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['question'],
            final_answer=data['answer'],
        )

    def __len__(self) -> int:
        return len(self.data)


class SQAEvalTrainDataset(MCQEvalDataset):
    NAME: str = 'SQAEval/train'
    DTYPE: str = 'sqa_all'
    SPLIT: str = 'train'


class SQAEvalTestDataset(MCQEvalDataset):
    NAME: str = 'SQAEval/test'
    DTYPE: str = 'sqa'
    SPLIT: str = 'train'


class CSREvalTrainDataset(MCQEvalDataset):
    NAME: str = 'CSREval/train'
    DTYPE: str = 'csr_all'
    SPLIT: str = 'train'


class CSREvalTestDataset(MCQEvalDataset):
    NAME: str = 'CSREval/test'
    DTYPE: str = 'csr'
    SPLIT: str = 'test'


class GSMEvalTrainDataset(MCQEvalDataset):
    NAME: str = 'GSMEval/train'
    DTYPE: str = 'gsm_all'
    SPLIT: str = 'train'


class GSMEvalTestDataset(MCQEvalDataset):
    NAME: str = 'GSMEval/test'
    DTYPE: str = 'gsm'
    SPLIT: str = 'train'


class GSMEvalTestDataset(MCQEvalDataset):
    NAME: str = 'arithmo/test'
    DTYPE: str = 'arithmo'
    SPLIT: str = 'train'