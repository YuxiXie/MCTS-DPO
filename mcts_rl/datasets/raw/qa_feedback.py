
from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from mcts_rl.datasets.base import RawDataset, RawSample
from mcts_rl.configs.constants import QA_EVAL_PROMPT, PROMPT_ASSISTANT


__all__ = [
    'QAFBDataset',
    'QAFBTrainDataset',
    'QAFBTestDataset',
]


class QAFBDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or self.PATH, split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        # question = QA_EVAL_PROMPT.format(input=question, prompt=PROMPT_ASSISTANT + f' {data["answer"]}').rstrip('ASSISTANT: ').rstrip()
        return RawSample(
            input=question,
            final_answer=data['answer'],
            final_answer_content='\n'.join([f'{r}: {e}' for r, e in zip(data['feedback']['rating'], data['feedback']['explanation'])])
        )

    def __len__(self) -> int:
        return len(self.data)


class QAFBTrainDataset(QAFBDataset):
    NAME: str = 'FeedbackQA/train'
    PATH: str = 'McGill-NLP/feedbackQA'
    SPLIT: str = 'train'


class QAFBTestDataset(QAFBDataset):
    NAME: str = 'FeedbackQA/test'
    PATH: str = 'McGill-NLP/feedbackQA'
    SPLIT: str = 'test'
