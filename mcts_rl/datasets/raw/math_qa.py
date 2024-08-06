"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar

from datasets import load_dataset
from mcts_rl.utils import get_math_data, get_arithmo_data, list_to_dict, tqdm
from mcts_rl.datasets.base import RawDataset, RawSample, jsonlines_load


__all__ = [
    'MathQADataset',
    'MathQATrainDataset',
    'MathQATestDataset',
    'MathQACodeTrainDataset',
    'MathQACodeTestDataset',
    'MathQASFTTrainDataset',
]

DATA_DIR = "path_to_dataset_folder"


class MathQADataset(RawDataset):
    SPLIT: ClassVar[str]
    TYPE: ClassVar[str]

    def __init__(self) -> None:
        if self.TYPE == 'pot':
            raise ValueError('Do not Support PoT for now.')
            ## PoT data
            gsm8k = jsonlines_load(os.path.join(DATA_DIR, f'gsm8k/gsm8k_{self.SPLIT}.jsonl'))
            raw_arithmo = jsonlines_load(os.path.join(DATA_DIR, f'arithmo/arithmo_code_{self.SPLIT}.jsonl'))
            arithmo = []
            for dt in raw_arithmo:
                prompt = dt['question'] if 'question' in dt else dt['problem']
                if all(not prompt.strip().startswith(x['question'].strip()) for x in gsm8k):
                    arithmo.append(dt)
            for i, dt in enumerate(gsm8k):
                gsm8k[i]['question'] = dt['question'] + ' Write a Python program to solve this.'
            self.data = gsm8k
        elif self.TYPE == 'all':
            raise ValueError('Do not Support PoT for now.')
            ## CoT + PoT data
            gsm8k = jsonlines_load(os.path.join(DATA_DIR, f'gsm8k/gsm8k_{self.SPLIT}.jsonl'))
            math = jsonlines_load(os.path.join(DATA_DIR, f'math/math_{self.SPLIT}.jsonl'))
            self.data = gsm8k + math
            if self.SPLIT == 'train':
                raw_arithmo = jsonlines_load(os.path.join(DATA_DIR, f'arithmo/arithmo_code_{self.SPLIT}.jsonl'))
                for dt in raw_arithmo[::-1]:
                    prompt = dt['question'] if 'question' in dt else dt['problem']
                    if any(prompt.strip().startswith(x['question'].strip()) for x in gsm8k):
                        self.data.append(dt)
            else:
                for i, dt in enumerate(gsm8k[:]):
                    dt['question'] = dt['question'] + ' Write a Python program to solve this.'
                    self.data.append(dt)
        else:
            gsm8k = load_dataset('openai/gsm8k', 'main', split=self.SPLIT, trust_remote_code=True)
            math = load_dataset('hendrycks/competition_math', split=self.SPLIT, trust_remote_code=True)
            try:
                arithmo = get_math_data(load_dataset('akjindal53244/Arithmo-Data', split=self.SPLIT))
            except:
                arithmo = get_math_data(jsonlines_load(os.path.join(DATA_DIR, 'arithmo/train.jsonl')))
            if self.TYPE == 'sft':
                arithmo, gsm8k, math = list_to_dict(arithmo), list_to_dict(gsm8k), list_to_dict(math)
                ## use the corresponding training data seen in SFT
                mathqa_dict = {k:v for k,v in arithmo.items() if k in math or k in gsm8k}
                self.data = [vv for v in mathqa_dict.values() for vv in v]
                # self.data = get_arithmo_data(mathqa_dict)
            else:
                self.data = gsm8k + math

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['question'] if 'question' in data else data['problem']
        return RawSample(
            input=prompt,
            answer=data['solution'],
            final_answer=data.get('answer', None),
            final_answer_content=data.get('answer_content', data.get('answer', None)),
        )

    def __len__(self) -> int:
        return len(self.data)


class MathQASFTTrainDataset(MathQADataset):
    NAME: str = 'MathQASFT/train'
    SPLIT: str = 'train'
    TYPE: str = 'sft'


class MathQATrainDataset(MathQADataset):
    NAME: str = 'MathQA/train'
    SPLIT: str = 'train'
    TYPE: str = 'cot'
    

class MathQAAllTrainDataset(MathQADataset):
    NAME: str = 'MathQAAll/train'
    SPLIT: str = 'train'
    TYPE: str = 'all'


class MathQAAllTestDataset(MathQADataset):
    NAME: str = 'MathQAAll/test'
    SPLIT: str = 'test'
    TYPE: str = 'all'


class MathQATestDataset(MathQADataset):
    NAME: str = 'MathQA/test'
    SPLIT: str = 'test'
    TYPE: str = 'cot'


class MathQACodeTrainDataset(MathQADataset):
    NAME: str = 'MathQACode/train'
    SPLIT: str = 'train'
    TYPE: str = 'pot'


class MathQACodeTestDataset(MathQADataset):
    NAME: str = 'MathQACode/test'
    SPLIT: str = 'test'
    TYPE: str = 'pot'