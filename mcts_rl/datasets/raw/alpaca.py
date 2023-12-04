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

from datasets import load_dataset
from mcts_rl.datasets.base import RawDataset, RawSample


__all__ = ['AlpacaDataset']


class AlpacaDataset(RawDataset):
    NAME: str = 'alpaca'
    ALIASES: tuple[str, ...] = ('stanford-alpaca',)

    def __init__(self, path: str | None = None) -> None:
        alpaca = load_dataset(path or 'tatsu-lab/alpaca', split='train')
        self.data = []
        # for data in load_dataset('McGill-NLP/feedbackQA', split='train'):
        #     question = data['question']
        #     answer = data['answer'].replace('\n', ' ')
        #     comment = '\n'.join([f'{r}: {e}' for r, e in zip(data['feedback']['rating'], data['feedback']['explanation'])])
        #     if ('Excellent' in comment or 'Acceptable' in comment) and 'Bad' not in comment:
        #         self.data.append({'instruction': question, 'input': '', 'output': answer})
        # self.data += list(alpaca)[:len(self.data) * 7]
        safe_data = {}
        for data in load_dataset('PKU-Alignment/PKU-SafeRLHF', split='train'):
            question = data['prompt']
            idx = data['better_response_id']
            if data[f'is_response_{idx}_safe']:
                answer = data[f'response_{idx}']
                if question not in safe_data or data['safer_response_id'] == idx:
                    safe_data[question] = {'instruction': question, 'input': '', 'output': answer}
        self.data = list(safe_data.values()) + list(alpaca)[:len(self.data) * 7]

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = (  # pylint: disable=redefined-builtin
            ' '.join((data['instruction'], data['input'])) if data['input'] else data['instruction']
        )
        answer = data['output']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
