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

from __future__ import annotations

import random
random.seed(0)

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number

from mcts_rl.configs import (
    DEFAULT_EOS_TOKEN,
    PROMPT_ASSISTANT, PROMPT_ASSISTANT_MCQ, 
    PROMPT_BEGIN, PROMPT_USER, 
    COT_INSTRUCTIONS, 
    MATH_PROMPT, 
    GSM8K_PROMPT, GSM8K_EXP,
    SQA_PROMPT,
)


def format_prompt(
    input: str | list[str],  # pylint: disable=redefined-builtin
    eos_token: str,
    use_mcq: bool = False,
    few_shot: bool = False,
) -> str:
    if isinstance(input, str):
        input = [input]
    elif not isinstance(input, list):
        raise ValueError(f'Unsupported type of `input`: {type(input)}. Expected: str or list[str].')

    if len(input) % 2 != 1:
        raise ValueError(
            'The length of `input` must be odd, while `input` must end at the user question.',
        )
    
    if 'USER:' in PROMPT_USER:
        buffer = [PROMPT_BEGIN]
    elif few_shot:
        # buffer = [GSM8K_PROMPT]
        buffer = [SQA_PROMPT]
    else:
        # exp = random.choice(GSM8K_EXP)
        # buffer = [PROMPT_USER.format(input=exp['Q']) + PROMPT_ASSISTANT + ' ' + exp['A'] + DEFAULT_EOS_TOKEN + '\n\n']
        # buffer = ['At the end of your answer output #### {final answer}.\n\n']
        buffer = []
    
    for i, line in enumerate(input):
        if i % 2 == 0:
            # User input
            buffer.extend((PROMPT_USER.format(input=line), 
                           PROMPT_ASSISTANT_MCQ if use_mcq else PROMPT_ASSISTANT))
        else:
            # Assistant response
            buffer.extend((line, eos_token))
    return ''.join(buffer)


def right_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
    ).flip(1)
