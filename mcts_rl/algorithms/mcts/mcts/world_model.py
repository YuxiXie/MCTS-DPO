# Adapted from: https://github.com/maitrix-org/llm-reasoners/blob/main/examples/RAP/gsm8k/world_model.py

from typing import NamedTuple, TypedDict

import torch
from transformers import GenerationConfig, PreTrainedTokenizerBase

from mcts_rl.algorithms.mcts.mcts.base import WorldModel


class StepSubResult(NamedTuple):
    next_step_ids: torch.LongTensor
    log_probs: torch.Tensor


StepLMState = list[StepSubResult]
StepLMAction = torch.LongTensor


class WorldModelArgs(NamedTuple):
    base_tokenizer: PreTrainedTokenizerBase
    generation_config: GenerationConfig
    stop_tokens: list[str] = []


class LMExample(TypedDict):
    input_ids: torch.LongTensor     # (L,)
    attention_mask: torch.BoolTensor    # (L,)


class StepLMWorldModel(WorldModel[StepLMState, StepLMAction, LMExample]):
    def __init__(self,
                 max_length: int,
                 base_tokenizer: PreTrainedTokenizerBase,
                 generation_config: GenerationConfig,
                 stop_tokens=[]) -> None:
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.generation_config = generation_config
        self.max_tokens_num = max_length
        self.stop_tokens = list(set(
            stop_tokens + [self.base_tokenizer.decode([self.generation_config.eos_token_id])]
        ))
    
    def init_state(self) -> list:
        return []
    
    def step(self, state: StepLMState, action: StepLMAction, log_probs: torch.Tensor) -> StepLMState:
        state = state.copy()
        state.append(StepSubResult(action, log_probs))
        return state

    def is_terminal(self, state: StepLMState) -> bool:
        input_length = self.example['attention_mask'].nonzero()[-1].item() + 1
        sum_tokens_num = sum(x.next_step_ids.size(0) for x in state) + input_length
        
        if sum_tokens_num >= self.max_tokens_num - 5:
            return True
        elif state[-1].next_step_ids.eq(self.base_tokenizer.eos_token_id).sum():
            return True
        elif state[-1].next_step_ids.eq(self.base_tokenizer.convert_tokens_to_ids("<|eot_id|>")).sum():
            return True
        elif self.base_tokenizer.decode(state[-1].next_step_ids).count('QUESTION: '):
            return True
        else:
            return False
