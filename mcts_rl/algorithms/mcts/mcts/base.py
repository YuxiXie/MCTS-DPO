from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional
from abc import ABC, abstractmethod

import numpy as np
from transformers import StoppingCriteriaList

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]
Args = TypeVar("Args")


class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State: ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    # @abstractmethod
    # def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
    #     return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...
    
    @abstractmethod
    def get_values(self, state: State, action: Action) -> list[tuple[float, bool]]: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class HasTerminalStateAndTrace(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> HasTerminalStateAndTrace: ...


class TreeConstructor(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Example, node=None, **kwargs) -> HasTerminalStateAndTrace[State]:
        self.world_model.update_example(example)
        self.search_config.update_example(example)
        return self.search_algo(self.world_model, 
                                self.search_config, 
                                root_node=node, 
                                **kwargs)
