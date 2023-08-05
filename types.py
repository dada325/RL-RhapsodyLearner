
from typing import NamedTuple
from abc import ABC, abstractmethod


class Experience(NamedTuple):
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class Agent(ABC):
    @abstractmethod
    def act(self, state: State, explore: bool = True) -> Action:
        pass

    @abstractmethod
    def learn(self, batch: Batch):
        pass
