### Base component for a agent
#
#
###



import abc
from typing import Sequence, TypeVar, List, Optional, Iterator

T = TypeVar('T')

class Actor(abc.ABC):
    """Interface for an agent that can act."""

    @abc.abstractmethod
    def select_action(self, observation) -> T:
        """Samples from the policy and returns an action."""
        pass

    @abc.abstractmethod
    def observe_first(self, timestep):
        """Make a first observation from the environment."""
        pass

    @abc.abstractmethod
    def observe(self, action: T, next_timestep):
        """Make an observation of timestep data from the environment."""
        pass

    @abc.abstractmethod
    def update(self, wait: bool = False):
        """Perform an update of the actor parameters from past observations."""
        pass


class VariableSource(abc.ABC):
    """Abstract source of variables."""

    @abc.abstractmethod
    def get_variables(self, names: Sequence[str]) -> List[T]:
        """Return the named variables as a collection."""
        pass


class Worker(abc.ABC):
    """An interface for workers."""

    @abc.abstractmethod
    def run(self):
        """Runs the worker."""
        pass


class Saveable(abc.ABC, Generic[T]):
    """An interface for saveable objects."""

    @abc.abstractmethod
    def save(self) -> T:
        """Returns the state from the object to be saved."""
        pass

    @abc.abstractmethod
    def restore(self, state: T):
        """Given the state, restores the object."""
        pass


class Learner(VariableSource, Worker, Saveable):
    """Abstract learner object."""

    @abc.abstractmethod
    def step(self):
        """Perform an update step of the learner's parameters."""
        pass

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""
        iterator = range(num_steps) if num_steps is not None else iter(int, 1)

        for _ in iterator:
            self.step()

    def save(self):
        raise NotImplementedError('Method "save" is not implemented.')

    def restore(self, state):
        raise NotImplementedError('Method "restore" is not implemented.')


class PrefetchingIterator(Iterator[T], abc.ABC):
    """Abstract iterator object which supports prefetching."""

    @abc.abstractmethod
    def ready(self) -> bool:
        """Is there any data waiting for processing."""
        pass

    @abc.abstractmethod
    def retrieved_elements(self) -> int:
        """How many elements were retrieved from the iterator."""
        pass
