### Base component for a agent
#
#This Interface define the 'Actor' and 'Learner'
#
###



import abc
from typing import Any

class Actor(abc.ABC):
  """Interface for an agent that can interact with an environment."""

  @abc.abstractmethod
  def select_action(self, observation: Any) -> Any:
    """Selects an action based on the current policy and observation."""

  @abc.abstractmethod
  def observe(self, action: Any, reward: Any, next_observation: Any):
    """Observes the result of taking an action in the environment."""

class Learner(abc.ABC):
  """Interface for an agent that can learn from experiences."""

  @abc.abstractmethod
  def learn(self):
    """Updates the policy based on the experiences collected by the actor."""
