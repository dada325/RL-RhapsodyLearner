import time
from typing import List, Optional, Sequence

from core import Actor, Environment
from utils import counting, loggers, observers_lib


class MultiActorEnvironmentLoop:
    """A RL multi-actor environment loop."""

    def __init__(
            self,
            environments: List[Environment],
            actors: List[Actor],
            counter: Optional[counting.Counter] = None,
            logger: Optional[loggers.Logger] = None,
            should_update: bool = True,
            label: str = 'multi_actor_environment_loop',
            observers: Sequence[observers_lib.EnvLoopObserver] = (),
    ):
        # Internalize agents and environments.
        self._environments = environments
        self._actors = actors
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label, steps_key=self._counter.get_steps_key())
        self._should_update = should_update
        self._observers = observers

    def _run_single_episode(self, environment: Environment, actor: Actor) -> dict:
        """Run one episode for a single actor-environment pair."""
        episode_start_time = time.time()
        episode_steps = 0
        episode_return = 0

        timestep = environment.reset()
        actor.observe_first(timestep)

        while not timestep.is_last():
            episode_steps += 1
            action = actor.select_action(timestep.observation)
            next_timestep = environment.step(action)
            actor.observe(action, next_timestep)
            
            if self._should_update:
                actor.update()

            episode_return += timestep.reward
            timestep = next_timestep

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - episode_start_time)
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
        }
        result.update(counts)
        return result

    def run(self, num_episodes: Optional[int] = None):
        episode_count = 0

        while num_episodes is None or episode_count < num_episodes:
            for env, actor in zip(self._environments, self._actors):
                result = self._run_single_episode(env, actor)
                episode_count += 1
                self._logger.write(result)
