from typing import Any

import casadi as cs
import numpy as np
from csnlp.wrappers import ScenarioBasedMpc
from env import ConstrainedLtiEnv as Env
from mpcrl import LearnableParametersDict, LstdQLearningAgent, optim
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler


class ConstrainedLtiEnvLstdQLearningAgent(LstdQLearningAgent[cs.MX, float]):
    """A LSTD Q-learning agent for the `ConstrainedLtiEnv` env that takes care of
    updating the samples of the disturbances affecting the prediction model at each time
    step.

    The disturbances are updated with the following rationale: the action value `Q(s,a)`
    and state value `V(s+)` functions must be computed with the same disturbance
    profiles (in order to have a meaning TD error estimate), but the disturbances are to
    be shifted by one time step forward in the case of `V` (since it is computed for the
    next time step). In practice, the samples are drawn on `on_timestep_end`, and are
    shifted on `on_env_step`. Note that we also have to sample the disturbances at the
    start of an episode to initialize the MPC policy for the first computation of `V`.
    """

    def __init__(self, mpc: ScenarioBasedMpc[cs.MX], *args: Any, **kwargs: Any) -> None:
        super().__init__(mpc, *args, **kwargs)
        self._scenarios = mpc.n_scenarios
        self._horizon = mpc.prediction_horizon

    def train(self, env: Env, *args: Any, **kwargs: Any):
        assert env.unwrapped.nd == 1, "only one disturbance is supported!"
        return super().train(env, *args, **kwargs)

    def on_episode_start(self, env: Env, episode: int, state: np.ndarray) -> None:
        super().on_episode_start(env, episode, state)
        self._sample_disturbances(env)
        self._shift_disturbances(0)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        super().on_timestep_end(env, episode, timestep)
        self._sample_disturbances(env)
        self._shift_disturbances(0)

    def on_env_step(self, env: Env, episode: int, timestep: int) -> None:
        super().on_env_step(env, episode, timestep)
        self._shift_disturbances(1)

    def _sample_disturbances(self, env: Env) -> None:
        """Draws the disturbance samples for the time step."""
        self._disturbances = env.unwrapped.sample_disturbance_profiles(self._scenarios)

    def _shift_disturbances(self, shift: int) -> None:
        """Updates the disturbance estimates."""
        dist = self._disturbances
        h = self._horizon
        pars = self.fixed_parameters
        for i in range(self._scenarios):
            pars[f"w__{i}"] = dist[i, shift : h + shift]


def get_lstd_qlearning_agent(
    scmpc: ScenarioBasedMpc[cs.MX],
    learnable_parameters: LearnableParametersDict,
    batch_size: int,
    learning_rate: float,
    exploration_epsilon: tuple[float, float],
    exploration_strength: tuple[float, float],
    name: str,
) -> ConstrainedLtiEnvLstdQLearningAgent:
    return ConstrainedLtiEnvLstdQLearningAgent(
        mpc=scmpc,
        discount_factor=1.0,
        update_strategy=batch_size,
        experience=batch_size,
        optimizer=optim.RMSprop(learning_rate),
        hessian_type="none",
        learnable_parameters=learnable_parameters,
        fixed_parameters={},
        exploration=EpsilonGreedyExploration(
            epsilon=ExponentialScheduler(*exploration_epsilon),
            strength=ExponentialScheduler(*exploration_strength),
        ),
        record_td_errors=True,
        remove_bounds_on_initial_action=True,
        name=name,
    )
