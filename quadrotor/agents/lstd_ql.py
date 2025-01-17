from typing import Any

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import ScenarioBasedMpc
from env import QuadrotorEnv as Env
from mpcrl import LearnableParametersDict, LstdQLearningAgent, optim
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler


class QuadrotorEnvEnvLstdQLearningAgent(LstdQLearningAgent[cs.MX, float]):
    """A LSTD Q-learning agent for the `QuadrotorEnv` env that takes care of updating
    the positions and directions of the environment obstacles, updating the previus
    action passed to the environment, and updating the samples of the disturbances
    affecting the prediction model at each time step.

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
        self._dist_names = [f"w__{i}" for i in range(self._scenarios)]

    def on_episode_start(self, env: Env, episode: int, state: np.ndarray) -> None:
        super().on_episode_start(env, episode, state)
        env = env.unwrapped
        self._set_obstacles(env)
        self._set_previous_action(env)
        self._sample_disturbances(env)
        self._shift_disturbances(False)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        super().on_timestep_end(env, episode, timestep)
        self._sample_disturbances(env.unwrapped)
        self._shift_disturbances(False)

    def on_env_step(self, env: Env, episode: int, timestep: int) -> None:
        super().on_env_step(env, episode, timestep)
        self._set_previous_action(env.unwrapped)  # incorrect if in on_timestep_end
        self._shift_disturbances(True)

    def _sample_disturbances(self, env: Env) -> None:
        """Draws the disturbance samples for the time step."""
        self._disturbances = env.sample_action_disturbance_profiles(
            self._scenarios, self._horizon + 1
        )

    def _shift_disturbances(self, shift: bool) -> None:
        """Updates the disturbance estimates."""
        dist = self._disturbances[:, 1:] if shift else self._disturbances[:, :-1]
        self.fixed_parameters.update(zip(self._dist_names, dist.mT))

    def _set_obstacles(self, env: Env) -> None:
        """Updates the obstacles positions and directions."""
        self.fixed_parameters["pos_obs"] = env.pos_obs
        self.fixed_parameters["dir_obs"] = env.dir_obs

    def _set_previous_action(self, env: Env) -> None:
        """Updates the previous action."""
        self.fixed_parameters["u_prev"] = env.previous_action


def get_lstd_qlearning_agent(
    scmpc: ScenarioBasedMpc[cs.MX],
    learnable_parameters: LearnableParametersDict,
    batch_size: int,
    learning_rate: float,
    exploration_epsilon: tuple[float, float],
    exploration_strength: tuple[npt.NDArray[np.floating], float],
    name: str,
) -> QuadrotorEnvEnvLstdQLearningAgent:
    exploration = (
        None
        if exploration_epsilon[0] <= 0 or np.all(exploration_strength[0] <= 0)
        else EpsilonGreedyExploration(
            epsilon=ExponentialScheduler(*exploration_epsilon),
            strength=ExponentialScheduler(*exploration_strength),
        )
    )
    return QuadrotorEnvEnvLstdQLearningAgent(
        mpc=scmpc,
        discount_factor=1.0,
        update_strategy=batch_size,
        experience=batch_size,
        optimizer=optim.RMSprop(learning_rate),
        hessian_type="none",
        learnable_parameters=learnable_parameters,
        fixed_parameters={},
        exploration=exploration,
        record_td_errors=True,
        remove_bounds_on_initial_action=True,
        name=name,
    )
