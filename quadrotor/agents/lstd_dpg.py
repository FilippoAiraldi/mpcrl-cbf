from typing import Any

import casadi as cs
import numpy as np
from csnlp.wrappers import ScenarioBasedMpc
from env import QuadrotorEnv as Env
from mpcrl import LearnableParametersDict, LstdDpgAgent, optim
from mpcrl.core.exploration import GreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler


class QuadrotorEnvLstdDpgAgent(LstdDpgAgent[cs.MX, float]):
    """A LSTD determinisitc policy-gradient agent for the `QuadrotorEnv` env that takes
    care of updating the positions and directions of the environment obstacles, updating
    the previus action passed to the environment, and updating the samples of the
    disturbances affecting the prediction model at each time step."""

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

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        super().on_timestep_end(env, episode, timestep)
        env = env.unwrapped
        self._sample_disturbances(env)
        self._set_previous_action(env)  # for DPG, it is ok to update prev action here

    def _sample_disturbances(self, env: Env) -> None:
        """Draws the disturbance samples for the time step."""
        dist = env.sample_disturbance_profiles(self._scenarios, self._horizon)
        self.fixed_parameters.update(zip(self._dist_names, dist.mT))

    def _set_obstacles(self, env: Env) -> None:
        """Updates the obstacles positions and directions."""
        self.fixed_parameters["pos_obs"] = env.pos_obs
        self.fixed_parameters["dir_obs"] = env.dir_obs

    def _set_previous_action(self, env: Env) -> None:
        """Updates the previous action."""
        self.fixed_parameters["u_prev"] = env.previous_action


def get_lstd_dpg_agent(
    scmpc: ScenarioBasedMpc[cs.MX],
    learnable_parameters: LearnableParametersDict,
    learning_rate: float,
    exploration_strength: tuple[float, float],
    name: str,
    *_: Any,
    **__: Any,
) -> QuadrotorEnvLstdDpgAgent:
    return QuadrotorEnvLstdDpgAgent(
        mpc=scmpc,
        discount_factor=1.0,
        update_strategy=1,
        experience=10,  # use last ten episodes
        optimizer=optim.RMSprop(learning_rate),
        learnable_parameters=learnable_parameters,
        fixed_parameters={},
        exploration=GreedyExploration(
            strength=ExponentialScheduler(*exploration_strength),
            mode="additive",
        ),
        record_policy_gradient=True,
        name=name,
    )
