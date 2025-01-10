from typing import Any

import casadi as cs
import numpy as np
from csnlp.wrappers import ScenarioBasedMpc
from env import ConstrainedLtiEnv as Env
from mpcrl import LearnableParametersDict, LstdDpgAgent, optim
from mpcrl.core.exploration import GreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler


class ConstrainedLtiEnvLstdDpgAgent(LstdDpgAgent[cs.MX, float]):
    """A LSTD determinisitc policy-gradient agent for the `ConstrainedLtiEnv` env that
    takes care of updating the samples of the disturbances affecting the prediction
    model at each time step."""

    def __init__(self, mpc: ScenarioBasedMpc[cs.MX], *args: Any, **kwargs: Any) -> None:
        super().__init__(mpc, *args, **kwargs)
        self._scenarios = mpc.n_scenarios
        self._horizon = mpc.prediction_horizon
        self._dist_names = [f"w__{i}" for i in range(self._scenarios)]

    def train(self, env: Env, *args: Any, **kwargs: Any):
        assert env.unwrapped.nd == 1, "only one disturbance is supported!"
        return super().train(env, *args, **kwargs)

    def on_episode_start(self, env: Env, episode: int, state: np.ndarray) -> None:
        super().on_episode_start(env, episode, state)
        self._sample_disturbances(env)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        super().on_timestep_end(env, episode, timestep)
        self._sample_disturbances(env)

    def _sample_disturbances(self, env: Env) -> None:
        """Draws the disturbance samples for the time step."""
        dist = env.unwrapped.sample_disturbance_profiles(self._scenarios, self._horizon)
        self.fixed_parameters.update(zip(self._dist_names, dist))


def get_lstd_dpg_agent(
    scmpc: ScenarioBasedMpc[cs.MX],
    learnable_parameters: LearnableParametersDict,
    learning_rate: float,
    exploration_strength: tuple[float, float],
    name: str,
    *_: Any,
    **__: Any,
) -> ConstrainedLtiEnvLstdDpgAgent:
    return ConstrainedLtiEnvLstdDpgAgent(
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
