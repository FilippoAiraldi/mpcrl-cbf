from typing import Any

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.core.cache import invalidate_caches_of
from csnlp.wrappers import ScenarioBasedMpc
from env import QuadrotorEnv as Env
from mpcrl import LearnableParametersDict, LstdQLearningAgent, optim
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler, NoScheduling


class QuadrotorEnvEnvLstdQLearningAgent(LstdQLearningAgent[cs.MX, float]):
    """A LSTD Q-learning agent for the `QuadrotorEnv` env that takes care of updating
    the previus action passed to the environment, and updating the samples of the
    disturbances affecting the prediction model at each time step.

    The disturbances are updated with the following rationale: the action value `Q(s,a)`
    and state value `V(s+)` must be computed with the same disturbance profiles (in
    order to have a meaningful TD error estimate), but the disturbances are to be
    shifted by one time step forward in the case of `V` (since it is computed for the
    next time step). In practice, the samples are drawn on `on_timestep_end`, and are
    shifted on `on_env_step`. Note that we also have to sample the disturbances at the
    start of an episode to initialize the MPC policy for the first computation of `V`.
    """

    def __init__(self, mpc: ScenarioBasedMpc[cs.MX], *args: Any, **kwargs: Any) -> None:
        super().__init__(mpc, *args, **kwargs)
        self._scenarios = mpc.n_scenarios
        self._horizon = mpc.prediction_horizon
        self._dist_names = [f"w__{i}" for i in range(self._scenarios)]

    def _post_setup_V_and_Q(self) -> None:
        # since Q's first action is constrained, and it is noisy due to exploration, we
        # remove 3 constraints from Q that cause the MPC to become infeasible: max_tilt,
        # min_dtilt, and max_dtilt (see scmpc.py to understand these constraints)
        super()._post_setup_V_and_Q()
        nlp = self.Q.nlp
        nlp.remove_constraints("max_tilt", (0, 0))
        nlp.remove_constraints("min_dtilt", [(0, 0), (1, 0)])
        nlp.remove_constraints("max_dtilt", [(0, 0), (1, 0)])

        # invalidate caches for Q since some modifications have been done
        nlp_ = self.Q
        nlp_unwrapped = nlp.unwrapped
        while nlp_ is not nlp_unwrapped:
            invalidate_caches_of(nlp_)
            nlp_ = nlp_.nlp
        invalidate_caches_of(nlp_unwrapped)

    def on_episode_start(self, env: Env, episode: int, state: np.ndarray) -> None:
        super().on_episode_start(env, episode, state)
        env = env.unwrapped
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
        self._disturbances = env.sample_disturbance_profiles(
            self._scenarios, self._horizon + 1
        )

    def _shift_disturbances(self, shift: bool) -> None:
        """Updates the disturbance estimates."""
        dist = self._disturbances[:, 1:] if shift else self._disturbances[:, :-1]
        self.fixed_parameters.update(zip(self._dist_names, dist.mT))

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
    eps_value, eps_decay = exploration_epsilon
    str_value, str_decay = exploration_strength
    if eps_value <= 0.0 or np.all(str_value <= 0.0):
        exploration = None
    else:
        eps = (
            ExponentialScheduler(eps_value, eps_decay)
            if eps_decay != 1.0
            else NoScheduling(eps_value)
        )
        str = (
            ExponentialScheduler(str_value, str_decay)
            if str_decay != 1.0
            else NoScheduling(str_value)
        )
        exploration = EpsilonGreedyExploration(eps, str)
    return QuadrotorEnvEnvLstdQLearningAgent(
        mpc=scmpc,
        discount_factor=1.0,
        update_strategy=batch_size,
        experience=batch_size,
        optimizer=optim.RMSprop(learning_rate, weight_decay=1e-3),
        hessian_type="none",
        learnable_parameters=learnable_parameters,
        fixed_parameters={},
        exploration=exploration,
        record_td_errors=True,
        remove_bounds_on_initial_action=True,
        name=name,
    )
