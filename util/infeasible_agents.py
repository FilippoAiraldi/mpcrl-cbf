from typing import Any, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from gymnasium import Env

SymType = TypeVar("SymType", cs.SX, cs.MX)
from mpcrl import Agent
from mpcrl.util.seeding import RngType, mk_seed


class InfeasibleAgent(Agent[SymType]):
    """Specialized agent that handles infeasible MPC solutions during evaluation."""

    def evaluate(
        self,
        env: Env,
        episodes: int,
        deterministic: bool = True,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] | None = None,
        terminate_on_infeas: bool = True,
        penalty_on_infeas: float = 1e4,
    ) -> npt.NDArray[np.floating]:
        r"""Evaluates the agent in a given environment.

        Parameters
        ----------
        env : Env
            The gym environment where to evaluate the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically, i.e., applying no
            exploration to the policy provided by the MPC. By default, ``True``.
        seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
            Seed for the agent's and env's random number generator. By default ``None``.
        raises : bool, optional
            If ``True``, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            evalution episode (optional, depending on the specific environment).
        terminate_on_infeas : bool, optional
            Whether to terminate the episode when the MPC solution is infeasible. By
            default, ``True``.
        penalty_on_infeas : float, optional
            The penalty to apply to the cost when the MPC solution is infeasible. By
            default, ``0.0``.

        Returns
        -------
        array of doubles
            The cumulative returns (one return per evaluation episode).

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises if the MPC optimization solver fails and ``raises=True``.

        Notes
        -----
        After solving :math:`V_\theta(s)` for the current env's state `s`, the action
        is passed to the environment as the concatenation of the first optimal action
        variables of the MPC (see `csnlp.Mpc.actions`).
        """
        rng = np.random.default_rng(seed)
        self.reset(rng)
        returns = np.zeros(episodes)
        self.on_validation_start(env)

        for episode in range(episodes):
            state, _ = env.reset(seed=mk_seed(rng), options=env_reset_options)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env, episode, state)

            while not (truncated or terminated):
                action, sol = self.state_value(state, deterministic)
                if terminate_on_infeas and sol.infeasible:
                    break  # or terminated = True
                if not sol.success:
                    self.on_mpc_failure(episode, timestep, sol.status, raises)

                state, r, truncated, terminated, _ = env.step(action)
                if sol.infeasible:
                    r += penalty_on_infeas
                self.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        return returns
