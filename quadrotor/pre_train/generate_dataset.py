import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from csnlp.util.io import save
from joblib import Parallel, delayed

quadrotor_dir, repo_dir = Path(__file__).resolve().parents[1:3]
sys.path.append(str(repo_dir))
sys.path.append(str(quadrotor_dir))

from quadrotor.env import QuadrotorEnv as Env
from quadrotor.eval import get_controller


def simulate_controller_once(
    controller_name: Literal["mpc", "scmpc"],
    controller_kwargs: dict[str, Any],
    n_ep: int,
    timesteps: int,
    initial_conditions: Sequence[npt.NDArray[np.float64]],
    seeds: Sequence[int],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Simulates one episode of the quadrotor environment using the given controller.

    Parameters
    ----------
    controller : callable
        A controller that takes the current state and the environment as input and
        returns the control input for the next timestep. Also has a `reset` method.
    n_ep : int
        The number of episodes to run for this controller.
    timesteps : int
        The number of timesteps to run each episode for.
    initial_conditions : sequence of arrays of shape `(ns,)`
        The initial conditions for each episode.
    seeds : int
        The seeds for the random number generator for every episode.

    Returns
    -------
    4 float arrays
        Returns the following objects:
         - an array containing the rewards for each episode's timestep
         - an array containing the state trajectories
         - an array containing the action trajectories (shifted one timestep backwards)
         - an array containing the obstacles positions and direction for each episode.
    """
    env = Env(timesteps)
    controller, _ = get_controller(controller_name, **controller_kwargs)
    R = np.empty((n_ep, timesteps))
    U_prev = np.empty((n_ep, timesteps, env.na))
    X = np.empty((n_ep, timesteps, env.ns))
    obstacles = np.empty((n_ep, 2, *env.safety_constraints.size_in(1)))

    for e, (seed, ic) in enumerate(zip(seeds, initial_conditions)):
        controller.reset()
        x, _ = env.reset(seed=int(seed), options={"ic": ic})
        obstacles[e] = env.pos_obs, env.dir_obs
        for t in range(timesteps):
            X[e, t] = x
            U_prev[e, t] = env.previous_action
            u, _ = controller(x, env)
            x, r, _, _, _ = env.step(u)
            R[e, t] = r
    return R, X, U_prev, obstacles


def generate_dataset_chunk(
    controller_name: Literal["mpc", "scmpc"],
    controller_kwargs: dict[str, Any],
    n_ep: int,
    timesteps: int,
    initial_conditions: Sequence[npt.NDArray[np.float64]],
    seeds: Sequence[int],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Generates a chunk of dataset corresponding to the specified amount of episodes.

    Parameters
    ----------
    controller : callable
        A controller that takes the current state and the environment as input and
        returns the control input for the next timestep. Also has a `reset` method.
    n_ep : int
        The number of episodes to run for this controller.
    timesteps : int
        The number of timesteps to run each episode for.
    initial_conditions : sequence of arrays of shape `(ns,)`
        The initial conditions for each episode.
    seeds : int
        The seeds for the random number generator for every episode.

    Returns
    -------
    4 float arrays
        Returns the following objects:
         - an array containing the cost-to-go target for the pre-training.
         - an array containing the state input for the pre-training
         - an array containing the previous action input for the pre-training
         - an array containing the obstacle data input for the pre-training
    """
    R, X, U_prev, obstacles = simulate_controller_once(
        controller_name, controller_kwargs, n_ep, timesteps, initial_conditions, seeds
    )
    obstacles_ = obstacles.reshape(n_ep, 2, -1, order="F")  # same orientation as casadi
    G = R[:, ::-1].cumsum(1)[:, ::-1]  # compute cost-to-go
    return G, X, U_prev, obstacles_


if __name__ == "__main__":
    default_save = f"pre_train_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    parser = ArgumentParser(
        description="Generation of terminal cost pre-training datasets.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Choice of controller")
    group.add_argument(
        "controller",
        choices=("mpc", "scmpc"),
        help="The controller to use for the simulation.",
    )
    group = parser.add_argument_group("MPC options")
    group.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="The horizon of the MPC controller.",
    )
    group.add_argument(
        "--soft",
        action="store_true",
        help="Whether to use soft constraints in the MPC controller.",
    )
    group.add_argument(
        "--scenarios",
        type=int,
        default=32,
        help="The number of scenarios to use in the SCMPC controller (used only when "
        " `controller=scmpc`).",
    )
    group = parser.add_argument_group("Dataset generation options")
    group.add_argument(
        "--timesteps",
        type=int,
        default=125,
        help="Number of timesteps per each simulation.",
    )
    group.add_argument(
        "--n-episodes",
        type=int,
        default=500,
        help="Number of episodes to include in the dataset.",
    )
    group = parser.add_argument_group("Storing options")
    group.add_argument(
        "--save",
        type=str,
        default=default_save,
        help="Saves results with this filename. If not set, a default name is given.",
    )
    group = parser.add_argument_group("Computational options")
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    args = parser.parse_args()
    assert args.save, "Please, provide a filename to save the dataset to."
    print(f"Args: {args}\n")

    # prepare arguments
    n_jobs = args.n_jobs
    controller = args.controller
    controller_kwargs = {
        "horizon": args.horizon,
        "soft": args.soft,
        "scenarios": args.scenarios,
        # kwargs below are default
        "dcbf": False,
        "use_kappann": False,
        "bound_initial_state": False,
        "terminal_cost": set(),
        "kappann_hidden_size": [],
        "pwqnn_hidden_size": 0,
        "psdnn_hidden_sizes": [],
    }
    timesteps = args.timesteps
    n_ep = args.n_episodes
    n_ep_per_job = ceil(n_ep / n_jobs)  # round up

    # generate all seeds at once to ensure reproducibility, and generate random initial
    # conditions for each episode
    seed = args.seed
    seeds = (
        np.random.SeedSequence(seed)
        .generate_state(n_ep_per_job * n_jobs)
        .reshape(n_jobs, n_ep_per_job)
    )
    initial_conditions = np.random.default_rng(seed).normal(
        Env.x0, [1.0, 1.0, 3.0, 1.0, 1.0, 1.0], (n_jobs, n_ep_per_job, Env.ns)
    )

    # simulate the controllers in parallel, and save dataset to disk
    data = Parallel(n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(generate_dataset_chunk)(
            controller, controller_kwargs, n_ep_per_job, timesteps, ics_, seeds_
        )
        for ics_, seeds_ in zip(initial_conditions, seeds)
    )
    cost_to_go, states, prev_actions, obstacles = map(np.concatenate, zip(*data))
    save(
        args.save,
        cost_to_go=cost_to_go,
        states=states,
        previous_actions=prev_actions,
        obstacles=obstacles,
        args=args.__dict__,
        compression="lzma",
    )
