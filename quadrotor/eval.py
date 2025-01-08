import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

repo_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_dir))

from env import QuadrotorEnv as Env


def get_controller(
    controller_name: Literal["mpc", "scmpc"],
    *args: Any,
    **kwargs: Any,
) -> Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]]:
    """Returns the controller function given its name.

    Parameters
    ----------
    controller_name : {"mpc", "scmpc"}
        The name of the controller to return.
    args, kwargs
        The arguments to pass to the controller function.

    Returns
    -------
    callable from (array-like, ConstrainedLtiEnv) to (array-like, float)
        A controller that maps the current state to the desired action, and returns also
        the time it took to compute the action.

    Raises
    ------
    ValueError
        Raises an error if the controller name is not recognized.
    """
    if controller_name == "mpc":
        from controllers.mpc import get_mpc_controller as func
    else:
        raise ValueError(f"Unknown controller: {controller_name}")
    return func(*args, **kwargs)


def simulate_controller_once(
    controller_name: Literal["mpc", "scmpc"],
    controller_kwargs: dict[str, Any],
    n_eval: int,
    timesteps: int,
    seed: int,
) -> tuple[npt.NDArray[np.floating], ...]:
    """Simulates one episode of the quadrotor environment using the given controller.

    Parameters
    ----------
    controller_name : {"mpc", "scmpc"}
        The name of the controller to simulate.
    controller_kwargs : dict of str to any
        The arguments to pass to the controller instantiation.
    n_eval : int
        The number of evaluations to perform for this controller.
    timesteps : int
        The number of timesteps to run each evaluation for.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    5 float arrays
        Returns the total cost for each episode, two arrays containing the  actions and
        states trajectories respectively, an array of solution computation times, and
        an array containing the obstacles positions and direction for each evaluation.
    """
    # create env and controller only once
    env = Env(timesteps)
    controller = get_controller(controller_name, **controller_kwargs, seed=seed)

    # simulate the controller on the environment for n_eval evaluations
    R = np.zeros(n_eval)
    U = np.empty((n_eval, timesteps, env.na))
    X = np.empty((n_eval, timesteps + 1, env.ns))
    sol_times = np.empty((n_eval, timesteps))
    obstacles = np.empty((n_eval, 2, *env.safety_constraints.size_in(1)))
    for e, s in enumerate(np.random.SeedSequence(seed).generate_state(n_eval)):
        controller.reset()
        x, _ = env.reset(seed=int(s))
        obstacles[e] = env.pos_obs, env.dir_obs
        X[e, 0] = x
        for t in range(timesteps):
            u, sol_time = controller(x, env)
            x, r, _, _, _ = env.step(u)
            R[e] += r
            U[e, t] = u
            X[e, t + 1] = x
            sol_times[e, t] = sol_time
    return R, U, X, sol_times, obstacles


if __name__ == "__main__":
    # parse script arguments
    parser = ArgumentParser(
        description="Evaluation of controllers on the quadrotor environment.",
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
        "--dcbf",
        action="store_true",
        help="Whether to use discrete-time CBF constraints in the MPC controller.",
    )
    group.add_argument(
        "--soft",
        action="store_true",
        help="Whether to use soft constraints in the MPC controller.",
    )
    group.add_argument(
        "--bound-initial-state",
        action="store_true",
        help="Whether to bound the initial state in the MPC controller.",
    )
    group.add_argument(
        "--terminal-cost",
        choices=("dlqr", "pwqnn", "psdnn"),
        nargs="*",
        default=set(),
        help="Which type of terminal cost to use in the MPC controller.",
    )
    group = parser.add_argument_group(
        "Scenario MPC (SCMPC) options (used only when `controller=scmpc`)"
    )
    group.add_argument(
        "--scenarios",
        type=int,
        default=32,
        help="The number of scenarios to use in the SCMPC controller.",
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument(
        "--n-ctrl", type=int, default=10, help="Number of controllers to simulate."
    )
    group.add_argument(
        "--n-eval", type=int, default=10, help="Number of evaluations per controller."
    )
    group.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Number of timesteps per each simulation.",
    )
    group = parser.add_argument_group("Storing and plotting options")
    group.add_argument(
        "--save",
        type=str,
        default="",
        help="Saves results with this filename. If not set, no data is saved.",
    )
    group.add_argument(
        "--plot",
        action="store_true",
        help="Shows results in a plot at the end of the simulation.",
    )
    group = parser.add_argument_group("Computational options")
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    args = parser.parse_args()
    args.terminal_cost = set(args.terminal_cost)

    # prepare arguments to the simulation
    controller = args.controller
    controller_kwargs = {
        "horizon": args.horizon,
        "dcbf": args.dcbf,
        "soft": args.soft,
        "bound_initial_state": args.bound_initial_state,
        "terminal_cost": args.terminal_cost,
        "scenarios": args.scenarios,
    }
    n_eval = args.n_eval
    ts = args.timesteps
    seeds = map(int, np.random.SeedSequence(args.seed).generate_state(args.n_ctrl))

    # run the simulations (possibly in parallel asynchronously)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_controller_once)(
            controller, controller_kwargs, n_eval, ts, seed
        )
        for seed in seeds
    )
    keys = ("cost", "actions", "states", "sol_times", "obstacles")
    data_dict = dict(zip(keys, map(np.asarray, zip(*data))))

    # finally, store and plot the results. If no filepath is passed, always plot
    if args.save:
        from csnlp.util.io import save

        save(args.save, **data_dict, args=args.__dict__, compression="lzma")
    if args.plot or not args.save:
        import matplotlib.pyplot as plt
        from plot import plot_returns, plot_solver_times, plot_states_and_actions

        data = [data_dict]
        plot_states_and_actions(data)
        plot_returns(data)
        plot_solver_times(data)
        plt.show()
