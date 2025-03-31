import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from itertools import repeat
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

repo_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_dir))

from env import QuadrotorEnv as Env

from util.defaults import QUADROTOR_NN_HIDDEN


def get_controller(
    controller_name: Literal["mpc", "scmpc"],
    *args: Any,
    **kwargs: Any,
) -> tuple[
    Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]],
    dict[str, npt.NDArray[np.floating]] | None,
]:
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
    dict of str to arrays, optional
        The numerical weights of the neural network used to learn the DCBF Kappa
        function (used only for saving to disk for plotting).

    Raises
    ------
    ValueError
        Raises an error if the controller name is not recognized.
    """
    if controller_name == "mpc":
        from controllers.mpc import get_mpc_controller as func
    elif controller_name == "scmpc":
        from controllers.scmpc import get_scmpc_controller as func
    else:
        raise ValueError(f"Unknown controller: {controller_name}")
    return func(*args, **kwargs)


def simulate_controller_once(
    controller_name: Literal["mpc", "scmpc"],
    controller_kwargs: dict[str, Any],
    n_eval: int,
    timesteps: int,
    weights: dict[str, npt.NDArray[np.floating]] | None,
    seed: int,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    dict[str, npt.NDArray[np.floating]],
]:
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
    weights : dict of str to array-like, optional
        The weights (e.g., for neural networks) to use in the controller. If `None`,
        these are initialized randomly or to default values.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    4 float arrays and an dict
        Returns the following objects:
         - an array of the total cost for each episode
         - two arrays containing the actions and states trajectories respectively
         - an array of solution computation times
         - a dict of the MPC numerical weights. If some `weights` were passed in, this
           contains the same values. If not passed, this contains the randomly
           initialized weights. Can be empty if MPC is not parametric.
    """
    # create env and controller only once
    env = Env(timesteps)
    controller, weights_ = get_controller(
        controller_name, **controller_kwargs, weights=weights, seed=seed
    )

    # simulate the controller on the environment for n_eval evaluations
    R = np.zeros(n_eval)
    U = np.empty((n_eval, timesteps, env.na))
    X = np.empty((n_eval, timesteps + 1, env.ns))
    sol_times = np.empty((n_eval, timesteps))
    for e, s in enumerate(np.random.SeedSequence(seed).generate_state(n_eval)):
        controller.reset()
        x, _ = env.reset(seed=int(s))
        X[e, 0] = x
        for t in range(timesteps):
            u, sol_time = controller(x, env)
            x, r, _, _, _ = env.step(u)
            R[e] += r
            U[e, t] = u
            X[e, t + 1] = x
            sol_times[e, t] = sol_time
    return R, U, X, sol_times, weights_


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
        "--use-kappann",
        action="store_true",
        help="Whether to use the NN to also provide CBF class Kappa function.",
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
        choices=("dlqr", "psdnn"),
        nargs="*",
        default=set(),
        help="Which type of terminal cost to use in the MPC controller.",
    )
    group.add_argument(
        "--nn-hidden",
        type=int,
        default=QUADROTOR_NN_HIDDEN,
        nargs=2,
        help="The number of hidden units in the NN for terminal cost and class Kappa "
        "function, if used.",
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
        default=125,
        help="Number of timesteps per each simulation.",
    )
    group.add_argument(
        "--from-pre-train",
        type=str,
        default="",
        help="Loads pre-trained quadrotor NN weights from the specified file.",
    )
    group.add_argument(
        "--from-train",
        type=str,
        default="",
        help="Loads a trained learning-based controller, alongside other options, from "
        "the specified training results' file.",
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

    # load weights from pre-training or training file
    if args.from_pre_train and args.from_train:
        raise ValueError(
            "Cannot specify both `from_pre_train` and `from_train` arguments."
        )
    elif args.from_pre_train:
        import torch

        data = torch.load(args.from_pre_train, weights_only=True)
        expected_shape = data["args"]["nn_hidden"]
        if not np.array_equal(args.nn_hidden, expected_shape):
            raise ValueError(
                f"Hidden sizes mismatch: {args.nn_hidden} != {expected_shape}"
            )
        weights_ = {
            "nn." + n: np.atleast_2d(w.numpy(force=True)).astype(np.float64)
            for n, w in data["model_state_dict"].items()
        }
        weights = repeat(weights_)
    elif args.from_train:
        from csnlp.util.io import load

        data = load(args.from_train)
        data_args = data.pop("args")
        args.n_ctrl = data_args["n_agents"]  # n_eval is left untouched
        for attr in (
            "horizon",
            "dcbf",
            "use_kappann",
            "soft",
            "bound_initial_state",
            "terminal_cost",
            "nn_hidden",
            "scenarios",
        ):
            setattr(args, attr, data_args[attr])
        weights = data["weights"]
        weights = [{n: w[a] for n, w in weights.items()} for a in range(args.n_ctrl)]
    else:
        weights = repeat(None)

    tcost = set(args.terminal_cost)
    print(f"Args: {args}\n")

    # prepare arguments to the simulation
    controller = args.controller
    controller_kwargs = {
        "horizon": args.horizon,
        "dcbf": args.dcbf,
        "use_kappann": args.use_kappann,
        "soft": args.soft,
        "bound_initial_state": args.bound_initial_state,
        "terminal_cost": tcost,
        "scenarios": args.scenarios,
        "nn_hidden_sizes": args.nn_hidden,
    }
    n_eval = args.n_eval
    ts = args.timesteps
    seeds = map(int, np.random.SeedSequence(args.seed).generate_state(args.n_ctrl))

    # run the simulations (possibly in parallel asynchronously)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_controller_once)(
            controller, controller_kwargs, n_eval, ts, weights_, seed
        )
        for weights_, seed in zip(weights, seeds)
    )

    # congregate data all together - weights is a dictionary, so requires further
    # attention
    keys = ("cost", "actions", "states", "sol_times", "weights")
    data_dict = dict(zip(keys, map(np.asarray, zip(*data))))
    weights = data_dict.pop("weights")
    wnames = weights[0].keys()
    data_dict["weights"] = {n: np.asarray([d[n] for d in weights]) for n in wnames}

    # finally, store and plot the results. If no filepath is passed, always plot
    if args.save:
        from csnlp.util.io import save

        save(args.save, **data_dict, args=args.__dict__, compression="lzma")
    if args.plot or not args.save:
        import matplotlib.pyplot as plt
        from plot import (
            plot_action_bounds,
            plot_returns,
            plot_safety,
            plot_solver_times,
            plot_states_and_actions,
        )

        data, names = [data_dict], ["eval"]
        plot_states_and_actions(data)
        plot_action_bounds(data)
        plot_safety(data, names)
        plot_returns(data, names)
        plot_solver_times(data, names)
        plt.show()
