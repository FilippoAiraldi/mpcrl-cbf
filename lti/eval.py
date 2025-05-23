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

from env import ConstrainedLtiEnv as Env


def get_controller(
    controller_name: Literal["dlqr", "dclf-dcbf", "mpc", "scmpc"],
    *args: Any,
    **kwargs: Any,
) -> Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]]:
    """Returns the controller function given its name.

    Parameters
    ----------
    controller_name : {"dlqr", "dclf-dcbf", "mpc", "scmpc"}
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
    if controller_name == "dlqr":
        from controllers.dlqr import get_dlqr_controller as func
    elif controller_name == "dclf-dcbf":
        from controllers.dclf_dcbf import get_dclf_dcbf_controller as func
    elif controller_name == "mpc":
        from controllers.mpc import get_mpc_controller as func
    elif controller_name == "scmpc":
        from controllers.scmpc import get_scmpc_controller as func
    else:
        raise ValueError(f"Unknown controller: {controller_name}")
    return func(*args, **kwargs)


def simulate_controller_once(
    controller_name: Literal["dlqr", "dclf-dcbf", "mpc", "scmpc"],
    controller_kwargs: dict[str, Any],
    n_eval: int,
    timesteps: int,
    reset_kwargs: dict[str, Any],
    weights: dict[str, npt.NDArray[np.floating]] | None,
    seed: int,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    dict[str, npt.NDArray[np.floating]],
]:
    """Simulates one episode of the constrained LTI environment using the given
    controller.

    Parameters
    ----------
    controller_name : {"dlqr", "dclf-dcbf", "mpc", "scmpc"}
        The name of the controller to simulate.
    controller_kwargs : dict of str to any
        The arguments to pass to the controller instantiation.
    n_eval : int
        The number of evaluations to perform for this controller.
    timesteps : int
        The number of timesteps to run each evaluation for.
    reset_kwargs : dict of str to any
        Optional arguments to pass to the environment's reset method.
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
        x, _ = env.reset(seed=int(s), options=reset_kwargs)
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
        description="Evaluation of controllers on the constrained LTI environment.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Choice of controller")
    group.add_argument(
        "controller",
        choices=("dlqr", "dclf-dcbf", "mpc", "scmpc"),
        help="The controller to use for the simulation.",
    )
    group = parser.add_argument_group(
        "MPC options (used only when `controller=mpc` or `controller=scmpc`)"
    )
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
        choices=("dlqr", "pwqnn"),
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
        default=30,
        help="Number of timesteps per each simulation.",
    )
    group.add_argument(
        "--ic",
        choices=("contour", "interior", "box"),
        default="contour",
        help="Sets whether the initial conditions (i.e., initial state) of the "
        "environment is drawn from the contour or interior of the max. invariant set, "
        "or its bounding box.",
    )
    group.add_argument(
        "--from-train",
        type=str,
        default="",
        help="Loads a trained learning-based controller from training results' file, "
        "instead of randomly initializing it. If set, `--n-ctrl` is overwritten to the "
        "number of controllers in the file. Only supported for `controller=scmpc`.",
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

    # if a training file is specified, load the last learnable weights from it and
    # overwrite the number of controllers
    if args.from_train:
        if args.controller != "scmpc":
            raise RuntimeError("Only SCMPC controllers can be loaded from file.")

        from csnlp.util.io import load

        data = load(args.from_train)
        if "updates_history" not in data:
            raise RuntimeError("No learning history found in the file.")
        params = data["updates_history"]
        args.n_ctrl = next(iter(params.values())).shape[0]  # n_eval is left untouched
        data_args = data.pop("args")
        for attr in (
            "horizon",
            "dcbf",
            "soft",
            "bound_initial_state",
            "terminal_cost",
            "scenarios",
        ):
            setattr(args, attr, data_args[attr])
        weights = [{n: w[i, -1] for n, w in params.items()} for i in range(args.n_ctrl)]
        print(f"Loaded {args.n_ctrl} controllers from {args.from_train}.")
    else:
        weights = [None] * args.n_ctrl
    print(f"Args: {args}\n")

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
    reset_kwargs = {"ic": args.ic}
    seeds = map(int, np.random.SeedSequence(args.seed).generate_state(args.n_ctrl))

    # run the simulations (possibly in parallel asynchronously)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_controller_once)(
            controller, controller_kwargs, n_eval, ts, reset_kwargs, weights_, seed
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
        from plot import plot_returns, plot_solver_times, plot_states_and_actions

        data = [data_dict]
        plot_states_and_actions(data)
        plot_returns(data)
        plot_solver_times(data)
        plt.show()
