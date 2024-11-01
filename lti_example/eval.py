import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from controllers import (
    get_dclf_dcbf_controller,
    get_dlqr_controller,
    get_mpc_controller,
    get_scmpc_controller,
)
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
        func = get_dlqr_controller
    elif controller_name == "dclf-dcbf":
        func = get_dclf_dcbf_controller
    elif controller_name == "mpc":
        func = get_mpc_controller
    elif controller_name == "scmpc":
        func = get_scmpc_controller
    else:
        raise ValueError(f"Unknown controller: {controller_name}")
    return func(*args, **kwargs)


def simulate_controller_once(
    controller_name: Literal["dlqr", "dclf-dcbf", "mpc", "scmpc"],
    controller_kwargs: dict[str, Any],
    timesteps: int,
    reset_kwargs: dict[str, Any],
    seed: int,
) -> tuple[
    float, list[npt.NDArray[np.floating]], list[npt.NDArray[np.floating]], list[float]
]:
    """Simulates one episode of the constrained LTI environment using the given
    controller.

    Parameters
    ----------
    controller_name : {"dlqr", "dclf-dcbf", "mpc", "scmpc"}
        The name of the controller to simulate.
    controller_kwargs : dict of str to any
        The arguments to pass to the controller instantiation.
    timesteps : int
        The number of timesteps to simulate.
    reset_kwargs : dict of str to any
        Optional arguments to pass to the environment's reset method.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    float, tuple of two lists of arrays, and a list of floats
        Returns the total cost of the episode, a tuple of two lists containing
        actions and states arrays, respectively, and a list of solution times.
    """
    env = Env(timesteps)
    controller = get_controller(controller_name, **controller_kwargs)
    x, _ = env.reset(seed=seed, options=reset_kwargs)
    R = 0.0
    U = []
    X = [x]
    sol_times = []
    terminated = truncated = False
    while not (terminated or truncated):
        u, sol_time = controller(x, env)
        x, r, terminated, truncated, _ = env.step(u)
        R += r
        U.append(u)
        X.append(x)
        sol_times.append(sol_time)
    return R, U, X, sol_times


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Evaluation of controllers on the constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Choice of controller")
    group.add_argument(
        "controller",
        choices=["dlqr", "dclf-dcbf", "mpc", "scmpc"],
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
        "--dlqr-terminal-cost",
        action="store_true",
        help="Whether to use the DLQR terminal cost in the MPC controller.",
    )
    group = parser.add_argument_group(
        "Scenario MPC (SCMPC) options (on top of MPC options)"
    )
    group.add_argument(
        "--scenarios",
        type=int,
        default=32,
        help="The number of scenarios to use in the SCMPC controller.",
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument("--n-sim", type=int, default=100, help="Number of simulations.")
    group.add_argument(
        "--timesteps",
        type=int,
        default=30,
        help="Number of timesteps per each simulation.",
    )
    group.add_argument(
        "--ic",
        choices=["contour", "interior", "box"],
        default="contour",
        help="Sets whether the initial conditions (i.e., initial state) of the "
        "environment is drawn from the contour or interior of the max. invariant set, "
        "or its bounding box.",
    )
    group = parser.add_argument_group("Storing and plotting options")
    group.add_argument(
        "--save",
        type=str,
        default="",
        help="Saves with this filename the results under controllers/data. If not set,"
        " no data is saved.",
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

    # prepare arguments to the simulation
    controller = args.controller
    controller_kwargs = {
        "horizon": args.horizon,
        "dcbf": args.dcbf,
        "soft": args.soft,
        "bound_initial_state": args.bound_initial_state,
        "dlqr_terminal_cost": args.dlqr_terminal_cost,
        "scenarios": args.scenarios,
    }
    ts = args.timesteps
    reset_kwargs = {"ic": args.ic}
    seeds = map(int, np.random.SeedSequence(args.seed).generate_state(args.n_sim))

    # run the simulations (possibly in parallel asynchronously)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_controller_once)(
            controller, controller_kwargs, ts, reset_kwargs, seed
        )
        for seed in seeds
    )
    data_dict = {"cost": [], "actions": [], "states": [], "sol_times": []}
    for datum_cost, datum_actions, datum_states, datum_sol_times in data:
        data_dict["cost"].append(datum_cost)
        data_dict["actions"].append(np.asarray(datum_actions))
        data_dict["states"].append(np.asarray(datum_states))
        data_dict["sol_times"].append(np.asarray(datum_sol_times))

    # finally, store and plot the results. If no filepath is passed, always plot
    if args.save:
        from csnlp.util.io import save

        path = Path("data")
        if not path.is_dir():
            path = "lti_example" / path
        path.mkdir(parents=True, exist_ok=True)
        fn = str(path / args.save)
        save(fn, **data_dict, args=args.__dict__, compression="lzma")
    if args.plot or not args.save:
        import matplotlib.pyplot as plt
        from plot import plot_states_and_actions_and_return

        plot_states_and_actions_and_return([data_dict])
        plt.show()
