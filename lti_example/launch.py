import argparse
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import save
from env import ConstrainedLtiEnv as Env
from joblib import Parallel, delayed
from plot import plot_states_and_actions_and_return


def simulate_once(
    controller: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    timesteps: int,
    seed: int,
    **reset_kwargs: Any,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    """Simulates one episode of the constrained LTI environment using the given
    controller.

    Parameters
    ----------
    controller : callable from array-like to array-like
        A controller that maps the current state to the desired action.
    timesteps : int
        The number of timesteps to simulate.
    seed : int
        The seed for the random number generator.
    reset_kwargs : any
        Optional arguments to pass to the environment's reset method.

    Returns
    -------
    tuple of two arrays and float
        The first array contains the state trajectory, and the second array contains
        the action trajectory. The float is the total cost of the episode.
    """
    if reset_kwargs is None:
        reset_kwargs = {}
    env = Env()
    x, _ = env.reset(seed=seed, options=reset_kwargs)
    X = np.empty((timesteps + 1, env.ns))
    X[0] = x
    U = np.empty((timesteps, env.na))
    cost = 0.0
    for i in range(timesteps):
        u = controller(x)
        x, c, _, _, _ = env.step(u)
        X[i + 1] = x
        U[i] = u
        cost += c
    return X, U, cost


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Simulation of the constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Controller options")
    group.add_argument(
        "controller",
        choices=["dlqr", "dclf-dcbf"],
        help="The controller to use for the simulation.",
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
        "--init-conditions",
        choices=["contour", "interior"],
        default="contour",
        help="Sets whether the initial state in the environment is on the contour or in"
        " the interior of the safe set.",
    )
    group = parser.add_argument_group("Storing and plotting options")
    group.add_argument(
        "--save",
        type=str,
        default="",
        help="Save results under controllers/data. If not set, no data is saved.",
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

    # get the controller
    controller_module_name = "controllers." + args.controller.replace("-", "_")
    controller_module = import_module(controller_module_name)
    controller = getattr(controller_module, "get_controller")()

    # run the simulations (possibly in parallel asynchronously)
    ic_on_contour = args.init_conditions == "contour"
    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_sim)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_once)(
            controller, args.timesteps, int(seed), contour=ic_on_contour
        )
        for seed in seeds
    )
    keys = ("states", "actions", "cost")
    data_dict = dict(zip(keys, map(np.asarray, zip(*data))))

    # finally, store and plot the results. If no filepath is passed, always plot
    if args.save:
        path = Path("data")
        if not path.is_dir():
            path = "lti_example" / path
        path.mkdir(parents=True, exist_ok=True)
        save(str(path / args.save), **data_dict, args=args, compression="matlab")
    if args.plot or not args.save:
        plot_states_and_actions_and_return([data_dict])
        plt.show()
