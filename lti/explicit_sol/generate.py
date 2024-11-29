"""Computes and plot and/or save to disk the optimal infinite-horizon value function and
policy for the constrained LTI environment."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

lti_dir = Path(__file__).parent.parent
sys.path.extend((str(lti_dir.parent), str(lti_dir)))

from controllers.mpc import create_mpc
from env import ConstrainedLtiEnv as Env


def compute_value_func_and_policy_on_partition(
    partition: npt.NDArray[np.int_],
    xs: npt.NDArray[np.floating],
    mpc_kwargs: dict[str, Any],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Computes the value function and optimal policy only on the given partition of the
    grid.

    Parameters
    ----------
    partition : array of 2d indices
        The indices of the grid points to compute the value function for.
    xs : array
        The whole grid of states.
    mpc_kwargs : dict
        The arguments to pass to `create_mpc`.

    Returns
    -------
    tuple of arrays
        Returns the partition itself and the value function and policy evaluated at the
        partition.
    """
    mpc, _ = create_mpc(**mpc_kwargs)
    N = partition.shape[0]
    vs = np.empty(N)
    us = np.empty((N, mpc.na))
    for k in range(N):
        i, j = partition[k]
        sol = mpc.solve(pars={"x_0": [xs[j], xs[i]]})
        if sol.infeasible or not sol.success:
            vs[k] = us[k] = np.nan
        else:
            vs[k] = sol.f
            us[k] = sol.vals["u"][:, 0].toarray().flatten()
    return partition, vs, us


def compute_value_func_and_policy(
    horizon: int,
    dcbf: bool,
    soft: bool,
    grid_side: int,
    n_jobs: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Computes the value function and optimal policy for the constrained LTI
    environment as the solution to the corresponding MPC problem (with a sufficiently
    long horizon)

    Parameters
    ----------
    horizon : int
        The horizon of the MPC controller. For this problem, 10 should be long enough.
    dcbf : bool
        Whether to use discrete-time CBF constraints in the MPC controller.
    soft : bool
        Whether to use soft constraints in the MPC controller.
    grid_side : int
        Size of the side of the grid for which to compute the value function.
    n_jobs : int
        Number of parallel processes to use.

    Returns
    -------
    tuple of arrays
        Returns a 1d array with the grid points, and two 2d arrays with the value
        function and policy evaluated at the grid points (after creation of a meshgrid).
    """
    # prepare the grid
    grid = np.linspace(-Env.x_soft_bound, Env.x_soft_bound, grid_side)

    # divide the X domain into partitions for each parallel process
    n_jobs = max(n_jobs, 1)
    partitions = np.array_split(list(np.ndindex((grid_side, grid_side))), n_jobs)

    # compute the value function and policy in parallel
    mpc_kwargs = {
        "horizon": horizon,
        "dcbf": dcbf,
        "soft": soft,
        "bound_initial_state": False,  # doesn't matter when `dcbf=True`
        "dlqr_terminal_cost": False,
        "terminal_cost": set(),
    }
    data = Parallel(n_jobs=n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(compute_value_func_and_policy_on_partition)(partition, grid, mpc_kwargs)
        for partition in partitions
    )

    # gather the results
    V = np.empty((grid_side, grid_side))
    U = np.empty((grid_side, grid_side, Env.na))
    for partition, vs, us in data:
        V[partition[:, 0], partition[:, 1]] = vs
        U[partition[:, 0], partition[:, 1]] = us
    return grid, V, U


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Computates optimal value function and policy for the constrained "
        "LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("MPC options")
    group.add_argument(
        "--horizon",
        type=int,
        default=20,
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
    group = parser.add_argument_group("Simulation options")
    group.add_argument(
        "--grid-side",
        type=int,
        default=300,
        help="Size of the side of the grid for which to compute the value function.",
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
    group.add_argument("--log", action="store_true", help="Plots in log scale.")
    group = parser.add_argument_group("Computational options")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes (positive)."
    )
    args = parser.parse_args()

    # compute the value function
    grid_points, V, U = compute_value_func_and_policy(
        args.horizon, args.dcbf, args.soft, args.grid_side, args.n_jobs
    )

    # save and plot
    if args.save:
        np.savez_compressed(args.save, grid=grid_points, V=V, U=U)
    if args.plot or not args.save:
        import matplotlib.pyplot as plt

        from lti.explicit_sol.plot import do_plot

        do_plot(({"grid": grid_points, "V": V, "U": U},), args.log)
        plt.show()
