import sys
from pathlib import Path
import argparse
from typing import Any

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

lti_example_dir = Path(__file__).parent.parent
sys.path.extend((str(lti_example_dir.parent), str(lti_example_dir)))

from env import ConstrainedLtiEnv as Env
from controllers.mpc import create_mpc


def compute_value_func_on_partition(
    partition: npt.NDArray[np.int_],
    xs: npt.NDArray[np.floating],
    mpc_kwargs: dict[str, Any],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating]]:
    """Computes the value function only on the given partition of the grid.

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
    tuple of array and array
        Returns the partition itself and the value function evaluated at the partition.
    """
    mpc, _ = create_mpc(**mpc_kwargs)
    N = partition.shape[0]
    vs = np.empty(N)
    for k in range(N):
        i, j = partition[k]
        sol = mpc.solve(pars={"x_0": xs[:, i, j], "alpha": 0.99})
        vs[k] = np.inf if sol.infeasible or not sol.success else sol.f
    return partition, vs


def compute_value_func(
    horizon: int,
    dcbf: bool,
    soft: bool,
    grid_side: int,
    n_jobs: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Computes the value function for the constrained LTI environment as the solution
    to the corresponding MPC problem (with a sufficiently long horizon)

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
    two arrays
        Returns a 1d array with the grid points and a 2d array with the value function
        evaluated at the grid points (after creation of a meshgrid).
    """
    # prepare the grid
    grid = np.linspace(-Env.x_soft_bound, Env.x_soft_bound, grid_side)
    X = np.asarray(np.meshgrid(grid, grid))

    # divide the X domain into partitions for each parallel process
    n_jobs = max(n_jobs, 1)
    partitions = np.array_split(list(np.ndindex(X.shape[1:])), n_jobs)

    # compute the value function in parallel
    mpc_kwargs = {
        "horizon": horizon,
        "dcbf": dcbf,
        "soft": soft,
        "bound_initial_state": False,  # doesn't matter when `dcbf=True`
        "dlqr_terminal_cost": False,
        "terminal_cost": set(),
    }
    data = Parallel(n_jobs=n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(compute_value_func_on_partition)(partition, X, mpc_kwargs)
        for partition in partitions
    )

    # gather the results
    V = np.empty((grid_side, grid_side))
    for partition, vs in data:
        for (i, j), v in zip(partition, vs):
            V[i, j] = v
    return grid, V


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Computates value function for the constrained LTI environment.",
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
        default=100,
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
    group.add_argument("--log-scale", action="store_true", help="Plots in log-scale.")
    group = parser.add_argument_group("Computational options")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes (positive)."
    )
    args = parser.parse_args()

    # compute the value function
    grid_points, V = compute_value_func(
        args.horizon, args.dcbf, args.soft, args.grid_side, args.n_jobs
    )

    # save and plot
    if args.save:
        np.savez_compressed(args.save, grid=grid_points, V=V)
    if args.plot or not args.save:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, FuncFormatter, MaxNLocator

        X, Y = np.meshgrid(grid_points, grid_points)
        V_finite = V[np.isfinite(V)]
        kwargs = {"vmin": V_finite.min(), "vmax": V_finite.max(), "cmap": "RdBu_r"}

        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        CS = ax1.contourf(X, Y, V, locator=LogLocator(subs="auto"), **kwargs)
        # ax1.clabel(CS, CS.levels[::10], inline=True, fontsize=10, colors='k')
        if args.log_scale:
            kwargs["vmin"] = np.log10(kwargs["vmin"])
            kwargs["vmax"] = np.log10(kwargs["vmax"])
            V = np.log10(V)
        SF = ax2.plot_surface(X, Y, V, **kwargs)

        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax2.set_xlabel("$x_1$")
        ax2.set_ylabel("$x_2$")
        ax2.set_zlabel("$V(x)$")
        if args.log_scale:
            # thanks to https://stackoverflow.com/a/67774238/19648688
            ax2.zaxis.set_major_formatter(
                FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$")
            )
            ax2.zaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
