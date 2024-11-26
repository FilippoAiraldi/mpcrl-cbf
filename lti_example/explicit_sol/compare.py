"""Compares the learned control policy (intended as the result of a training process)
with the optimal control policy for the explicit constrained LTI system."""

import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentError, ArgumentParser
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import load
from joblib import Parallel, delayed

lti_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.extend((lti_dir, os.path.dirname(lti_dir)))

from controllers.mpc import create_mpc
from env import ConstrainedLtiEnv as Env


def load_precomputed_policy(
    filename: str,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Loads the pre-comptued policy from a NumPy file.

    Parameters
    ----------
    filename : str
        The filename of the policy to load.

    Returns
    -------
    tuple of 2 arrays
        The grid and the control policy.
    """
    data = np.load(filename)
    return data["grid"], data["U"]


def compute_policy_on_partition(
    partition: npt.NDArray[np.int_],
    xs: npt.NDArray[np.floating],
    mpc_kwargs: dict[str, Any],
    mpc_pars: dict[str, npt.NDArray[np.floating]],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating]]:
    mpc, _ = create_mpc(**mpc_kwargs)
    # TODO: set the parameters here
    N = partition.shape[0]
    us = np.empty((N, Env.na))
    for k in range(N):
        i, j = partition[k]
        mpc_pars["x_0"] = [xs[j], xs[i]]
        sol = mpc.solve(pars=mpc_pars)
        us[k] = (
            np.nan
            if sol.infeasible or not sol.success
            else sol.vals["u"][:, 0].toarray().flatten()
        )
    return partition, us


def compute_policy(
    filename: str, grid: npt.NDArray[np.floating], n_jobs: int
) -> npt.NDArray[np.floating]:
    # load the data from the file - we need the final parameters and the training args
    data = load(filename)
    args = data.pop("args")
    mpc_params = {n: par[:, -1].mean(0) for n, par in data["updates_history"].items()}

    # prepare the grid and divide it into partitions
    grid_side = grid.size
    partitions = np.array_split(list(np.ndindex((grid_side, grid_side))), n_jobs)

    # compute the value function in parallel
    mpc_kwarg_names = [
        "horizon",
        "dcbf",
        "soft",
        "bound_initial_state",
        "terminal_cost",
        "scenarios",
    ]
    mpc_kwargs = {n: args[n] for n in mpc_kwarg_names}
    data = Parallel(n_jobs=n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(compute_policy_on_partition)(partition, grid, mpc_kwargs, mpc_params)
        for partition in partitions
    )

    # gather the results
    U = np.empty((grid_side, grid_side, Env.na))
    for partition, us in data:
        U[partition[:, 0], partition[:, 1]] = us
    return U


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Comparison of learned policy w.r.t. optimal one.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Policies to compare")
    group.add_argument(
        "filename1",
        type=str,
        help="First filename of the policies to compare (either the results from "
        "training or a pre-computed explicit solution).",
    )
    group.add_argument(
        "filename2",
        type=str,
        help="Second filename of the policies to compare (see previous).",
    )
    group = parser.add_argument_group("Other options")
    group.add_argument("--log-scale", action="store_true", help="Plots in log-scale.")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes (positive)."
    )
    args = parser.parse_args()

    fn1: str = args.filename1
    fn2: str = args.filename2
    n_jobs = max(args.n_jobs, 1)

    # load or compute the two policies
    is_fn1_npz = fn1.endswith(".npz")
    is_fn2_npz = fn2.endswith(".npz")
    if is_fn1_npz and is_fn2_npz:
        grid, policy1 = load_precomputed_policy(fn1)
        grid_, policy2 = load_precomputed_policy(fn2)
        assert np.array_equal(grid, grid_), "Grids must be the same for comparison."
    elif not (is_fn1_npz or is_fn2_npz):
        raise ArgumentError(
            "At least one of the two policies must be pre-computed (a .npz file)."
        )
    elif is_fn1_npz:
        grid, policy1 = load_precomputed_policy(fn1)
        policy2 = compute_policy(fn2, grid, n_jobs)
    else:
        grid, policy2 = load_precomputed_policy(fn2)
        policy2 = compute_policy(fn1, grid, n_jobs)

    # plot differences between the two policies
    X1, X2 = np.meshgrid(grid, grid)
    delta = np.abs(policy1 - policy2)
    vmin = np.nanmin(delta)
    vmax = np.nanmax(delta)
    if args.log_scale:
        delta = np.log10(delta)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    kwargs = {"vmin": vmin, "vmax": vmax, "cmap": "RdBu_r"}

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    ax1.contourf(X1, X2, delta[..., 0], **kwargs)
    CS = ax2.contourf(X1, X2, delta[..., 1], **kwargs)
    ax1.set_title("Abs. error in $u_1$")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")
    ax2.set_title("Abs. error in $u_2$")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_aspect("equal")
    fig.colorbar(CS, ax=(ax1, ax2), orientation="horizontal")
    plt.show()
