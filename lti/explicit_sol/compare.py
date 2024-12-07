"""Compares the learned value function and control policy (intended as the result of a
training process) with the optimal explicit value function and control policy for the
constrained LTI system."""

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import load
from csnn.convex import PwqNN
from joblib import Parallel, delayed
from matplotlib import cm, colors
from mpcrl.util.control import dlqr

expl_sol_dir, lti_dir, repo_dir = Path(__file__).resolve().parents[:3]
sys.path.extend((str(repo_dir), str(lti_dir)))

from controllers.mpc import create_mpc
from env import ConstrainedLtiEnv as Env

from util.nn import nn2function


def load_explicit(dcbf: bool, soft: bool) -> tuple[npt.NDArray[np.floating], ...]:
    """Loads the pre-computed explicit solution from a NumPy file.

    Parameters
    ----------
    dcbf : bool
        Whether the policy was computed with DCBF constraints.
    soft : bool
        Whether the policy was computed with soft constraints.

    Returns
    -------
    tuple of 3 arrays
        The grid, optimal value function and optimal control policy over the grid.
    """
    filename = str(expl_sol_dir / "data")
    if dcbf:
        filename += "_dcbf"
    if soft:
        filename += "_soft"
    filename += ".npz"
    data = np.load(filename)
    return data["grid"], data["V"], data["U"]


def compute_learned_value_function(
    grid: npt.NDArray[np.floating],
    args: dict[str, Any],
    mpc_pars: dict[str, npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    """Computes the learned value function over the given grid.

    Parameters
    ----------
    grid : array
        1D array of grid points.
    args : dict of any
        The arguments used for the simulation.
    mpc_pars : dict of arrays
        The NN weights for the MPC controller.

    Returns
    -------
    array
        The learned value function for each point of the 2D grid.
    """
    N = grid.size
    X = np.asarray(np.meshgrid(grid, grid)).reshape(Env.ns, -1)
    V = np.zeros((N, N))
    tc_components = args.get("terminal_cost", set())

    if "dlqr" in tc_components:
        _, P = dlqr(Env.A, Env.B, Env.Q, Env.R)
        V += (X.transpose(1, 2, 0).dot(P) * X.transpose(1, 2, 0)).sum(-1)
    if "pwqnn" in tc_components:
        hidden_features = mpc_pars["pwqnn.input_layer.weight"].shape[0]
        pwqnn = nn2function(PwqNN(Env.ns, hidden_features), prefix="pwqnn")
        V += pwqnn(x=X, **mpc_pars)["y"].toarray().reshape(N, N)
    return V


def compute_policy_on_partition(
    partition: npt.NDArray[np.int_],
    xs: npt.NDArray[np.floating],
    mpc_kwargs: dict[str, Any],
    mpc_pars: dict[str, npt.NDArray[np.floating]],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating]]:
    """_summary_

    Parameters
    ----------
    partition : array of ints
        The partition of the gridded state space to compute the policy on.
    xs : array
        The grid points of the state space.
    mpc_kwargs : dict of any
        Keywords arguments for the instantiation of the MPC controller.
    mpc_pars : dict of arrays
        The NN weights for the MPC controller.

    Returns
    -------
    tuple of 2 arrays
        Returns the partition and the computed control policy on that partition.
    """
    N = partition.shape[0]
    hidden_size = mpc_pars["pwqnn.input_layer.weight"].shape[0]
    mpc, _ = create_mpc(**mpc_kwargs, hidden_size=hidden_size)

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


def compute_learned_policy(
    grid: npt.NDArray[np.floating],
    args: dict[str, Any],
    mpc_params: dict[str, npt.NDArray[np.floating]],
    n_jobs: int,
) -> npt.NDArray[np.floating]:
    """Computes the learning-based MPC policy for the given grid in parallel.

    Parameters
    ----------
    grid : array
        1D array of grid points.
    args : dict of any
        The arguments used for the simulation.
    mpc_params : dict of arrays
        The NN weights for the MPC controller.
    n_jobs : int
        Number of parallel processes.

    Returns
    -------
    array
        The learned control policy for each point of the 2D grid.
    """
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


def do_plot(data: list[tuple[npt.NDArray[np.floating], ...]], log: bool) -> None:
    """Plots the comparison of the optimal infinite-horizon value function and policy
    for the constrained LTI environment w.r.t. the learned ones.

    Parameters
    ----------
    data : list of tuples of 3 arrays
        The data to plot. Each tuple must contain the grid and the difference between
        the optimal and learned value functions and policies.
    log : bool
        Whether to plot the value function in log scale.
    """
    vf_min = min(np.nanmin(delta) for _, delta, _ in data)
    vf_max = max(np.nanmax(delta) for _, delta, _ in data)
    pol_min = min(np.nanmin(delta) for _, _, delta in data)
    pol_max = max(np.nanmax(delta) for _, _, delta in data)

    fig, axs = plt.subplots(
        len(data), 3, constrained_layout=True, sharex=True, sharey=True
    )
    axs = np.atleast_2d(axs)
    cmap = "RdBu_r"
    if not log:
        vf_norm = colors.Normalize(vmin=vf_min, vmax=vf_max)
    else:
        vf_norm = colors.SymLogNorm(1.0, vmin=vf_min, vmax=vf_max)
    pol_norm = colors.Normalize(vmin=pol_min, vmax=pol_max)
    vf_kwargs = {"norm": vf_norm, "cmap": cmap}
    pol_kwargs = {"norm": pol_norm, "cmap": cmap}

    for i, (grid, vf_delta, pol_delta) in enumerate(data):
        ax1, ax2, ax3 = axs[i, :]
        X1, X2 = np.meshgrid(grid, grid)
        ax1.contourf(X1, X2, vf_delta, **vf_kwargs)
        ax2.contourf(X1, X2, pol_delta[..., 0], **pol_kwargs)
        ax3.contourf(X1, X2, pol_delta[..., 1], **pol_kwargs)
        for ax in axs[i, :]:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_aspect("equal")
        if i == 0:
            ax1.set_title("Error in $V(x)$")
            ax2.set_title("Error in $u_1(x)$")
            ax3.set_title("Error in $u_2(x)$")

    for norm, axs in ((vf_norm, ax1), (pol_norm, (ax2, ax3))):
        fig.colorbar(cm.ScalarMappable(norm, cmap), ax=axs, orientation="horizontal")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Comparison of learned value function and policy w.r.t. optimal.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Results to compare")
    group.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Filenames with training results whose value functions and policies are to"
        " be compared to the explicit optimal ones.",
    )
    group.add_argument(
        "agent", type=int, help="Index of the agent to compare the results for."
    )
    group = parser.add_argument_group("Other options")
    group.add_argument("--log", action="store_true", help="Plots in log scale.")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes (positive)."
    )
    args = parser.parse_args()
    idx = args.agent
    n_jobs = max(args.n_jobs, 1)

    # do first the computations
    results = []
    for fn in args.filenames:
        # load the data from the file - we need training args and final learned params
        data = load(fn)
        sim_args = data.pop("args")
        mpc_pars = {n: p[idx, -1] for n, p in data["updates_history"].items()}
        print(fn.upper(), f"Args: {sim_args}\n", sep="\n")

        # load the corresponding optima and compute the learned ones
        grid, opt_valfun, opt_policy = load_explicit(sim_args["dcbf"], sim_args["soft"])
        learned_valfun = compute_learned_value_function(grid, sim_args, mpc_pars)
        learned_policy = compute_learned_policy(grid, sim_args, mpc_pars, n_jobs)

        # compute the difference between the two value functions and policies
        vf_delta = learned_valfun - opt_valfun
        pol_delta = learned_policy - opt_policy
        results.append((grid, vf_delta, pol_delta))

    # then, do the plotting itself
    do_plot(results, args.log)
    plt.show()
