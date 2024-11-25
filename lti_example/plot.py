import argparse
import sys
from collections.abc import Collection
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import load
from csnn.convex import PwqNN
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpcrl.util.control import dlqr

lti_dir = Path(__file__).parent
sys.path.append(str(lti_dir.parent))

from env import ConstrainedLtiEnv as Env

from util.nn import nn2function
from util.visualization import plot_population, plot_single_violin

plt.style.use("seaborn-v0_8-pastel")
plt.rcParams["lines.linewidth"] = 0.75
plt.rcParams["lines.markersize"] = 6


def load_single_file(
    filename: str,
) -> tuple[dict[str, Any], dict[str, npt.NDArray[np.floating]]]:
    """Loads the data from a single file on disk.

    Parameters
    ----------
    filename : str
        The filename of the file to load.

    Returns
    -------
    tuple of two dictionaries
        Returns the arguments used to run the simualtion script, as well as the data
        itselft. Both of these are dictionaries.
    """
    data = load(filename)
    args = data.pop("args")
    return args, data


def plot_states_and_actions(
    data: Collection[dict[str, npt.NDArray[np.floating]]], *_: Any, **__: Any
) -> None:
    """Plots the state trajectories and actions of the simulations. This plot does not
    distinguish between different agents as it plots them all together.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the keys
        `"actions"` and `"states"`.
    """
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    x_max = Env.x_soft_bound
    ax1.add_patch(Rectangle((-x_max, -x_max), 2 * x_max, 2 * x_max, fill=False))

    for i, datum in enumerate(data):
        returns = datum["cost"]  # n_agents x n_ep
        actions = datum["actions"]  # n_agents x n_ep x timesteps x na
        states = datum["states"]  # n_agents x n_ep x timesteps + 1 x ns
        c = f"C{i}"
        timesteps = actions.shape[2]
        time = np.arange(timesteps)

        # flatten the first two axes as we do not distinguish between different agents
        returns = returns.reshape(-1)
        actions = actions.reshape(-1, timesteps, Env.na)
        states = states.reshape(-1, timesteps + 1, Env.ns)

        for state_traj, action_traj in zip(states, actions):
            ax1.plot(*state_traj.T, c)
            violating = (np.abs(state_traj) > x_max + 1e-3).any(1)
            ax1.plot(*state_traj[violating].T, "r", ls="none", marker="x")
            ax2.step(time, action_traj[:, 0], c, where="post")
            ax3.step(time, action_traj[:, 1], c, where="post")

    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")
    ax2.set_ylabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_xlabel("$k$")


def plot_returns(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    names: Collection[str] | None = None,
) -> None:
    """Plots the returns of the simulations.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the key `"cost"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    _, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

    for i, datum in enumerate(data):
        c = f"C{i}"
        returns = datum["cost"]  # n_agents x n_ep
        episodes = np.arange(returns.shape[1])
        # print(f"cost: {np.mean(returns)} ± {np.std(returns)} | {np.median(returns)}")
        # in the first axis, flatten the first two axes as we do not distinguish between
        # different agents
        returns_flat = returns.reshape(-1)
        plot_single_violin(
            ax1,
            i,
            returns_flat,
            showextrema=False,
            quantiles=[0.25, 0.5, 0.75],
            color=c,
        )
        plot_population(ax2, episodes, returns, axis=0, color=c)

    ax1.set_xticks(range(len(data)))
    if names is not None:
        ax1.set_xticklabels(names)
    ax1.set_ylabel("Return")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Return")


def plot_solver_times(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    names: Collection[str] | None = None,
) -> None:
    """Plots the solver times of the simulations. For agents with a solver for the
    state and action value functions, their computation times are plotted separately.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the key
        `"sol_times"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    _, ax = plt.subplots(1, 1, constrained_layout=True)

    for i, datum in enumerate(data):
        c = f"C{i}"
        sol_times = datum["sol_times"]  # n_agents x n_ep x timestep (x 2, optional)
        kw = {"showextrema": False, "quantiles": [0.25, 0.5, 0.75], "color": c}

        # flatten the first three axes as we do not distinguish between different
        # agents, episodes or time steps
        sol_times_flat = sol_times.reshape(-1, *sol_times.shape[3:])
        # print(f"sol. time: {np.mean(st):e} ± {np.std(st):e} | {np.median(st):e}")
        if sol_times_flat.ndim == 1:
            plot_single_violin(ax, i, sol_times_flat, **kw)
        else:
            assert sol_times_flat.ndim == 2, "Unexpected shape of `sol_times_flat`"
            sol_times_V, sol_times_Q = sol_times_flat.T
            for side, times in (("low", sol_times_V), ("high", sol_times_Q)):
                plot_single_violin(ax, i, times, side=side, **kw)

    ax.set_xticks(range(len(data)))
    if names is not None:
        ax.set_xticklabels(names)
    ax.set_ylabel("Solver time [s]")
    ax.set_yscale("log")


def plot_training(
    data: Collection[dict[str, npt.NDArray[np.floating]]], *_: Any, **__: Any
) -> None:
    """Plots the training results of the simulations. This plot does not show anything
    if the data does not contain training results.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each potentially containing the
        keys `"updates_history"`, `"td_errors"`, or `"policy_performances"`.
    """
    param_names = set()
    for datum in data:
        if "updates_history" in datum:
            param_names.update(datum["updates_history"].keys())
    any_td = any("td_errors" in d for d in data)
    any_pg = any("policy_gradients" in d for d in data)
    any_other = any_td or any_pg
    n = len(param_names)
    if n == 0 and not any_other:
        return

    fig = plt.figure(constrained_layout=True)
    ncols = int(np.round(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    gs = GridSpec(nrows + int(any_other), ncols, fig)
    if any_other:
        offset = 1
        if any_td and any_pg:
            ax_td = fig.add_subplot(gs[0, :])
            ax_pg = ax_td.twinx()
        elif any_td:
            ax_td = fig.add_subplot(gs[0, :])
        else:
            ax_pg = fig.add_subplot(gs[0, :])
    else:
        offset = 0
    param_names = sorted(param_names)
    start = nrows * ncols - n
    ax_par_first = fig.add_subplot(gs[start // ncols + offset, start % ncols])
    ax_pars = {param_names[0]: ax_par_first}
    for i, name in enumerate(param_names[1:], start=start + 1):
        ax_pars[name] = fig.add_subplot(
            gs[i // ncols + offset, i % ncols], sharex=ax_par_first
        )

    for i, datum in enumerate(data):
        c = f"C{i}"

        if "td_errors" in datum:
            td_errors = datum["td_errors"]  # n_agents x n_ep x timestep
            td_errors = np.abs(td_errors).sum(2)
            episodes = np.arange(td_errors.shape[1])
            plot_population(ax_td, episodes, td_errors, axis=0, color=c)
        elif "policy_gradients" in datum:
            n_ep = datum["cost"].shape[1]
            grads = datum["policy_gradients"]  # n_agents x n_up x n_pars
            norms = np.linalg.norm(grads, axis=2)
            updates = np.linspace(0, n_ep - 1, grads.shape[1])
            plot_population(ax_pg, updates, norms, axis=0, color=c, log=True)

        if "updates_history" in datum:
            for name, param in datum["updates_history"].items():
                updates = np.arange(param.shape[1])
                param = param.reshape(*param.shape[:2], -1)  # n_agents x n_up x ...
                ax = ax_pars[name]
                n = param.shape[2]
                alpha = 0.25 if n == 1 else 0
                for idx in range(param.shape[2]):
                    plot_population(
                        ax, updates, param[..., idx], axis=0, color=c, alpha=alpha
                    )

    if any_td:
        ax_td.set_xlabel("Episode")
        ax_td.set_ylabel(r"$\sum{|\delta|}$")
    if any_pg:
        ax_pg.set_xlabel("Episode")
        ax_pg.set_ylabel(r"$\| \nabla_\theta J(\pi_\theta) \|_2$")
    for name, ax in ax_pars.items():
        ax.set_xlabel("Episode")
        ax.set_ylabel(name)
        ax._label_outer_xaxis(skip_non_rectangular_axes=False)


def plot_terminal_cost_evolution(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    args: Collection[dict[str, Any]],
    *_: Any,
    **__: Any,
) -> None:
    """Plots the evolution of the terminal cost approximation in terms of normalized
    RMSE and R^2 coefficient. This plot does not show anything if the data does not
    contain training results.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each potentially containing the
        key `"updates_history"`.
    args : collection of dictionaries (str, Any)
        The arguments used to run the simulation scripts.
    """
    terminal_cost_components = [arg.get("terminal_cost", set()) for arg in args]
    if not any("pwqnn" in cmp for cmp in terminal_cost_components):
        return

    _, P = dlqr(Env.A, Env.B, Env.Q, Env.R)
    value_func_dir = lti_dir / "explicit_sol"
    value_func_cache = {}
    pwqnn_cache = {}

    ncols = int(np.round(np.sqrt(len(data))))
    nrows = int(np.ceil(len(data) / ncols))
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(nrows, nrows + ncols, fig)
    ax_nrmse = fig.add_subplot(gs[:, :nrows])
    ax_r_squared = ax_nrmse.twinx()
    axs_logres = [
        fig.add_subplot(gs[i // ncols, i % ncols + nrows], projection="3d")
        for i in range(nrows * ncols)
    ]
    r_sq_ls = (0, (5, 10))
    ep2idxs, logresiduals = [], []
    for i, (arg, tcosts, datum) in enumerate(zip(args, terminal_cost_components, data)):
        if "pwqnn" not in tcosts:
            continue

        # load the true value function
        filename = "data"
        if arg["dcbf"]:
            filename += "_dcbf"
        if arg["soft"]:
            filename += "_soft"
        filename += ".npz"
        if filename in value_func_cache:
            value_func_data = value_func_cache[filename]
        else:
            value_func_data = np.load(value_func_dir / filename)
            value_func_cache[filename] = value_func_data

        # compute value function evolution during learning
        grid = value_func_data["grid"]
        N = grid.size
        n_agents, n_ep = arg["n_agents"], arg["n_episodes"] + 1
        params_history = datum["updates_history"]

        X = np.asarray(np.meshgrid(grid, grid))
        Xf = X.reshape(Env.ns, -1)
        V = np.zeros((n_agents, n_ep, N, N))
        if "dlqr" in tcosts:
            V += (X.transpose(1, 2, 0).dot(P) * X.transpose(1, 2, 0)).sum(-1)
        if "pwqnn" in tcosts:
            hidden_features = params_history["pwqnn.input_layer.weight"].shape[2]
            if hidden_features in pwqnn_cache:
                pwqnn = pwqnn_cache[hidden_features]
            else:
                pwqnn = nn2function(PwqNN(Env.ns, hidden_features), prefix="pwqnn")
                pwqnn_cache[hidden_features] = pwqnn

            tot_ep_to_plot = 20
            n_jobs = cpu_count() // 4
            episodes = np.linspace(0, n_ep - 1, tot_ep_to_plot, dtype=int)
            ep2idx = dict(map(reversed, enumerate(episodes)))
            indices = filter(lambda ae: ae[1] in ep2idx, np.ndindex((n_agents, n_ep)))
            partitions = np.array_split(list(indices), n_jobs)

            def func(partition):
                n_elem = partition.shape[0]
                V_ = np.empty((n_elem, N, N))
                for i in range(n_elem):
                    agent, ep = partition[i]
                    w = {n: params_history[n][agent, ep] for n in pwqnn.name_in()[1:]}
                    partition[i] = agent, ep2idx[ep]
                    V_[i] = pwqnn(x=Xf, **w)["y"].toarray().reshape(N, N)
                return partition, V_

            data = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(func)(p) for p in partitions
            )
            V = np.empty((n_agents, len(episodes), N, N))
            for partition, V_ in data:
                V[partition[:, 0], partition[:, 1]] = V_

        # compute NRMSE and R^2
        V_true = value_func_data["V"]
        p2p = np.nanmax(V_true) - np.nanmin(V_true) + 1e-27
        residuals = np.square(V - V_true)
        rmse = np.sqrt(np.nanmean(residuals, (2, 3)))
        nrmse = rmse / p2p
        ss_total = np.nansum(np.square(V_true - np.nanmean(V_true)))
        ss_residual = np.nansum(residuals, (2, 3))
        r_squared = 1.0 - (ss_residual / ss_total)

        c = f"C{i}"
        episodes = np.sort(list(episodes))
        plot_population(ax_nrmse, episodes, nrmse, axis=0, color=c)
        plot_population(ax_r_squared, episodes, r_squared, axis=0, color=c, ls=r_sq_ls)

        # store the log of the residuals averaged over agents for later plotting
        ep2idxs.append(ep2idx)
        logresiduals.append(np.log(np.nanmean(residuals / np.square(p2p), 0) + 1e-27))

    # plot log of the residuals in a second loop
    cmap = plt.get_cmap("RdBu_r")
    range_ = np.linspace(0, 1, cmap.N)  # np.geomspace(1, 10, cmap.N) / 10.0
    colors = cmap(range_)
    colors[..., -1] = range_  # overwrite alphas
    transpcmap = LinearSegmentedColormap.from_list("transp_RdBu_r", colors, cmap.N)
    kw3d = {
        "cmap": transpcmap,
        "antialiased": True,
        "zdir": "z",
        "vmin": np.nanmin(logresiduals),
        "vmax": np.nanmax(logresiduals),
    }
    for ax_logres, ep2idx, logresidual in zip(axs_logres, ep2idxs, logresiduals):
        for ep, idx in ep2idx.items():
            ax_logres.contourf(*X, logresidual[idx], offset=ep, **kw3d)

    ax_nrmse.set_xlabel("Update")
    ax_nrmse.set_ylabel(r"$NRMSE$")
    ax_r_squared.set_ylabel(r"$R^2$")
    ax_nrmse.spines.right.set_visible(False)
    ax_r_squared.spines.right.set_linestyle(r_sq_ls)

    for ax_logres, arg in zip(axs_logres, args):
        ax_logres.set_xlabel("$x_1$")
        ax_logres.set_ylabel("$x_2$")
        ax_logres.set_zlabel("Episode")
        ax_logres.set_xlim(-Env.x_soft_bound, Env.x_soft_bound)
        ax_logres.set_ylim(-Env.x_soft_bound, Env.x_soft_bound)
        ax_logres.set_zlim(0, arg["n_episodes"] + 1)
        ax_logres.invert_zaxis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting of simulations of the constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        nargs="+",
        help="Filenames of the results on disk to load and plot.",
    )
    parser.add_argument(
        "--state-action",
        action="store_true",
        help="Plots the state and action trajectories.",
    )
    parser.add_argument(
        "--returns",
        action="store_true",
        help="Plots the returns across episodes.",
    )
    parser.add_argument(
        "--solver-time",
        action="store_true",
        help="Plots the recorded solver time.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Plots the evolution of the training process.",
    )
    parser.add_argument(
        "--terminal-cost",
        action="store_true",
        help="Plots the evolution of the terminal cost approximation.",
    )
    parser.add_argument("--all", action="store_true", help="Plots all visualizations.")
    args = parser.parse_args()
    if not any(
        (
            args.state_action,
            args.returns,
            args.solver_time,
            args.training,
            args.terminal_cost,
            args.all,
        )
    ):
        warn("No type of visualizations selected.", RuntimeWarning)

    sim_args = []
    data = []
    unique_names = []
    for filename in args.filename:
        if filename in unique_names:
            continue
        sim_arg, datum = load_single_file(filename)
        unique_names.append(filename)
        sim_args.append(sim_arg)
        data.append(datum)
        print(filename.upper(), f"Args: {sim_arg}\n", sep="\n")

    if args.all or args.state_action:
        plot_states_and_actions(data, unique_names)
    if args.all or args.returns:
        plot_returns(data, unique_names)
    if args.all or args.solver_time:
        plot_solver_times(data, unique_names)
    if args.all or args.training:
        plot_training(data, unique_names)
    if args.all or args.terminal_cost:
        from time import perf_counter

        t0 = perf_counter()
        plot_terminal_cost_evolution(data, sim_args)
        print(f"Time to plot terminal cost evolution: {perf_counter() - t0:.6f}s")
    plt.show()
