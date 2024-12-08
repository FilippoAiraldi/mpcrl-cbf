import argparse
import sys
from collections.abc import Collection
from itertools import repeat
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnn.convex import PwqNN
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpcrl.util.control import dlqr
from scipy.stats import sem

lti_dir, repo_dir = Path(__file__).resolve().parents[:2]
sys.path.append(str(repo_dir))

from env import ConstrainedLtiEnv as Env

from util.nn import nn2function
from util.visualization import (
    load_file,
    plot_population,
    plot_returns,
    plot_solver_times,
    plot_training,
)

plt.style.use("seaborn-v0_8-pastel")
plt.rcParams["lines.linewidth"] = 0.75
plt.rcParams["lines.markersize"] = 6


def plot_states_and_actions(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    args: Collection[dict[str, Any]],
    names: Collection[str] | None = None,
    *_: Any,
    **__: Any,
) -> None:
    """Plots the state trajectories and actions of the simulations, as well as the
    constraint violations. For states and actions, this plot does not distinguish
    between different agents as it plots all trajectories together.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the keys
        `"actions"` and `"states"`.
    args : collection of dictionaries (str, Any)
        The arguments used to run the simulation scripts.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, fig)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs[2, :])

    env = Env(0)
    ns = env.ns
    na = env.na
    nc = env.dcbf_constraints.size1_out(0)
    x_max = env.x_soft_bound
    ax1.add_patch(Rectangle((-x_max, -x_max), 2 * x_max, 2 * x_max, fill=False))

    violations_data = []

    for i, (arg, datum) in enumerate(zip(args, data)):
        actions = datum["actions"]  # n_agents x n_ep x timesteps x na
        states = datum["states"]  # n_agents x n_ep x timesteps + 1 x ns
        c = f"C{i}"
        n_ag, n_ep, timesteps = actions.shape[:3]
        time = np.arange(timesteps)

        # flatten the first two axes as we do not distinguish between different agents
        actions_ = actions.reshape(-1, timesteps, na)
        states_ = states.reshape(-1, timesteps + 1, ns)
        for state_traj, action_traj in zip(states_, actions_):
            ax1.plot(*state_traj.T, c)
            violating = (np.abs(state_traj) > x_max + 1e-3).any(1)
            ax1.plot(*state_traj[violating].T, "r", ls="none", marker="x")
            ax2.step(time, action_traj[:, 0], c, where="post")
            ax3.step(time, action_traj[:, 1], c, where="post")

        states__ = states[..., :-1, :].reshape(-1, ns).T
        if not arg["dcbf"]:
            violations = env.safety_constraints(states__)
        else:
            actions__ = actions.reshape(-1, na).T
            violations = env.dcbf_constraints(states__, actions__)
        violations = -violations.toarray().T.reshape(n_ag, n_ep, timesteps, nc)
        prob_violations = (violations > 0.0).any(3).mean((1, 2)) * 100.0
        mean = prob_violations.mean(0)
        se = sem(prob_violations)
        violations_data.append((mean, se))

    violations_mean, violations_se = zip(*violations_data)
    ax4.bar(names, violations_mean, yerr=violations_se, capsize=5)

    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")
    ax2.set_ylabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_xlabel("$k$")
    ax4.set_ylabel("Num. of Violations (%)")


def plot_terminal_cost_evolution(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    args: Collection[dict[str, Any]],
    names: Collection[str] | None = None,
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
    names : collection of str, optional
        The names of the simulations to use in the plot.
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
        for i in range(len(data))
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

    if names is None:
        names = repeat(None)
    for ax_logres, arg, name in zip(axs_logres, args, names):
        ax_logres.set_xlabel("$x_1$")
        ax_logres.set_ylabel("$x_2$")
        ax_logres.set_zlabel("Episode")
        if name is not None:
            ax_logres.set_title(name)
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
        "filenames",
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
    for filename in args.filenames:
        if filename in unique_names:
            continue
        sim_arg, datum = load_file(filename)
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
        plot_terminal_cost_evolution(data, sim_args, unique_names)
    plt.show()
