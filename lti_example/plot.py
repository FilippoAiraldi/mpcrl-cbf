import argparse
import os
import sys
from collections.abc import Collection
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import load
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env import ConstrainedLtiEnv as Env

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
        keys `"updates_history"` and `"td_errors"`.
    """
    param_names = set()
    for datum in data:
        if "updates_history" in datum:
            param_names.update(datum["updates_history"].keys())
    any_td_errors = any("td_errors" in d for d in data)
    n = len(param_names)
    if n == 0 and not any_td_errors:
        return

    fig = plt.figure(constrained_layout=True)
    ncols = int(np.round(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    gs = GridSpec(nrows + int(any_td_errors), ncols, fig)
    if any_td_errors:
        ax_td = fig.add_subplot(gs[0, :])
        offset = 1
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

    if any_td_errors:
        ax_td.set_xlabel("Episode")
        ax_td.set_ylabel(r"$\sum{|\delta|}$")
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
    terminal_cost_components = [arg.get("terminal_cost", set()) for arg in args]
    if not any("pwqnn" in cmp for cmp in terminal_cost_components):
        return

    for i, (arg, datum) in enumerate(zip(args, data)):
        f"C{i}"
        # TODO


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
    args = parser.parse_args()

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

    # plot_states_and_actions(data, unique_names)
    # plot_returns(data, unique_names)
    # plot_solver_times(data, unique_names)
    # plot_training(data, unique_names)
    plot_terminal_cost_evolution(data, sim_args)
    plt.show()
