import argparse
from collections.abc import Collection
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import load
from env import ConstrainedLtiEnv as Env
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

plt.style.use("seaborn-v0_8-pastel")


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


def plot_states_and_actions_and_return(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    names: Collection[str] | None = None,
) -> None:
    """Plots the state trajectories, actions, and returns of the simulations.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the keys `"cost"`,
        `"actions"`, `"states"`, and `"sol_times"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, fig)
    lw = 1.0
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs[2, 1])

    x_max = Env.x_soft_bound
    ax1.add_patch(Rectangle((-x_max, -x_max), 2 * x_max, 2 * x_max, fill=False))

    for i, datum in enumerate(data):
        returns = datum["cost"]  # n_ctrl x n_eval
        actions = datum["actions"]  # n_ctrl x n_eval x timesteps x na
        states = datum["states"]  # n_ctrl x n_eval x timesteps + 1 x ns
        c = f"C{i}"
        timesteps = actions.shape[2]
        time = np.arange(timesteps)

        # flatten the first two axes as we do not distinguish between different
        returns = returns.reshape(-1)
        actions = actions.reshape(-1, timesteps, Env.na)
        states = states.reshape(-1, timesteps + 1, Env.ns)

        for state_traj, action_traj in zip(states, actions):
            ax1.plot(*state_traj.T, c, lw=lw)
            violating = (np.abs(state_traj) > x_max + 1e-3).any(1)
            ax1.plot(*state_traj[violating].T, "r", ls="none", marker="x", ms=6)
            ax2.step(time, action_traj[:, 0], c, lw=lw, where="post")
            ax3.step(time, action_traj[:, 1], c, lw=lw, where="post")

        print(f"cost: {np.mean(returns)} ± {np.std(returns)} | {np.median(returns)}")
        st = datum["sol_times"]
        print(f"sol. time: {np.mean(st):e} ± {np.std(st):e} | {np.median(st):e}")

        VL = ax4.violinplot(returns, [i], showmedians=True, showextrema=False)
        perpline = VL["cmedians"].get_paths()[0]
        path = VL["bodies"][0].get_paths()[0].vertices
        median = perpline.vertices[0, 1]
        closest = np.abs(path[:, 1] - median).argmin()
        width = np.abs(path[closest, 0]) - i
        perpline.vertices[:, 0] = [i - width, i + width]
        pos = i + np.random.uniform(-0.01, 0.01, size=len(returns))
        ax4.scatter(pos, returns, s=10, facecolor="none", edgecolors=c)

    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")
    ax2.set_ylabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_xlabel("$k$")
    ax4.set_xticks(range(len(data)))
    if names is not None:
        ax4.set_xticklabels(names)
    ax4.set_ylabel("Cost")


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

    data = []
    unique_names = []
    for filename in args.filename:
        if filename in unique_names:
            continue
        sim_args, datum = load_single_file(filename)
        unique_names.append(filename)
        data.append(datum)
        print(filename.upper(), f"Args: {sim_args}\n", sep="\n")

    plot_states_and_actions_and_return(data, unique_names)
    plt.show()
