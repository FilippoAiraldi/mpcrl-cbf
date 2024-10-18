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


def plot_states_and_actions(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
) -> None:
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, fig)
    lw = 1.0
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    x_max = Env.x_soft_bound
    ax1.add_patch(Rectangle((-x_max, -x_max), 2 * x_max, 2 * x_max, fill=False))

    for i, datum in enumerate(data):
        states = datum["states"]  # n_sim x timesteps + 1 x ns
        actions = datum["actions"]  # n_sim x timesteps x na
        timesteps = np.arange(actions.shape[1])
        c = f"C{i}"
        ax1.plot(*states[:, 0].T, c, ls="none", marker=".", markersize=3)
        for state_trajectory, action_trajectory in zip(states, actions):
            ax1.plot(*state_trajectory.T, c, lw=lw)
            ax2.step(timesteps, action_trajectory[:, 0], c, lw=lw, where="post")
            ax3.step(timesteps, action_trajectory[:, 1], c, lw=lw, where="post")

    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")
    ax2.set_ylabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_xlabel("$k$")


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
    for filename in set(args.filename):
        sim_args, datum = load_single_file(filename)
        data.append(datum)
        print(filename.upper(), f"Args: {sim_args}\n", sep="\n")

    plot_states_and_actions(data)
    plt.show()
