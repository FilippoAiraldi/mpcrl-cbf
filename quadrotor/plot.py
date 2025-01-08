import argparse
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec

sys.path.append(str(Path(__file__).resolve().parents[1]))

from env import QuadrotorEnv as Env

from util.visualization import (
    load_file,
    plot_cylinder,
    plot_returns,
    plot_solver_times,
    plot_training,
)

plt.style.use("seaborn-v0_8-pastel")
plt.rcParams["lines.linewidth"] = 0.75
plt.rcParams["lines.markersize"] = 6


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

    gs_states3d = gs[1, 0].subgridspec(1, 1)
    ax_3d = fig.add_subplot(gs_states3d[0, 0], projection="3d")
    n_arrows = 20

    gs_states = gs[0, :].subgridspec(2, 3)
    ax_px = fig.add_subplot(gs_states[0, 0])
    ax_py = fig.add_subplot(gs_states[0, 1], sharex=ax_px)
    ax_pz = fig.add_subplot(gs_states[0, 2], sharex=ax_px)
    ax_vx = fig.add_subplot(gs_states[1, 0], sharex=ax_px)
    ax_vy = fig.add_subplot(gs_states[1, 1], sharex=ax_px)
    ax_vz = fig.add_subplot(gs_states[1, 2], sharex=ax_px)
    axs_states = (ax_px, ax_py, ax_pz, ax_vx, ax_vy, ax_vz)

    gs_actions = gs[1, 1].subgridspec(2, 2)
    ax_az = fig.add_subplot(gs_actions[0, 0])
    ax_phi = fig.add_subplot(gs_actions[1, 0], sharex=ax_az)
    ax_theta = fig.add_subplot(gs_actions[0, 1], sharex=ax_az)
    ax_psi = fig.add_subplot(gs_actions[1, 1], sharex=ax_az)
    axs_action = (ax_az, ax_phi, ax_theta, ax_psi)

    env = Env(0)
    ns = env.ns
    na = env.na
    for i, ax in enumerate(axs_states):
        ax.axhline(env.xf[i], color="k", ls="--")

    for i, datum in enumerate(data):
        actions = datum["actions"]  # n_agents x n_ep x timesteps x na
        states = datum["states"]  # n_agents x n_ep x timesteps + 1 x ns
        print(np.mean(states, (0, 1, 2)), np.std(states, (0, 1, 2)))
        c = f"C{i}"
        timesteps = actions.shape[2]
        time = np.arange(timesteps + 1)

        # flatten the first two axes as we do not distinguish between different agents
        actions_ = actions.reshape(-1, timesteps, na)
        states_ = states.reshape(-1, timesteps + 1, ns)
        for state_traj, action_traj in zip(states_, actions_):
            pos = state_traj[..., :3].T
            vel = state_traj[..., -3:].T
            quiver_step = max(1, timesteps // n_arrows)
            ax_3d.plot(*pos, c)
            ax_3d.quiver(
                *pos[:, ::quiver_step], *vel[:, ::quiver_step], length=0.2, color=c
            )
            for i, ax in enumerate(axs_states):
                ax.plot(time, state_traj[..., i], c)
            for i, ax in enumerate(axs_action):
                ax.step(time[:-1], action_traj[..., i], c, where="post")

        # plot obstacles
        obs = datum["obstacles"]  # n_agents x n_ep x 2 (pos & dir) x 3d x n_obstacles
        obs = obs.reshape(-1, 2, 3, obs.shape[-1])
        obs_pos, obs_dir = obs[:, 0], obs[:, 1]
        r = env.radius_obstacles
        for poss, dirs in zip(obs_pos, obs_dir):
            for pos, dir in zip(poss.T, dirs.T):
                plot_cylinder(ax_3d, r, 20, pos, dir, color=c, alpha=0.1)

    ax_3d.set_xlabel("$p_x$")
    ax_3d.set_ylabel("$p_y$")
    ax_3d.set_zlabel("$p_z$")
    ax_3d.set_aspect("equal")
    ylbls = ("$p_x$", "$p_y$", "$p_z$", "$v_x$", "$v_y$", "$v_z$")
    for ax, ylbl in zip(axs_states, ylbls):
        ax.set_ylabel(ylbl)
    ax_az.set_ylabel("$a_z$")
    ax_phi.set_ylabel(r"$\phi$")
    ax_theta.set_ylabel(r"$\theta$")
    ax_psi.set_ylabel(r"$\psi$")
    for ax in (ax_phi, ax_psi, ax_vx, ax_vy, ax_vz):
        ax.set_xlabel("$k$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting of simulations of the quadrotor environment.",
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
    parser.add_argument("--all", action="store_true", help="Plots all visualizations.")
    args = parser.parse_args()
    if not any(
        (
            args.state_action,
            args.returns,
            args.solver_time,
            args.training,
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
    plt.show()
