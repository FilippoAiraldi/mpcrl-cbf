import argparse
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnn import ReLU, Sigmoid
from csnn.feedforward import Mlp
from matplotlib.gridspec import GridSpec
from scipy.stats import sem

sys.path.append(str(Path(__file__).resolve().parents[1]))

from env import QuadrotorEnv as Env

from util.nn import nn2function
from util.visualization import (
    load_file,
    plot_cylinder,
    plot_returns,
    plot_solver_times,
    plot_training,
)

plt.style.use("bmh")
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
    if not any("states" in d and "actions" in d and "obstacles" in d for d in data):
        return
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

    ns = Env.ns
    na = Env.na
    for i, ax in enumerate(axs_states):
        ax.axhline(Env.xf[i], color="k", ls="--")

    for i, datum in enumerate(data):
        if "actions" not in datum or "states" not in datum or "obstacles" not in datum:
            continue
        actions = datum["actions"]  # n_agents x n_ep x timesteps x na
        states = datum["states"]  # n_agents x n_ep x timesteps + 1 x ns
        # print(np.mean(states, (0, 1, 2)), np.std(states, (0, 1, 2)))
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
        r = Env.radius_obstacles
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


def plot_action_bounds(
    data: Collection[dict[str, npt.NDArray[np.floating]]], *_: Any, **__: Any
) -> None:
    """Plots the tilt and tilt rate bounds of the quadrotor actions. This plot does not
    distinguish between different agents as it plots them all together.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the keys
        `"actions"`.
    """
    if not any("actions" in d for d in data):
        return
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)

    na = Env.na
    tiltmax = Env.tiltmax
    dtiltmax = Env.dtiltmax
    dt = Env.sampling_time

    axs[0].axhline(tiltmax, color="k", ls="--")
    for i in (1, 2):
        axs[i].axhline(dtiltmax, color="k", ls="--")
        axs[i].axhline(-dtiltmax, color="k", ls="--")

    for i, datum in enumerate(data):
        if "actions" not in datum:
            continue
        actions = datum["actions"]  # n_agents x n_ep x timesteps x na
        c = f"C{i}"
        timesteps = actions.shape[2]
        time = np.arange(timesteps)

        # flatten the first two axes as we do not distinguish between different agents
        actions_ = actions.reshape(-1, timesteps, na)
        for action_traj in actions_:
            u = action_traj[:, [1, 2]]
            u_prev = np.insert(u[:-1], 0, Env.a0[[1, 2]], 0)
            du = (u - u_prev) / dt
            axs[0].step(time, np.prod(np.cos(u), 1), c, where="post")
            axs[1].step(time, du[:, 0], c, where="post")
            axs[2].step(time, du[:, 1], c, where="post")

    axs[-1].set_xlabel("$k$")
    axs[0].set_ylabel(r"$cos(\phi) cos(\theta)$ [tilt]")
    axs[1].set_ylabel(r"$\Delta t^{-1} (\phi_k - \phi_{k-1})$")
    axs[2].set_ylabel(r"$\Delta t^{-1} (\theta_k - \theta_{k-1})$")


def plot_safety(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    names: Collection[str] | None = None,
    *_: Any,
    **__: Any,
) -> None:
    """Plots the safety of the quadrotor trajectories w.r.t. the obstacles in the
    simulations. This plot does not distinguish between different agents as it plots
    them all together.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the keys
        `"states"` and `"obstacles"`, as well as optionally `"kappann_weights"` and
        `"updates_history"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    if not any("states" in d and "obstacles" in d for d in data):
        return
    env = Env(0)
    n_obs = env.n_obstacles
    safety = env.safety_constraints
    kappann_cache = {}

    plot_gamma = any("weights" in d for d in data)
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(n_obs + 1, 2 if plot_gamma else 1, fig)
    axs_h = [fig.add_subplot(gs[i, 0]) for i in range(n_obs)]
    if plot_gamma:
        axs_gamma = [fig.add_subplot(gs[i, 1]) for i in range(n_obs)]
    ax_viol_prob = fig.add_subplot(gs[-1, :])
    ax_viol_tot = ax_viol_prob.twinx()

    for ax in axs_h:
        ax.axhline(0.0, color="k", ls="--")
    if plot_gamma:
        for ax in axs_gamma:
            ax.axhline(0.0, color="k", ls="--")
            ax.axhline(1.0, color="k", ls="--")

    violations_data = []
    for i, datum in enumerate(data):
        if "states" not in datum or "obstacles" not in datum:
            continue
        states = datum["states"]  # n_agents x n_ep x timesteps + 1 x ns
        obs = datum["obstacles"]  # n_agents x n_ep x 2 (pos & dir) x 3d x n_obstacles
        c = f"C{i}"
        n_agents, n_ep, timesteps, ns = states.shape
        time = np.arange(timesteps)

        violations = np.empty((n_agents, n_ep))
        violating = np.empty((n_agents, n_ep, timesteps), dtype=np.bool)
        for n_a in range(n_agents):
            for n_e in range(n_ep):
                state_traj = states[n_a, n_e]
                pos_obs, dir_obs = obs[n_a, n_e]
                h = safety(state_traj.T, pos_obs, dir_obs).toarray()
                violating_ = h < 0
                for ax, h_, viol_ in zip(axs_h, h, violating_):
                    ax.plot(time, h_, c)
                    ax.plot(time[viol_], h_[viol_], "r", ls="none", marker="x", ms=3)
                violations[n_a, n_e] = np.maximum(0, -h).sum()
                violating[n_a, n_e] = violating_.any(0)

        prob_violating = violating.mean((1, 2)) * 100.0
        prob_mean = prob_violating.mean(0)
        prob_se = sem(prob_violating)
        tot_violations = violations.mean(1)
        tot_mean = tot_violations.mean(0)
        tot_se = sem(tot_violations)
        violations_data.append((prob_mean, prob_se, tot_mean, tot_se))

        # the rest is dedicated to plotting the output of the Kappa neural function
        if "kappann_weights" not in datum and "updates_history" not in datum:
            continue
        is_eval = "kappann_weights" in datum
        kappann_weights = (
            datum["kappann_weights"]
            if is_eval
            else {
                n: w
                for n, w in datum["updates_history"].items()
                if n.startswith("kappann.")
            }
        )

        features = [
            w.shape[-1] for n, w in kappann_weights.items() if n.endswith(".weight")
        ] + [n_obs]
        assert features[0] == ns + n_obs * 6, "Invalid input features."
        if features in kappann_cache:
            nnfunc = kappann_cache[features]
        else:
            kappann = Mlp(features, [ReLU] * (len(features) - 2) + [Sigmoid])
            nnfunc = nn2function(kappann, "kappann")
            kappann_cache[features] = nnfunc

        for a in range(n_agents):
            if is_eval:
                kappann_weights_ = {n: w[a] for n, w in kappann_weights.items()}

            for e in range(n_ep):
                if not is_eval:
                    kappann_weights_ = {n: w[a, e] for n, w in kappann_weights.items()}

                state_traj = states[a, e]
                pos_obs, dir_obs = obs[a, e]
                pos_obs_ = pos_obs.reshape(1, -1, order="F").repeat(timesteps, 0)
                dir_obs_ = dir_obs.reshape(1, -1, order="F").repeat(timesteps, 0)
                context = np.concatenate((state_traj, pos_obs_, dir_obs_), 1)
                gammas = nnfunc(x=context.T, **kappann_weights_)["y"].toarray()
                for ax, gamma in zip(axs_gamma, gammas):
                    ax.plot(time, gamma, c)

    width = 0.4
    prob_mean, prob_se, viol_mean, viol_se = zip(*violations_data)
    x = np.arange(len(names))
    rects = ax_viol_prob.bar(x, prob_mean, width, yerr=prob_se, capsize=5, color="C0")
    ax_viol_prob.bar_label(rects, padding=3)
    rects = ax_viol_tot.bar(
        x + width, viol_mean, width, yerr=viol_se, capsize=5, color="C1"
    )
    ax_viol_tot.bar_label(rects, padding=3)

    for ax in axs_h:
        ax.set_ylabel(f"$h_{i}$")
    axs_h[-1].set_xlabel("$k$")
    if plot_gamma:
        for ax in axs_gamma:
            ax.set_ylabel(f"$\\gamma_{i}$")
        axs_gamma[-1].set_xlabel("$k$")
    ax_viol_prob.set_xticks(x + width / 2, names)
    ax_viol_prob.set_ylabel("Num. of Violations (%)")
    ax_viol_tot.set_ylabel("Tot. Violations")


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
        "--action-bounds", action="store_true", help="Plots the bounds of the actions."
    )
    parser.add_argument(
        "--safety",
        action="store_true",
        help="Plots the safety w.r.t. obstacles for each episode.",
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
            args.safety,
            args.returns,
            args.solver_time,
            args.training,
            args.all,
        )
    ):
        warn("No type of visualization selected.", RuntimeWarning)

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
    if args.all or args.action_bounds:
        plot_action_bounds(data, unique_names)
    if args.all or args.safety:
        plot_safety(data, unique_names)
    if args.all or args.returns:
        plot_returns(data, unique_names)
    if args.all or args.solver_time:
        plot_solver_times(data, unique_names)
    if args.all or args.training:
        plot_training(data, unique_names)
    if plt.get_fignums():
        plt.show()
