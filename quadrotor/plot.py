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
    if not any("states" in d and "actions" in d for d in data):
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
    plot_cylinder(
        ax_3d, Env.radius_obs, 20, Env.pos_obs, Env.dir_obs, color="k", alpha=0.1
    )

    for i, datum in enumerate(data):
        if "actions" not in datum or "states" not in datum:
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
    """Plots the safety of the quadrotor trajectories w.r.t. the obstacle in the
    simulations. This plot does not distinguish between different agents as it plots
    them all together.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the key `"states"`,
        as well as optionally `"actions"` and Kappa NN weights in `"weights"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    if not any("states" in d for d in data):
        return
    env = Env(0)
    safety = env.safety_constraint
    kappann_cache = {}

    plot_gamma = any(
        "actions" in d and any(w.startswith("kappann.") for w in d.get("weights", {}))
        for d in data
    )
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2 if plot_gamma else 1, fig)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_h.axhline(0.0, color="k", ls="--")
    if plot_gamma:
        ax_gamma = fig.add_subplot(gs[0, 1])
        ax_gamma.axhline(0.0, color="k", ls="--")
        ax_gamma.axhline(1.0, color="k", ls="--")
    ax_viol_prob = fig.add_subplot(gs[1, :])
    ax_viol_tot = ax_viol_prob.twinx()

    violations_data = []
    for i, datum in enumerate(data):
        if "states" not in datum:
            continue
        states = datum["states"]  # n_agents x n_ep x timesteps + 1 x ns
        c = f"C{i}"
        n_agents, n_ep, timesteps, _ = states.shape
        time = np.arange(timesteps)

        violations = np.empty((n_agents, n_ep))
        violating = np.empty((n_agents, n_ep, timesteps), dtype=np.bool)
        for n_a in range(n_agents):
            for n_e in range(n_ep):
                state_traj = states[n_a, n_e]
                h = safety(state_traj.T).toarray().flatten()
                viols = h < 0
                ax_h.plot(time, h, c)
                ax_h.plot(time[viols], h[viols], "r", ls="none", marker="x", ms=3)
                violations[n_a, n_e] = np.maximum(0, -h).sum()
                violating[n_a, n_e] = viols

        prob_violating = violating.mean((1, 2)) * 100.0
        prob_mean = prob_violating.mean(0)
        prob_se = sem(prob_violating)
        tot_violations = violations.mean(1)
        tot_mean = tot_violations.mean(0)
        tot_se = sem(tot_violations)
        violations_data.append((prob_mean, prob_se, tot_mean, tot_se))

        # the rest is dedicated to plotting the output of the Kappa neural function -
        # this can happen only for evaluation files
        if (
            "actions" not in datum
            or "weights" not in datum
            or not any(w.startswith("kappann.") for w in datum["weights"])
        ):
            continue
        kweights = {
            n: w for n, w in datum["weights"].items() if n.startswith("kappann.")
        }
        features = tuple(
            w.shape[-1] for n, w in kweights.items() if n.endswith(".weight")
        ) + (1,)
        assert features[0] == Env.ns + Env.na + 1, "Invalid input features."
        if features in kappann_cache:
            nnfunc = kappann_cache[features]
        else:
            kappann = Mlp(features, [ReLU] * (len(features) - 2) + [Sigmoid])
            nnfunc = nn2function(kappann, "kappann")
            kappann_cache[features] = nnfunc

        actions = datum["actions"]  # n_agents x n_ep x timesteps x na
        actions_prev = np.insert(actions, 0, Env.a0, 2)
        for a in range(n_agents):
            kweights_ = {n: w[a] for n, w in kweights.items()}
            for e in range(n_ep):
                state_traj = states[a, e]
                action_prev_traj = actions_prev[a, e]
                h = safety(state_traj.T).toarray().reshape(-1, 1)
                context = np.concatenate((state_traj, action_prev_traj, h), 1).T
                gamma = nnfunc(x=context, **kweights_)["y"]
                ax_gamma.plot(time, gamma.toarray().flatten(), c)

    width = 0.4
    prob_mean, prob_se, viol_mean, viol_se = zip(*violations_data)
    x = np.arange(len(names))
    rects = ax_viol_prob.bar(x, prob_mean, width, yerr=prob_se, capsize=5, color="C0")
    ax_viol_prob.bar_label(rects, padding=3)
    rects = ax_viol_tot.bar(
        x + width, viol_mean, width, yerr=viol_se, capsize=5, color="C1"
    )
    ax_viol_tot.bar_label(rects, padding=3)

    ax_h.set_ylabel(f"$h_{i}$")
    ax_h.set_xlabel("$k$")
    if plot_gamma:
        ax_gamma.set_ylabel(f"$\\gamma_{i}$")
        ax_gamma.set_xlabel("$k$")
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
        help="Plots the safety w.r.t. the obstacle for each episode.",
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
