import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from env import ConstrainedLtiEnv as Env
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpcrl.util.control import dlqr


def simulate_once(
    K: npt.NDArray[np.floating], timesteps: int, seed: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Simulates one episode of the navigation environment using the DLQR controller
    with saturated actions, and returns the state trajectory and actions."""
    env = Env()
    x, _ = env.reset(seed=seed)
    lba = env.action_space.low
    uba = env.action_space.high
    X = np.empty((timesteps + 1, env.ns))
    X[0] = x
    U = np.empty((timesteps, env.na))
    for i in range(timesteps):
        u = np.clip(-np.dot(K, x), lba, uba)
        x, _, _, _, _ = env.step(u)
        X[i + 1] = x
        U[i] = u
    return X, U


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Simulations for the discrete-time LQR with action saturation "
        "applied to the constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-sim", type=int, default=100, help="Number of simulations.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30,
        help="Number of timesteps per each simulation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    args = parser.parse_args()

    # compute discrete-time LQR policy only once
    K, _ = dlqr(Env.A, Env.B, Env.Q, Env.R)

    # run the simulations in parallel asynchronously
    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_sim)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_once)(K, args.timesteps, int(seed)) for seed in seeds
    )
    state_data, action_data = map(np.asarray, zip(*data))

    # finally, plot the results
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, fig)
    lw = 1.0

    ax1 = fig.add_subplot(gs[:, 0])
    bounds = Env.x_soft_bounds
    ax1.add_patch(Rectangle(bounds[0], *(bounds[1] - bounds[0]), fill=False))
    for states in state_data:
        ax1.plot(*states.T, "C0", lw=lw, marker=".", markersize=0)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")

    timesteps = np.arange(args.timesteps)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    for actions in action_data:
        ax2.step(timesteps, actions[:, 0], "C0", lw=lw, where="post")
        ax3.step(timesteps, actions[:, 1], "C0", lw=lw, where="post")
    ax2.set_ylabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_xlabel("$k$")
    plt.show()
