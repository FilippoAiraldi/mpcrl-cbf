import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from env import NavigationEnv as NavEnv
from gymnasium.wrappers import TimeLimit
from joblib import Parallel, delayed
from mpcrl.util.control import dlqr


def simulate_once(
    K: npt.NDArray[np.floating], timesteps: int, seed: int
) -> npt.NDArray[np.floating]:
    """Simulates one episode of the navigation environment using the LQR controller, and
    returns the state trajectory."""
    env = TimeLimit(NavEnv(), max_episode_steps=timesteps)
    x, _ = env.reset(seed=seed)
    X = [x]
    cost = 0
    terminated = truncated = False
    while not (terminated or truncated):
        u = -np.dot(K, x)
        x, c, terminated, truncated, _ = env.step(u)
        X.append(x)
        cost += c
    print("Simulation terminated with cost", cost)
    return np.asarray(X)


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Discrete-time LQR simulations for navigation environment.",
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

    # compute LQR policy only once
    K, _ = dlqr(NavEnv.A, NavEnv.B, NavEnv.Q, NavEnv.R)

    # run the simulations in parallel
    trajectories = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(simulate_once)(K, args.timesteps, args.seed + i)
        for i in range(args.n_sim)
    )

    # finally, plot the results
    fig, ax = plt.subplots(constrained_layout=True)
    NavEnv.plot(ax)

    # plot trajectories and violations
    positions = np.asarray([traj[..., :2] for traj in trajectories])
    collisions = NavEnv.collide(positions)
    for pos, coll in zip(positions, collisions):
        ax.plot(*pos.T, "C0", lw=1)
        # ax.plot(*pos[~coll].T, "C0", marker=".", markersize=4, lw=0)
        ax.plot(*pos[coll].T, "r", marker="x", markersize=4, lw=0)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    plt.show()
