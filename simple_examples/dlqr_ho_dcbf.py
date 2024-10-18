from typing import Any, TypeAlias

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from mpcrl.util.control import dcbf, dlqr

ObsType: TypeAlias = npt.NDArray[np.floating]
ActType: TypeAlias = npt.NDArray[np.floating]


class DiscreteTimeLqrEnv(gym.Env[ObsType, ActType]):
    ns = 4
    na = 2
    dt = 0.2
    A = np.asarray([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.asarray([[0, 0], [0, 0], [dt, 0], [0, dt]])
    Q = np.asarray([[10, 5, 0, 15], [5, 10, 0, 0], [0, 0, 10, 0], [15, 0, 0, 10]])
    # Q = 10.0 * np.eye(ns)
    R = np.eye(na)

    # obstacles' centers and radii
    center1 = (-4, -2.25)
    radii1 = (1.5, 1.5)
    center2 = (-1, -2.5)
    radii2 = (6.5, 3.5)

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = Box(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = Box(-np.inf, np.inf, (self.na,), np.float64)
        self.x: ObsType

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.x = np.asarray([-5, -5, 0, 0])
        assert self.observation_space.contains(self.x), f"invalid reset state {self.x}"
        return self.x, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        x = self.x
        u = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(u), f"invalid action {u}"

        cost = np.dot(np.dot(self.Q, x), x) + np.dot(np.dot(self.R, u), u)
        x_new = np.dot(self.A, x) + np.dot(self.B, u)
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"
        self.x = x_new
        return x_new, cost, False, False, {}

    def obstacle_constraints(
        self, y: np.ndarray | cs.SX
    ) -> tuple[np.ndarray, np.ndarray] | tuple[cs.SX, cs.SX]:
        h1 = (
            (y[0] - self.center1[0]) ** 2 / self.radii1[0] ** 2
            + (y[1] - self.center1[1]) ** 2 / self.radii1[1] ** 2
            - 1
        )
        h2 = (
            1
            - (y[0] - self.center2[0]) ** 2 / self.radii2[0] ** 2
            - (y[1] - self.center2[1]) ** 2 / self.radii2[1] ** 2
        )
        return h1, h2


# create env
env = DiscreteTimeLqrEnv()
timesteps = 100

# compute LQR policy
K, P = dlqr(env.A, env.B, env.Q, env.R)

# compute high-order discrete-time CBF filter
x = cs.SX.sym("x", env.ns)
u = cs.SX.sym("u", env.na)
u_nom = cs.SX.sym("u_nominal", env.na)
alphas = [lambda y: 0.9 * y] * 2
dynamics = lambda x_, u_: env.A @ x_ + env.B @ u_
cbf1 = dcbf(lambda x_: env.obstacle_constraints(x_)[0], x, u, dynamics, alphas)
cbf2 = dcbf(lambda x_: env.obstacle_constraints(x_)[1], x, u, dynamics, alphas)
qp = {
    "x": u,
    "p": cs.vertcat(x, u_nom),
    "f": cs.sumsqr(u - u_nom),
    "g": cs.vertcat(cbf1, cbf2),
}
opts = {
    "error_on_fail": True,
    "print_time": False,
    "verbose": False,
    "osqp": {"verbose": False},
}
solver = cs.qpsol("solver", "osqp", qp, opts)


def safety_filter(x: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
    res = solver(x0=u_nom, p=cs.vertcat(x, u_nom), lbg=0, ubg=np.inf)
    return res["x"].full().reshape(env.na)


# simulate
x, _ = env.reset()
S, A, R = [x], [], []
for _ in range(timesteps):
    action = -np.dot(K, x)
    # safe_action = action
    safe_action = safety_filter(x, action)
    x, cost, _, _, _ = env.step(safe_action)
    S.append(x)
    A.append(safe_action)
    R.append(cost)

print("Costs (optimal vs actual):", cs.bilin(P, S[0]), "vs", sum(R))

# plot
fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 2, fig)

K = np.arange(timesteps + 1)
S = np.asarray(S)
ax1 = fig.add_subplot(gs[0, 0])
ax1.step(K, S[:, 0], where="post")
ax1.set_ylabel("$x_1$")
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.step(K, S[:, 1], where="post")
ax2.set_xlabel("$k$")
ax2.set_ylabel("$x_2$")

ax3 = fig.add_subplot(gs[:2, 1])
ellipse1 = Ellipse(
    env.center1, env.radii1[0] * 2, env.radii1[1] * 2, facecolor="k", alpha=0.25
)
ellipse2 = Ellipse(env.center2, env.radii2[0] * 2, env.radii2[1] * 2, facecolor="w")
ax3.patch.set_facecolor("k")
ax3.patch.set_alpha(0.25)
ax3.add_patch(ellipse2)
ax3.add_patch(ellipse1)
ax3.plot(*S[0, :2], "o", color="C0")
ax3.plot(S[:, 0], S[:, 1], color="C0", marker="o", markersize=3)
ax3.plot(0, 0, "*", color="k")
ax3.set_xlabel("$x_1$")
ax3.set_ylabel("$x_2$")
ax3.set_aspect("equal")

A = np.asarray(A)
ax4 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax4.step(K[:-1], A[:, 0], where="post")
ax4.set_xlabel("$k$")
ax4.set_ylabel("$u_1$")

ax5 = fig.add_subplot(gs[2, 1], sharex=ax1)
ax5.step(K[:-1], A[:, 1], where="post")
ax5.set_xlabel("$k$")
ax5.set_ylabel("$u_2$")

plt.show()
