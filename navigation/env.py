from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.spatial import ConvexHull

ObsType: TypeAlias = npt.NDArray[np.floating]
ActType: TypeAlias = npt.NDArray[np.floating]


class NavigationEnv(gym.Env[ObsType, ActType]):
    """
    ## Description

    This environment simulates a simple discrete-time LTI system for a navigation task
    with obstacles. The system is a 2D point mass with double integrator dynamics, and
    the obstacles are represented as rectangular polytopes.

    ## Observation Space

    The observation (a.k.a., state) space is an array of shape `(ns,)`, where `ns = 4`.
    In order, these are the two positions and the two velocities.

    ## Action Space

    The action is an array of shape `(na,)`, where `na = 2`, whose elements correspond
    to the accelerations in the two directions.

    ## Rewards/Costs

    The reward here is intended as an linear quadratic regulation (LQR) cost, thus it
    must be minized, with the point mass navigating towards the origin. The cost is
    computed according to the current state and action (based on cost matrices `Q` and
    `R`), but it also includes a penalisation in case the agent collides with one of the
    obstacles in the environment (based on `collision_penalty`).

    ## Starting States

    The initial state is randomly uniformly selected within the a specific initial
    region.

    ## Episode End

    The episode does not have an end, so wrapping it in, e.g., `TimeLimit`, is strongly
    suggested.
    """

    # polytopic constraints of the form Ax <= b (if satisfied, the point is inside one
    # of the obstacles)
    obstacles = np.asarray(
        [
            [[-2, 1], [-2, -1], [-1, -1], [-1, 1]],
            [[-2, 1], [-2, 2], [1, 2], [1, 1]],
            [[-2, -1], [-2, -2], [1, -2], [1, -1]],
            [[2, -2], [2, 2], [3, 2], [3, -2]],
        ]
    )
    _qhulls = [ConvexHull(obs) for obs in obstacles]
    _equations = np.stack([qhull.equations for qhull in _qhulls], 0)
    Acon = _equations[..., :-1]
    bcon = -_equations[..., -1, None]

    initial_states = np.asarray([[-5.5, -1, 0, 0], [-4.5, 1, 0, 0]])

    ns: int = 4
    na: int = 2
    dt = 0.2
    A = np.asarray([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.asarray([[0, 0], [0, 0], [dt, 0], [0, dt]])
    Q = 10.0 * np.eye(ns)
    R = np.eye(na)

    def __init__(self, collision_penalty: float = 1e2) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.na,), np.float64)
        self.collision_penalty = collision_penalty

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.x = self.np_random.uniform(*self.initial_states, size=self.ns)
        assert self.observation_space.contains(self.x), f"invalid reset state {self.x}"
        return self.x, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        x = self.x
        u = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(u), f"invalid action {u}"
        x_new = np.dot(self.A, x) + np.dot(self.B, u)
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"
        self.x = x_new
        return x_new, self._compute_cost(x, u), False, False, {}

    def _compute_cost(self, x: ObsType, u: ActType) -> float:
        cost = np.dot(np.dot(self.Q, x), x) + np.dot(np.dot(self.R, u), u)
        # loop over all obstacles, and if inside one, penalise by a high value and break
        if self.collide(x[:2]):
            cost += self.collision_penalty
        return cost

    @staticmethod
    def collide(pos: npt.NDArray[np.floating]) -> npt.NDArray[np.bool]:
        """Checks if the given position is inside any of the obstacles.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            A batch of positions of shape `(..., 2)`.

        Returns
        -------
        array of bools
            A boolean array indicating whether each position is inside any of the
            obstacles.
        """
        con_values = NavigationEnv.Acon @ pos[..., None, :, None] - NavigationEnv.bcon
        return (con_values.squeeze(-1) <= 0).all(-1).any(-1)

    @staticmethod
    def plot(ax: Axes) -> None:
        """Plots the environment.

        Parameters
        ----------
        ax : Axes
            Axis object to plot on.
        """
        from matplotlib.patches import Polygon, Rectangle

        # plot obstacles
        for obs in NavigationEnv.obstacles:
            ax.add_patch(Polygon(obs, fill=True, color="lightgray"))

        # plot starting region and target point
        init_pos = NavigationEnv.initial_states[:, :2]
        ax.add_patch(
            Rectangle(
                init_pos[0],
                *np.diff(init_pos, axis=0).flat,
                fill=True,
                color="C0",
                alpha=0.25,
            )
        )
        ax.plot(0, 0, "k*", markersize=10, zorder=1000)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(constrained_layout=True)
    NavigationEnv.plot(ax)

    # plot cluster of points
    n = 100
    X = np.stack(
        np.meshgrid(np.linspace(-6, 4, n), np.linspace(-5, 5, n)), 0
    ).transpose(1, 2, 0)
    violations = NavigationEnv.collide(X)
    ax.plot(*X[violations].T, "ko", markersize=2)
    ax.plot(*X[~violations].T, "yo", markersize=2)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    plt.show()
