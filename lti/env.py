from typing import Any, TypeAlias

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from mpcrl.util.control import dcbf
from mpcrl.util.geometry import ConvexPolytopeUniformSampler

from util.defaults import DCBF_GAMMA
from util.loose_box import LooseBox

ObsType: TypeAlias = npt.NDArray[np.floating]
ActType: TypeAlias = npt.NDArray[np.floating]


MAX_INV_SET_V = 0.99 * np.asarray(  # computed with MATLAB MPT3 toolbox
    [
        [2.5962, 2.3221],
        [2.0312, 3.0000],
        [3.0000, 1.3125],
        [3.0000, -3.0000],
        [-3.0000, 3.0000],
        [-3.0000, -1.3125],
        [-2.0313, -3.0000],
        [-2.5962, -2.3221],
    ]
)


class ConstrainedLtiEnv(gym.Env[ObsType, ActType]):
    """
    ## Description

    This environment simulates a simple discrete-time constrained LTI system.

    ## Observation Space

    The observation (a.k.a., state) space is an array of shape `(ns,)`, where `ns = 2`.
    States are softly constrained to be within the interval `[-3, 3]`. If that's not the
    case, a penalty is incurred based on `constraint_penalty`.

    ## Action Space

    The action is an array of shape `(na,)`, where `na = 2`, bounded to `[-0.5, 0.5]`.

    ## Disturbances

    There is a disturbance acting on the system's dynamics.

    ## Rewards/Costs

    The reward here is intended as an linear quadratic regulation (LQR) cost, thus it
    must be minized, with the goal being to regulate the system to the origin. The cost
    is computed according to the current state and action (based on cost matrices `Q`
    and `R`), but it also includes a penalisation in case the agent exits the hypercube
    state soft bounds.

    ## Starting States

    The initial state is randomly uniformly selected within the polytopic state
    boundary, so it is guaranteed to be initially safe.

    ## Episode End

    The episode ends when `max_timesteps` time steps are reached.
    """

    A = np.asarray([[1.0, 0.4], [-0.1, 1.0]])
    B = np.asarray([[1.0, 0.05], [0.5, 1.0]])
    D = np.asarray([[0.03], [0.01]])
    ns, na = B.shape
    nd = D.shape[1]
    Q = np.ones(ns)
    R = np.full(na, 0.1)
    a_bound = 0.5
    x_soft_bound = 3.0
    constraint_penalty = 1e3

    def __init__(self, max_timesteps: int) -> None:
        super().__init__()
        a_max = self.a_bound
        x_max = self.x_soft_bound
        self.observation_space = LooseBox(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = LooseBox(-a_max, a_max, (self.na,), np.float64)
        self._sampler = ConvexPolytopeUniformSampler(MAX_INV_SET_V)
        self._max_timesteps = max_timesteps

        # build also the symbolic dynamics and safety constraints
        x = cs.MX.sym("x", self.ns)
        u = cs.MX.sym("u", self.na)
        x_next = self.A @ x + self.B @ u
        dynamics = cs.Function("f", [x, u], [x_next], ["x", "u"], ["x_next"])
        h = cs.veccat(x + x_max, x_max - x)  # >= 0
        safety_constraints = cs.Function("h", [x], [h], ["x"], ["h"])
        dcbf_ = dcbf(safety_constraints, x, u, dynamics, [lambda y: DCBF_GAMMA * y])
        dcbf_constraints = cs.Function("h", [x, u], [dcbf_], ["x", "u"], ["dcbf"])
        self.dynamics = dynamics
        self.safety_constraints = safety_constraints
        self.dcbf_constraints = dcbf_constraints

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._sampler.seed(self.np_random)
        reset_from = options.get("ic", "interior")
        if reset_from == "contour":
            x = self._sampler.sample_from_surface()
        elif reset_from == "interior":
            x = self._sampler.sample_from_interior()
            # for _ in range(100):
            #     x = self._sampler.sample_from_interior()
            #     if np.linalg.norm(x) >= 2.5:
            #         break
        else:
            points = self._sampler._qhull.points
            x = self.np_random.uniform(points.min(0), points.max(0), size=self.ns)
        assert self.observation_space.contains(x), f"invalid initial state {x}"
        self._x = x
        self._t = 0
        return x, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        u = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(u), f"invalid action {u}"
        x = self._x
        w = self.sample_disturbance_profiles(1, 1)[0]
        x_new = np.dot(self.A, x) + np.dot(self.B, u) + np.dot(self.D, w)
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"
        self._x = x_new
        self._t += 1
        truncated = self._t >= self._max_timesteps
        return x_new, self._compute_cost(x, u), False, truncated, {}

    def _compute_cost(self, x: ObsType, u: ActType) -> float:
        violations = np.maximum(0.0, -self.dcbf_constraints(x, u)).sum()
        return (
            (self.Q * np.square(x)).sum()
            + (self.R * np.square(u)).sum()
            + self.constraint_penalty * violations
        )

    def sample_disturbance_profiles(
        self, n: int, length: int | None = None
    ) -> npt.NDArray[np.floating]:
        """Samples i.i.d. disturbance profiles from the disturbance profile's
        distribution.

        Parameters
        ----------
        n : int
            The number of disturbance profiles to sample.
        length : int, optional
            The number of timesteps in each disturbance profile. If `None`, the
            maximum number of timesteps is used.

        Returns
        -------
        array
            An array of shape `(n, length)` containing the disturbance profiles.
        """
        if length is None:
            length = self._max_timesteps
        return self.np_random.normal(0.0, 1.0, size=(n, length))
