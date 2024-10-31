import os
import sys
from typing import Any, TypeAlias, TypeVar

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from mpcrl.util.geometry import ConvexPolytopeUniformSampler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util.loose_box import LooseBox

ObsType: TypeAlias = npt.NDArray[np.floating]
ActType: TypeAlias = npt.NDArray[np.floating]
SymType = TypeVar("SymType", cs.SX, cs.MX)


MAX_INV_SET_V = np.asarray(  # computed with MATLAB MPT3 toolbox
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

    The episode does not have an end, so wrapping it in, e.g., `TimeLimit`, is strongly
    suggested.
    """

    A = np.asarray([[1.0, 0.4], [-0.1, 1.0]])
    B = np.asarray([[1.0, 0.05], [0.5, 1.0]])
    ns, na = B.shape
    Q = np.eye(ns)
    R = 0.1 * np.eye(na)
    a_bound = 0.5
    x_soft_bound = 3.0
    constraint_penalty = 1e3

    def __init__(self) -> None:
        super().__init__()
        a_max = self.a_bound
        x_max = self.x_soft_bound
        self.observation_space = LooseBox(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = LooseBox(-a_max, a_max, (self.na,), np.float64)
        self._sampler = ConvexPolytopeUniformSampler(MAX_INV_SET_V)

        # build also the symbolic dynamics and safety constraints
        x = cs.MX.sym("x", self.ns)
        u = cs.MX.sym("u", self.na)
        x_next = self.A @ x + self.B @ u
        h = cs.veccat(x + x_max, x_max - x)
        self.dynamics = cs.Function("f", [x, u], [x_next], ["x", "u"], ["x_next"])
        self.safety_constraints = cs.Function("h", [x], [h], ["x"], ["h"])

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
        self.x = x
        return x, {}

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
        lb_violation = np.maximum(0, -self.x_soft_bound - x).sum()
        ub_violation = np.maximum(0, x - self.x_soft_bound).sum()
        return (
            np.dot(np.dot(self.Q, x), x)
            + np.dot(np.dot(self.R, u), u)
            + self.constraint_penalty * (lb_violation + ub_violation)
        )
