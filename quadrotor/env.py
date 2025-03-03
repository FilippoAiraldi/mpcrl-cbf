from typing import Any, TypeAlias

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from scipy.stats import truncnorm

from util.loose_box import LooseBox

ObsType: TypeAlias = npt.NDArray[np.floating]
ActType: TypeAlias = npt.NDArray[np.floating]


class QuadrotorEnv(gym.Env[ObsType, ActType]):
    """
    ## Description

    This environment simulates a quadrotor reaching a target position while avoiding
    cylindrical obstacles.

    ## Observation Space

    The observation (a.k.a., state) space is an array of shape `(ns,)`, where `ns = 6`.
    The first three states are the positions of the quadrotor, and the last three are
    the velocities of the quadrotor. States are unbounded, but if the quadrotor position
    collides with one of the obstacles, a penalty is incurred based on
    `constraint_penalty`.

    ## Action Space

    The action is an array of shape `(na,)`, where `na = 4`, bounded according to
    `QuadrotorActionSpace`. The first action is the vertical acceleration, the second
    and third are the roll and pitch angles, and the fourth is the yaw rate.

    ## Disturbances

    There is a disturbance perturbing the quadrotor's control action before it is
    applied to the system.

    ## Rewards/Costs

    The reward here is intended as an linear quadratic regulation (LQR) cost, thus it
    must be minized, with the goal being to regulate the quadrotor to the final
    destination `xf`. The cost is computed according to the current state and action
    (based on cost matrices `Q` and `R`), but it also includes a penalization in case
    the agent collides with one of the obstacles.

    ## Starting States

    The initial state is `x0`.

    ## Episode End

    The episode ends when `max_timesteps` time steps are reached.
    """

    # dimensions
    ns = 6
    na = nd = 4

    # initial, final, mean and std of states
    x0 = np.asarray([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    xf = np.asarray([12.0, 12.0, 12.0, 0.0, 0.0, 0.0])

    # default action and action space bounds
    a0 = np.asarray([9.81, 0.0, 0.0, 0.0])
    a_lb = np.asarray([0.0, -np.pi / 2, -np.pi / 2, -np.pi])
    a_ub = np.asarray([9.18 * 5, np.pi / 2, np.pi / 2, np.pi])
    tiltmax = np.cos(np.deg2rad(30))
    dtiltmax = 3.0

    # dynamics and cost matrices
    sampling_time = 5e-2
    Q = np.eye(ns)
    R = np.diag([1e-4, 1e-4, 1e-4, 1e2])

    # obstacles
    radius_obs = 3.3  # radius of the cylindrical obstacles + quadrotor size
    pos_obs = np.asarray([8, 8, 0], dtype=float)
    dir_obs = np.asarray([0, 0, 1], dtype=float)  # must be unit vector
    constraint_penalty = 1e3  # penalty for bumping into obstacles

    # noise
    action_noise_scale = (a_ub - a_lb) / 20.0

    def __init__(self, max_timesteps: int) -> None:
        super().__init__()
        self.observation_space = LooseBox(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = LooseBox(self.a_lb, self.a_ub, (self.na,), np.float64)
        self._max_timesteps = max_timesteps

        # build the symbolic dynamics
        x = cs.MX.sym("x", self.ns)
        u = cs.MX.sym("u", self.na)
        d = cs.MX.sym("d", self.nd)
        pos, vel = x[:3], x[3:]
        u_noisy = u + d
        az, phi, theta, psi = u_noisy[0], u_noisy[1], u_noisy[2], u_noisy[3]
        cphi, sphi = cs.cos(phi), cs.sin(phi)
        ctheta, stheta = cs.cos(theta), cs.sin(theta)
        cpsi, spsi = cs.cos(psi), cs.sin(psi)
        acc = cs.vertcat(
            (cpsi * stheta * cphi + spsi * sphi) * az,
            (spsi * stheta * cphi - cpsi * sphi) * az,
            (ctheta * cphi) * az - 9.81,
        )
        x_dot = cs.vertcat(vel, acc)
        self.dynamics = cs.Function(
            "dynamics", [x, u, d], [x_dot], ["x", "u", "d"], ["xf"], {"cse": True}
        )
        ode = {"x": x, "p": cs.vertcat(u, d), "ode": x_dot}
        self.integrator = cs.integrator("intg", "cvodes", ode, 0.0, self.sampling_time)

        # build the symbolic safety constraint for the cylindrical obstacle
        r2 = self.radius_obs**2
        h = cs.sumsqr(cs.cross(pos - self.pos_obs, self.dir_obs)) - r2  # >= 0
        self.safety_constraint = cs.Function("h", [x], [h], ["x"], ["h"])

    @property
    def previous_action(self) -> ActType:
        return self._u_prev

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        x = (
            np.asarray(options["ic"])
            if options is not None and "ic" in options
            else self.x0
        )
        assert (
            self.observation_space.contains(x) and self.safety_constraint(x) >= 0
        ), f"invalid initial state {x}"
        self._x = x
        self._t = 0
        self._dist_profile = self.sample_disturbance_profiles(1)[0]
        self._u_prev = self.a0
        return x, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        u = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(u), f"invalid action {u}"

        x = self._x
        d = self._dist_profile[self._t]
        x_new = self.integrator(x0=x, p=np.concat((u, d)))["xf"].full().flatten()
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"

        self._x = x_new
        self._t += 1
        self._u_prev = u
        truncated = self._t >= self._max_timesteps
        return x_new, self._compute_cost(x, u, x_new), False, truncated, {}

    def _compute_cost(self, x: ObsType, u: ActType, x_new: ObsType) -> float:
        # NOTE: for now, we penalize the violations of the least stringent CBF, i.e.,
        # when h(x_new) >= 0
        h = self.safety_constraint(x_new)
        cbf_violations = np.maximum(0.0, -h).sum()
        dx = x - self.xf
        return (
            np.dot(np.dot(self.Q, dx), dx)
            + np.dot(np.dot(self.R, u), u)
            + self.constraint_penalty * cbf_violations
        )

    def sample_disturbance_profiles(
        self, n: int, length: int | None = None
    ) -> npt.NDArray[np.floating]:
        """Samples i.i.d. action disturbance profiles from the corresponding
        distribution.

        Parameters
        ----------
        n : int
            The number of action disturbance profiles to sample.
        length : int, optional
            The number of timesteps in each actopm disturbance profile. If `None`, the
            maximum number of timesteps is used.

        Returns
        -------
        array
            An array of shape `(n, length, nd)` containing the action disturbance
            profiles.
        """
        if length is None:
            length = self._max_timesteps
        return truncnorm.rvs(
            self.a_lb,
            self.a_ub,
            scale=self.action_noise_scale,
            random_state=self.np_random,
            size=(n, length, self.nd),
        )


NORM_FACTORS = {
    "state_mean": np.asarray(
        [
            6.0236548373104934,
            6.151740949800862,
            8.93540450257289,
            1.7912016729274385,
            1.7951859511426536,
            1.6033835275569344,
        ]
    ),
    "state_std": np.asarray(
        [
            3.8305345178751216,
            3.867992622023662,
            3.444267861725363,
            1.0424283860886545,
            1.0149382660368655,
            1.5038041427261817,
        ]
    ),
    "action_mean": np.asarray(
        [
            10.895486281952921,
            -0.003770499934022998,
            0.004818233523290866,
            -0.000921798165769234,
        ]
    ),
    "action_std": np.asarray(
        [
            9.255642102619968,
            0.21482505173097652,
            0.23155736550237246,
            0.05589278934746118,
        ]
    ),
    "dist_mean": np.array([0.0]),
    "dist_std": np.array([396.61696249]),
}
NORMALIZATION = tuple(
    np.concatenate(
        [NORM_FACTORS[name + suffix] for name in ("state", "action", "dist")]
    )
    for suffix in ("_mean", "_std")
)
