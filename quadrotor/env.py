from typing import Any, TypeAlias

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from mpcrl.util.control import rk4
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

    # initial, final states and bounds
    x0 = np.asarray([0.5, 0.0, 2.0, 0.0, 0.0, 0.0])
    xf = np.asarray([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    x_lb = np.full(ns, -10.0)
    x_ub = np.asarray([20.0, 20.0, 20.0, 10.0, 10.0, 10.0])

    # default action and action space bounds
    a0 = np.asarray([9.81, 0.0, 0.0, 0.0])
    a_lb = np.asarray([0.0, -np.pi / 2, -np.pi / 2, -np.pi])
    a_ub = np.asarray([9.18 * 5, np.pi / 2, np.pi / 2, np.pi])
    tiltmax = np.cos(np.deg2rad(30))
    dtiltmax = 3.0

    # dynamics and cost matrices
    sampling_time = 1e-1
    Q = np.ones(ns)
    R = np.asarray([1e-4, 1e-4, 1e-4, 1e2])

    # obstacles
    radius_obs = 2.5
    radius_quadrotor = 0.5
    pos_obs = np.asarray([5.0, 5.0, 0.0])
    dir_obs = np.asarray([0.0, 0.0, 1.0])  # must be unit vector
    constraint_penalty = 1e3  # penalty for bumping into obstacles

    # noise
    action_noise_scale = (a_ub - a_lb) / 20.0

    def __init__(self, max_timesteps: int) -> None:
        super().__init__()
        self.observation_space = LooseBox(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = LooseBox(self.a_lb, self.a_ub, (self.na,), np.float64)
        self._max_timesteps = max_timesteps

        # build the symbolic safety constraint for the cylindrical obstacle
        ns, na, nd = self.ns, self.na, self.nd
        x = cs.MX.sym("x", ns)
        u = cs.MX.sym("u", na)
        pos, vel = x[:3], x[3:]
        r2 = (self.radius_obs + self.radius_quadrotor) ** 2
        h = cs.sumsqr(cs.cross(pos - self.pos_obs, self.dir_obs)) - r2  # >= 0
        self.safety_constraint = cs.Function("h", [x], [h], ["x"], ["h"])

        # build the symbolic dynamics
        d = cs.MX.sym("d", nd)
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

        # continuous-time dynamics + integration (for simulation)
        dt = self.sampling_time
        self.dynamics = cs.Function(
            "dyn", [x, u, d], [x_dot], ["x", "u", "d"], ["xf"], {"cse": True}
        )
        self._integrator = cs.integrator(
            "intg", "cvodes", {"x": x, "p": cs.vertcat(u, d), "ode": x_dot}, 0.0, dt
        )

        # approximate discrete-time + linearized discrete-time dynamics (for control)
        x_next = cs.simplify(rk4(lambda x_: self.dynamics(x_, u, d), x, dt))
        self.dtdynamics = cs.Function(
            "dtdyn", [x, u, d], [x_next], ["x", "u", "d"], ["xf"], {"cse": True}
        )
        u_lin = cs.MX.sym("u_lin", na)
        A = cs.evalf(cs.jacobian(x_dot, x))  # constant!
        B = cs.substitute(
            cs.jacobian(x_dot, u), cs.vertcat(u, d), cs.vertcat(u_lin, cs.DM.zeros(nd))
        )
        I = np.eye(ns)
        A2 = A @ A
        A3 = A2 @ A
        Ad = cs.sparsify(I + dt * np.diag(np.ones(ns // 2), k=ns // 2))  # constant!
        Bd = (dt * I + dt**2 / 2 * A + dt**3 / 6 * A2 + dt**4 / 24 * A3) @ B  # RK4
        self.lindtdynamics = cs.Function(
            "lindtdyn", [u_lin], [Ad, Bd], ["u_lin"], ["A", "B"], {"cse": True}
        )

    @property
    def previous_action(self) -> ActType:
        return self._u_prev

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # NOTE: to ensure reproducibility independently of the timesteps and MPC dist.
        # sampling, we spawn two independent random number generators
        internal_rng, self._sampling_rng = self.np_random.spawn(2)

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
        self._dist_profile = self.sample_disturbance_profiles(1, rng=internal_rng)[0]
        self._u_prev = self.a0
        return x, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        u = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(u), f"invalid action {u}"

        x = self._x
        d = self._dist_profile[self._t]
        x_new = self._integrator(x0=x, p=np.concat((u, d)))["xf"].full().flatten()
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"

        self._x = x_new
        self._t += 1
        self._u_prev = u
        truncated = self._t >= self._max_timesteps
        return x_new, self._compute_cost(x, u), False, truncated, {}

    def _compute_cost(self, x: ObsType, u: ActType) -> float:
        cbf_violations = np.maximum(0.0, -self.safety_constraint(x)).sum()
        return (
            (self.Q * np.square(x - self.xf)).sum()
            + (self.R * np.square(u)).sum()
            + self.constraint_penalty * cbf_violations
        )

    def sample_disturbance_profiles(
        self, n: int, length: int | None = None, rng: np.random.Generator | None = None
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
        rng : np.random.Generator, optional
            The random number generator to use for sampling. If `None`, the
            environment's default generator is used.

        Returns
        -------
        array
            An array of shape `(n, length, nd)` containing the action disturbance
            profiles.
        """
        if length is None:
            length = self._max_timesteps
        if rng is None:
            rng = self._sampling_rng
        return truncnorm.rvs(
            self.a_lb,
            self.a_ub,
            scale=self.action_noise_scale,
            random_state=rng,
            size=(n, length, self.nd),
        )
