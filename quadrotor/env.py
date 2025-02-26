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
    xf = np.asarray([15.0, 15.0, 13.0, 0.0, 0.0, 0.0])

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
    n_obstacles = 3  # number of cylindrical obstacles
    radius_obstacles = 2.3  # radius of the cylindrical obstacles + quadrotor size
    pos_obs_mean = np.asarray([[2, 3, 0], [8, 0, 8], [12, 12, 0]], dtype=float).T
    pos_obs_std = 1.0
    dir_obs_mean = np.asarray([[0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=float).T
    dir_obs_mean /= np.linalg.norm(dir_obs_mean, axis=0)
    dir_obs_std = 0.1
    constraint_penalty = 1e3  # penalty for bumping into obstacles

    # noise
    action_noise_scale = (a_ub - a_lb) / 10.0

    def __init__(self, max_timesteps: int) -> None:
        super().__init__()
        self.observation_space = LooseBox(-np.inf, np.inf, (self.ns,), np.float64)
        self.action_space = LooseBox(self.a_lb, self.a_ub, (self.na,), np.float64)
        self._max_timesteps = max_timesteps

        # build the symbolic dynamics
        x = cs.MX.sym("x", self.ns)
        u = cs.MX.sym("u", self.na)
        pos, vel = x[:3], x[3:]
        az, phi, theta, psi = u[0], u[1], u[2], u[3]
        cphi, sphi = cs.cos(phi), cs.sin(phi)
        ctheta, stheta = cs.cos(theta), cs.sin(theta)
        cpsi, spsi = cs.cos(psi), cs.sin(psi)
        acc = cs.vertcat(
            (cpsi * stheta * cphi + spsi * sphi) * az,
            (spsi * stheta * cphi - cpsi * sphi) * az,
            (ctheta * cphi) * az - 9.81,
        )
        x_dot = cs.vertcat(vel, acc)
        ode = {"x": x, "p": u, "ode": x_dot}
        self.dynamics = cs.Function(
            "dynamics", [x, u], [x_dot], ["x", "u"], ["xf"], {"cse": True}
        )
        self.integrator = cs.integrator("intg", "cvodes", ode, 0.0, self.sampling_time)

        # build the symbolic safety constraint for the cylindrical obstacle
        pos_obs = cs.MX.sym("p_o", (3, self.n_obstacles))
        dir_obs = cs.MX.sym("d_o", (3, self.n_obstacles))  # unit vectors
        r2 = self.radius_obstacles**2
        h = cs.sum1(cs.cross(pos - pos_obs, dir_obs) ** 2).T - r2  # >= 0
        self.safety_constraints = cs.Function(
            "h", [x, pos_obs, dir_obs], [h], ["x", "p_o", "d_o"], ["h"]
        )

    @property
    def previous_action(self) -> ActType:
        return self._u_prev

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if options is not None and "ic" in options:
            x = np.asarray(options["ic"])
            assert self.observation_space.contains(x), f"invalid initial state {x}"
        else:
            x = self.x0

        for _ in range(1000):
            self.pos_obs = self.np_random.normal(self.pos_obs_mean, self.pos_obs_std)
            self.dir_obs = self.np_random.normal(self.dir_obs_mean, self.dir_obs_std)
            self.dir_obs /= np.linalg.norm(self.dir_obs, axis=0)
            if np.all(
                self.safety_constraints(x, self.pos_obs, self.dir_obs) >= 0
            ) and np.all(
                self.safety_constraints(self.xf, self.pos_obs, self.dir_obs) >= 0
            ):
                break
        else:
            raise RuntimeError("could not generate valid obstacles")

        self._x = x
        self._t = 0
        self._u_prev = self.a0
        return x, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        u = np.asarray(action).reshape(self.na)
        assert self.action_space.contains(u), f"invalid action {u}"

        w = self.sample_action_disturbance_profiles(1, 1)[0, 0]
        x = self._x
        x_new = self.integrator(x0=x, p=u + w)["xf"].full().flatten()
        assert self.observation_space.contains(x_new), f"invalid new state {x_new}"

        self._x = x_new
        self._t += 1
        self._u_prev = u
        truncated = self._t >= self._max_timesteps
        return x_new, self._compute_cost(x, u, x_new), False, truncated, {}

    def _compute_cost(self, x: ObsType, u: ActType, x_new: ObsType) -> float:
        # NOTE: for now, we penalize the violations of the least stringent CBF, i.e.,
        # when h(x_new) >= 0
        h = self.safety_constraints(x_new, self.pos_obs, self.dir_obs)
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
    "pos_obs_mean": np.asarray(
        [
            2.1501183931938646,
            3.149195729198508,
            0.01915161847424313,
            8.070382473003617,
            0.07263722684384825,
            7.910910323305432,
            11.86892471734808,
            11.975422194254207,
            0.06951498743674385,
        ]
    ),
    "pos_obs_std": np.asarray(
        [
            0.9882564008971457,
            1.0240661424691657,
            1.0029827469791857,
            1.0409915674956833,
            1.0364195836037582,
            0.9836045719212586,
            0.9964865665158052,
            0.977362603387804,
            0.9814155602746699,
        ]
    ),
    "dir_obs_mean": np.asarray(
        [
            5.830065365259106e-5,
            -0.006040709069421122,
            0.9895049792346912,
            -0.0014712701304950841,
            0.9896505187192365,
            -0.0057590560373283915,
            -0.010106393569256784,
            -0.0034225069279508052,
            0.989990141410835,
        ]
    ),
    "dir_obs_std": np.asarray(
        [
            0.10226346768120201,
            0.10180337023361205,
            0.009558758700248624,
            0.10257391678649148,
            0.011094264318978846,
            0.09990384993548733,
            0.09555828673217691,
            0.10316701077724287,
            0.009853353658495698,
        ]
    ),
}
NORMALIZATION = tuple(
    np.concatenate(
        [
            NORM_FACTORS[name + suffix]
            for name in ("state", "action", "pos_obs", "dir_obs")
        ]
    )
    for suffix in ("_mean", "_std")
)
