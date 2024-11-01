from collections.abc import Callable
from typing import Any

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.options import OPTS
from csnlp import Nlp
from csnlp.wrappers import ScenarioBasedMpc
from env import ConstrainedLtiEnv
from mpcrl.util.control import dlqr

from util.output_supress import nostdout


def create_scmpc(
    horizon: int,
    scenarios: int,
    dcbf: bool,
    soft: bool,
    bound_initial_state: bool,
    dlqr_terminal_cost: bool,
    env: ConstrainedLtiEnv | None = None,
) -> ScenarioBasedMpc[cs.MX]:
    """Creates a linear Scenario-based MPC controller for the `ConstrainedLtiEnv` env.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.
    scenarios : int
        The number of scenarios to consider in the SCMPC controller.
    dcbf : bool
        Whether to use discrete-time control barrier functions to enforce safety
        constraints or not.
    soft : bool
        Whether to impose soft constraints on the states or not. If not, note that the
        optimization problem may become infeasible.
    bound_initial_state : bool
        If `False`, the initial state is excluded from the state constraints. Otherwise,
        the initial state is also constrained (useful for RL for predicting the value
        functions of the current state). Disregarded if `dcbf` is `True`.
    dlqr_terminal_cost : bool
        If `True`, the quadratic DLQR terminal cost (i.e., the solution to the
        corresponding Riccati equation) is added to the terminal cost.
    env : ConstrainedLtiEnv, optional
        The environment to build the SCMPC for. If `None`, a new default environment is
        instantiated.

    Returns
    -------
    Mpc
        The single-shooting SCMPC controller.
    """
    if env is None:
        env = ConstrainedLtiEnv(0)
    A = env.A
    B = env.B
    D = env.D
    Q = env.Q
    R = env.R
    ns, na, nd = env.ns, env.na, env.nd
    a_bnd = env.a_bound
    x_bnd = env.x_soft_bound

    # create states, actions and disturbances, and set the dynamics
    scmpc = ScenarioBasedMpc(Nlp("MX"), scenarios, horizon, shooting="single")
    x, _, _ = scmpc.state("x", ns)
    u, _ = scmpc.action("u", na, lb=-a_bnd, ub=a_bnd)
    scmpc.disturbance("w", nd)
    scmpc.set_linear_dynamics(A, B, D)

    # set state constraints (the same for stage and terminal) - the initial constraint
    # is needed to penalize the current state in RL, but can be removed in other cases
    if dcbf:
        h = env.safety_constraints(x)
        alpha = 0.99
        # dcbf = h[:, 1:] - (1 - alpha) * h[:, :-1]  # vanilla CBF constraints
        decay = cs.repmat(cs.power(1 - alpha, range(1, horizon + 1)).T, h.shape[0], 1)
        dcbf = h[:, 1:] - decay * h[:, 0]  # unrolled CBF constraints
        dcbf_lbx, dcbf_ubx = cs.vertsplit_n(dcbf, 2)
        if soft:
            _, _, slack = scmpc.constraint("lbx", dcbf_lbx, ">=", 0.0, soft=True)
            scmpc.constraint("ubx", dcbf_ubx + slack, ">=", 0.0)
        else:
            scmpc.constraint("lbx", dcbf_lbx, ">=", 0.0)
            scmpc.constraint("ubx", dcbf_ubx, ">=", 0.0)
    else:
        # impose the safety constraints normally (not via CBFs), taking care of the
        # initial state and slack if needed
        x_ = x if bound_initial_state else x[:, 1:]
        if soft:
            _, _, slack, _ = scmpc.constraint_from_single(
                "lbx", x_, ">=", -x_bnd, soft=True
            )
            scmpc.constraint_from_single("ubx", x_, "<=", x_bnd + slack)
        else:
            scmpc.constraint_from_single("lbx", x_, ">=", -x_bnd)
            scmpc.constraint_from_single("ubx", x_, "<=", x_bnd)

    # compute stage cost
    J = sum(cs.bilin(Q, x[:, i]) + cs.bilin(R, u[:, i]) for i in range(horizon))

    # compute terminal cost
    if dlqr_terminal_cost:
        _, P = dlqr(A, B, Q, R)
        J += cs.bilin(P, x[:, -1])

    # add penalty cost (if needed) and set the solver
    if soft:
        J += env.constraint_penalty * cs.sum1(cs.sum2(slack))
    scmpc.minimize_from_single(J)
    with nostdout():
        scmpc.init_solver(OPTS["qpoases"], "qpoases")
    return scmpc


def get_scmpc_controller(
    *args: Any, **kwargs: Any
) -> Callable[[npt.NDArray[np.floating]], tuple[npt.NDArray[np.floating], float]]:
    """Returns the Scenario-based MPC controller as a callable function.

    Parameters
    ----------
    args, kwargs
        The arguments to pass to the `create_mpc` method.

    Returns
    -------
    callable from array-like to (array-like, float)
        A controller that maps the current state to the desired action, and returns also
        the time it took to compute the action.
    """
    # create the SCMPC and convert it to a function (faster than going through csnlp)
    scmpc = create_scmpc(*args, **kwargs)
    x0 = scmpc.initial_states["x_0"]
    u_opt = scmpc.actions["u"][:, 0]
    primals = scmpc.nlp.x
    disturbances = cs.vvcat(scmpc.disturbances.values())  # cannot do otherwise
    func = scmpc.nlp.to_function("scmpc", (x0, disturbances, primals), (u_opt, primals))
    inner_solver = scmpc.nlp.solver
    last_sol = 0.0
    horizon = scmpc.prediction_horizon
    n_scenarios = scmpc.n_scenarios

    def _f(x, env: ConstrainedLtiEnv):
        nonlocal last_sol
        disturbances = env.sample_disturbance_profiles(n_scenarios)[:, :horizon]
        with nostdout():
            u_opt, last_sol = func(x, disturbances.reshape(-1), last_sol)
        return u_opt.toarray().reshape(-1), inner_solver.stats()["t_proc_total"]

    return _f
