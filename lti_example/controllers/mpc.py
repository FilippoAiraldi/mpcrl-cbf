from collections.abc import Callable
from typing import Any

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.options import OPTS
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dlqr


def create_mpc(
    horizon: int, soft: bool, bound_initial_state: bool, dlqr_terminal_cost: bool
) -> Mpc[cs.MX]:
    """Creates a linear MPC controller for the `ConstrainedLtiEnv` env with the given
    horizon.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.
    soft : bool
        Whether to impose soft constraints on the states or not. If not, note that the
        optimization problem may become infeasible.
    bound_initial_state : bool
        If `False`, the initial state is excluded from the state constraints. Otherwise,
        the initial state is also constrained (useful for RL for predicting the value
        functions of the current state).
    dlqr_terminal_cost : bool
        If `True`, the quadratic DLQR terminal cost (i.e., the solution to the
        corresponding Riccati equation) is added to the terminal cost.

    Returns
    -------
    Mpc
        The single-shooting MPC controller.
    """
    A = Env.A
    B = Env.B
    Q = Env.Q
    R = Env.R
    ns, na = B.shape
    a_bnd = Env.a_bound
    x_bnd = Env.x_soft_bound

    # create actions and get states when rolling the dynamics along the horizon
    nlp = Nlp("MX")
    mpc = Mpc(nlp, prediction_horizon=horizon, shooting="single")
    mpc.state("x", ns)
    u, _ = mpc.action("u", na, lb=-a_bnd, ub=a_bnd)
    mpc.set_linear_dynamics(A, B)
    x = mpc.states["x"]

    # set state constraints (the same for stage and terminal) - the initial constraint
    # is needed to penalize the current state in RL, but can be removed in other cases
    if bound_initial_state:
        x_ = x
    else:
        x_ = x[:, 1:]
    if soft:
        _, _, slack = mpc.constraint("lbx", x_, ">=", -x_bnd, soft=True)
        mpc.constraint("ubx", x_, "<=", x_bnd + slack)
    else:
        mpc.constraint("lbx", x_, ">=", -x_bnd)
        mpc.constraint("ubx", x_, "<=", x_bnd)

    # compute stage cost
    J = sum(cs.bilin(Q, x[:, i]) + cs.bilin(R, u[:, i]) for i in range(horizon))

    # compute terminal cost
    if dlqr_terminal_cost:
        _, P = dlqr(A, B, Q, R)
        J += cs.bilin(P, x[:, -1])

    # add penalty cost (if needed) and set the solver
    if soft:
        J += Env.constraint_penalty * cs.sum1(cs.sum2(slack))
    nlp.minimize(J)
    nlp.init_solver(OPTS["qpoases"], "qpoases")
    return mpc


def get_mpc_controller(
    *args: Any, **kwargs: Any
) -> Callable[[npt.NDArray[np.floating]], tuple[npt.NDArray[np.floating], float]]:
    """Returns the MPC controller with the given horizon as a callable function.

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
    # create the MPC and convert it to a function (it's faster than going through csnlp)
    mpc = create_mpc(*args, **kwargs)
    x0 = mpc.initial_states["x_0"]
    u_opt = mpc.actions["u"][:, 0]
    primals = mpc.nlp.x
    func = mpc.nlp.to_function("mpc", (x0, primals), (u_opt, primals))
    inner_solver = mpc.nlp.solver
    last_sol = 0.0

    def _f(x):
        nonlocal last_sol
        u_opt, last_sol = func(x, last_sol)
        return u_opt.toarray().reshape(-1), inner_solver.stats()["t_proc_total"]

    return _f
