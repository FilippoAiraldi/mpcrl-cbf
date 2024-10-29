from collections.abc import Callable

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.options import OPTS
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env import ConstrainedLtiEnv as Env


def create_mpc(horizon: int, soft: bool) -> Mpc[cs.MX]:
    """Creates a linear MPC controller for the `ConstrainedLtiEnv` env with the given
    horizon.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.
    soft : bool
        Whether to impose soft constraints on the states or not. If not, note that the
        optimization problem may become infeasible.

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

    # set state constraints
    if soft:
        _, _, slack = mpc.constraint("lbx", x, ">=", -x_bnd, soft=True)
        mpc.constraint("ubx", x, "<=", x_bnd + slack)
    else:
        mpc.constraint("lbx", x, ">=", -x_bnd)
        mpc.constraint("ubx", x, "<=", x_bnd)

    # set cost and solver
    J = sum(cs.bilin(Q, x[:, i]) + cs.bilin(R, u[:, i]) for i in range(horizon))
    if soft:
        J += Env.constraint_penalty * cs.sum1(cs.sum2(slack))
    nlp.minimize(J)
    nlp.init_solver(OPTS["qpoases"], "qpoases")
    return mpc


def get_mpc_controller(
    horizon: int, soft: bool
) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Returns the MPC controller with the given horizon as a callable function.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.
    soft : bool
        Whether to impose soft constraints on the states or not. If not, note that the
        optimization problem may become infeasible.

    Returns
    -------
    callable from array-like to array-like
        A controller that maps the current state to the desired action.
    """
    # create the MPC and convert it to a function (it's faster than going through csnlp)
    mpc = create_mpc(horizon, soft)
    x0 = mpc.initial_states["x_0"]
    u_opt = mpc.actions["u"][:, 0]
    primals = mpc.nlp.x
    func = mpc.nlp.to_function("mpc", (x0, primals), (u_opt, primals))
    last_sol = 0.0

    def _f(x):
        nonlocal last_sol
        u_opt, last_sol = func(x, last_sol)
        return u_opt.toarray().reshape(-1)

    return _f
