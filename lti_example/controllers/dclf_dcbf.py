from collections.abc import Callable
from typing import Any

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.config import SOLVER_OPTS
from csnlp import Nlp
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dcbf, dlqr


def create_dclf_dcbf_qcqp(env: Env | None = None, *_: Any, **__: Any) -> Nlp[cs.MX]:
    """Creates a DCLF-DCBF-QCQP controller for the `ConstrainedLtiEnv` env with the
    given horizon.

    Parameters
    ----------
    env : ConstrainedLtiEnv, optional
        The environment to build the controller for. If `None`, a new default
        environment is instantiated.

    Returns
    -------
    Nlp
        The corresponding optimization problem.
    """
    if env is None:
        env = Env(0)
    nlp = Nlp("MX")
    ns, na = env.ns, env.na
    x = nlp.parameter("x", (ns, 1))
    u, _, _ = nlp.variable("u", (na, 1), lb=-env.a_bound, ub=env.a_bound)
    delta, _, _ = nlp.variable("delta", lb=0)
    x_next = env.dynamics(x, u)
    alpha_dclf = 0.5
    alpha_dcbf = 0.5
    penalty_delta = 10.0

    _, P = dlqr(env.A, env.B, env.Q, env.R)
    lyapunov = lambda x_: cs.bilin(P, x_)
    V = lyapunov(x)
    V_next = lyapunov(x_next)
    nlp.constraint("dclf", V_next - (1 - alpha_dclf) * V, "<=", delta)
    h = env.safety_constraints
    dcbf_cnstr = dcbf(h, x, u, env.dynamics, [lambda y: alpha_dcbf * y])  # >= 0
    nlp.constraint("dcbf", dcbf_cnstr, ">=", 0)
    nlp.minimize(cs.bilin(env.R, u) + penalty_delta * delta)

    # it is a QCQP problem due to the CLF constraint, so we have to solve it nonlinearly
    # as casadi does not support SOCP yet
    nlp.init_solver(SOLVER_OPTS["fatrop"], "fatrop", "nlp")
    return nlp


def get_dclf_dcbf_controller(
    *args: Any, **kwargs: Any
) -> Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]]:
    """Returns the discrete-time LQR controller with action saturation.

    Parameters
    ----------
    args, kwargs
        The arguments to pass to the `create_dclf_dcbf_qcqp` method.

    Returns
    -------
    callable from (array-like, ConstrainedLtiEnv) to (array-like, float)
        A controller that maps the current state to the desired action, and returns also
        the time it took to compute the action.
    """
    # create the QCQP and convert it to a function
    nlp = create_dclf_dcbf_qcqp(*args, **kwargs)
    x0 = nlp.parameters["x"]
    u_dclf_dcbf = nlp.variables["u"]
    primals = nlp.x
    func = nlp.to_function("dclf_dcbf_qcqp", (x0, primals), (u_dclf_dcbf, primals))
    last_sol = 0.0

    def _f(x, _):
        nonlocal last_sol
        u, last_sol = func(x, last_sol)
        return u.toarray().reshape(-1), func.stats()["t_proc_total"]

    return _f
