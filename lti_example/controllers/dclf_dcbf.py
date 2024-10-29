from collections.abc import Callable

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.options import OPTS
from csnlp import Nlp
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dcbf, dlqr
from mpcrl.util.math import clip


def create_dclf_dcbf_qcqp() -> Nlp[cs.MX]:
    """Creates a DCLF-DCBF-QCQP controller for the `ConstrainedLtiEnv` env with the
    given horizon.

    Returns
    -------
    Nlp
        The corresponding optimization problem.
    """
    nlp = Nlp("MX")
    env = Env()
    ns, na = Env.ns, Env.na
    x = nlp.parameter("x", (ns, 1))
    u, _, _ = nlp.variable("u", (na, 1))
    delta, _, _ = nlp.variable("delta", lb=0)
    x_next = env.dynamics(x, u)
    alpha_dclf = 0.5
    alpha_dcbf = 0.5
    penalty_u = 0.5
    penalty_delta = 0.1

    _, P = dlqr(Env.A, Env.B, Env.Q, Env.R)
    lyapunov = lambda x_: cs.bilin(P, x_)
    V = lyapunov(x)
    V_next = lyapunov(x_next)
    nlp.constraint("dclf", V_next - (1 - alpha_dclf) * V, "<=", delta)
    h = env.safety_constraints
    dcbf_cnstr = dcbf(h, x, u, env.dynamics, [lambda y: alpha_dcbf * y])  # >= 0
    nlp.constraint("dcbf", dcbf_cnstr, ">=", 0)
    nlp.minimize(penalty_u * cs.sumsqr(u) + penalty_delta * delta)

    # it is a QCQP problem due to the CLF constraint, so we have to solve it nonlinearly
    # as casadi does not support SOCP yet
    nlp.init_solver(OPTS["fatrop"], "fatrop", "nlp")
    return nlp


def get_dclf_dcbf_controller() -> (
    Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
):
    """Returns the discrete-time LQR controller with action saturation.

    Returns
    -------
    callable from array-like to array-like
        A controller that maps the current state to the desired action.
    """
    # create the QCQP and convert it to a function
    nlp = create_dclf_dcbf_qcqp()
    x0 = nlp.parameters["x"]
    u_dclf_dcbf = clip(nlp.variables["u"], -Env.a_bound, Env.a_bound)
    primals = nlp.x
    func = nlp.to_function("dclf_dcbf_qcqp", (x0, primals), (u_dclf_dcbf, primals))
    last_sol = 0.0

    def _f(x):
        nonlocal last_sol
        u, last_sol = func(x, last_sol)
        return u.toarray().reshape(-1)

    return _f
