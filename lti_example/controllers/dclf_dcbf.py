from collections.abc import Callable
from typing import Any

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.options import OPTS
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dcbf, dlqr
from mpcrl.util.math import clip

env = Env()
ns, na = env.ns, env.na
x = cs.MX.sym("x", ns)
u = cs.MX.sym("u", na)
delta = cs.MX.sym("delta")
primals = cs.vertcat(u, delta)
x_next = env.dynamics(x, u)
alpha_dclf = 0.5
alpha_dcbf = 0.5
penalty_u = 0.5
penalty_delta = 0.1

_, P = dlqr(env.A, env.B, env.Q, env.R)
lyapunov = lambda x_: cs.bilin(P, x_)
V = lyapunov(x)
V_next = lyapunov(x_next)
dclf_cnstr = V_next - (1 - alpha_dclf) * V - delta  # <= 0
h = env.safety_constraints
dcbf_cnstr = dcbf(h, x, u, env.dynamics, [lambda y: alpha_dcbf * y])  # >= 0

# it is a QCQP problem due to the CLF constraint, so we have to solve it nonlinearly
# as casadi does not support SOCP yet
qcqp = {
    "x": primals,
    "p": x,
    "f": penalty_u * cs.sumsqr(u) + penalty_delta * delta,
    "g": cs.vertcat(-dclf_cnstr, dcbf_cnstr),
}
solver = cs.nlpsol("solver_dclf_dcbf_qcqp", "fatrop", qcqp, OPTS["fatrop"])
lbx = np.append(np.full(env.na, -np.inf), 0)  # don't bound the action, but clip it
ubx = np.full(env.na + 1, np.inf)
res = solver(p=x, lbx=lbx, ubx=ubx, lbg=0, ubg=np.inf)
u_dclf_dcbf = clip(res["x"][: env.na], -env.a_bound, env.a_bound)
controller = cs.Function("dclf_dcbf_qcqp", [x], [u_dclf_dcbf], ["x"], ["u_opt_clip"])

for key, var in locals().copy().items():
    if key not in ("na", "controller"):
        del var


def get_controller(
    *_: Any, **__: Any
) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Returns the discrete-time Control Lyapunov Function and Control Barrier Function
    controller with action saturation.

    Returns
    -------
    callable from array-like to array-like
        A controller that maps the current state to the desired action.
    """

    def _f(x):
        return controller(x).full().reshape(na)

    return controller
