import casadi as cs
from controllers.options import OPTS
from csnlp import Nlp
from csnlp.wrappers import Mpc
from env import ConstrainedLtiEnv as Env


def create_mpc_controller(horizon: int) -> Mpc[cs.MX]:
    """Creates a linear MPC controller for the `ConstrainedLtiEnv` env with the given
    horizon.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.

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
    nlp = Nlp("MX")
    mpc = Mpc(nlp, prediction_horizon=horizon, shooting="single")
    mpc.state("x", ns)
    u, _ = mpc.action("u", na, lb=-a_bnd, ub=a_bnd)
    mpc.set_linear_dynamics(A, B)
    x = mpc.states["x"]
    mpc.constraint("lbx", x, ">=", -x_bnd)
    mpc.constraint("ubx", x, "<=", x_bnd)
    J = sum(cs.bilin(Q, x[:, i]) + cs.bilin(R, u[:, i]) for i in range(horizon))
    nlp.minimize(J)
    nlp.init_solver(OPTS["qpoases"], "qpoases")
    return mpc
