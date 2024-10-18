import argparse

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from env import ConstrainedLtiEnv as Env
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpcrl.util.control import dcbf, dlqr
from mpcrl.util.math import clip

SOLVER = "fatrop"
OPTS = {
    "error_on_fail": True,
    "expand": True,
    "print_time": False,
    "bound_consistency": True,
    "calc_lam_p": False,
    "calc_lam_x": False,
    SOLVER: {"max_iter": 500, "print_level": 0},
}


def create_dclf_dcbf_controller() -> cs.Function:
    """Creates a function that solves the DCLF-DCBF-based controller for the constrained
    LTI environment. The output action is clipped in the action space bounds."""
    env = Env()
    x = cs.MX.sym("x", env.ns)
    u = cs.MX.sym("u", env.na)
    delta = cs.MX.sym("delta")
    primals = cs.vertcat(u, delta)
    x_next = env.dynamics(x, u)
    alpha_dclf = 0.5
    alpha_dcbf = 0.5
    penalty_u = 0.5
    penalty_delta = 0.1

    _, P = dlqr(Env.A, Env.B, Env.Q, Env.R)
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
    solver = cs.nlpsol("solver_dclf_dcbf_qcqp", SOLVER, qcqp, OPTS)
    lbx = np.append(np.full(env.na, -np.inf), 0)  # don't bound the action, but clip it
    ubx = np.full(env.na + 1, np.inf)
    res = solver(p=x, lbx=lbx, ubx=ubx, lbg=0, ubg=np.inf)
    u_dclf_dcbf_qp = clip(res["x"][: env.na], *env.a_bounds)
    return cs.Function("dclf_dcbf_qcqp", [x], [u_dclf_dcbf_qp], ["x"], ["u_opt_clip"])


def simulate_once(
    controller: cs.Function, timesteps: int, seed: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Simulates one episode of the navigation environment using the DCLF-DCBF
    controller with saturated actions, and returns the state trajectory and actions."""
    env = Env()
    x, _ = env.reset(seed=seed, options={"contour": True})
    X = np.empty((timesteps + 1, env.ns))
    X[0] = x
    U = np.empty((timesteps, env.na))
    for i in range(timesteps):
        u = np.reshape(controller(x), env.na)
        x, _, _, _, _ = env.step(u)
        X[i + 1] = x
        U[i] = u
    return X, U


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Simulations for the DCLF-DCBF-QCQP controller applied to the "
        "constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-sim", type=int, default=100, help="Number of simulations.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of timesteps per each simulation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    args = parser.parse_args()

    # get the controller
    ctrl = create_dclf_dcbf_controller()

    # run the simulations in parallel asynchronously
    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_sim)
    data = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(simulate_once)(ctrl, args.timesteps, int(seed)) for seed in seeds
    )
    state_data, action_data = map(np.asarray, zip(*data))

    # finally, plot the results
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, fig)
    lw = 1.0

    ax1 = fig.add_subplot(gs[:, 0])
    bounds = Env.x_soft_bounds
    ax1.add_patch(Rectangle(bounds[0], *(bounds[1] - bounds[0]), fill=False))
    ax1.plot(*state_data[:, 0].T, "C0", ls="none", marker=".", markersize=3)
    for states in state_data:
        ax1.plot(*states.T, "C0", lw=lw)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")

    timesteps = np.arange(args.timesteps)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    for actions in action_data:
        ax2.step(timesteps, actions[:, 0], "C0", lw=lw, where="post")
        ax3.step(timesteps, actions[:, 1], "C0", lw=lw, where="post")
    ax2.set_ylabel("$u_1$")
    ax3.set_ylabel("$u_2$")
    ax3.set_xlabel("$k$")
    plt.show()
