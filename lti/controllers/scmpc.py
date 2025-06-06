from collections.abc import Callable
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import ScenarioBasedMpc
from csnn.convex import PwqNN
from env import ConstrainedLtiEnv as Env
from mpcrl.util.seeding import RngType
from scipy.linalg import solve_discrete_are as dlqr

from util.defaults import DCBF_GAMMA, PWQNN_HIDDEN, SOLVER_OPTS, TIME_MEAS
from util.nn import nn2function
from util.wrappers import nostdout


def create_scmpc(
    horizon: int,
    scenarios: int,
    dcbf: bool,
    soft: bool,
    bound_initial_state: bool,
    terminal_cost: set[Literal["dlqr", "pwqnn"]],
    pwq_hidden: int = PWQNN_HIDDEN,
    env: Env | None = None,
    *_: Any,
    **__: Any,
) -> tuple[ScenarioBasedMpc[cs.MX], PwqNN | None]:
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
    terminal_cost : set of {"dlqr", "pwqnn"}
        The type of terminal cost to use. If "dlqr", the terminal cost is the solution
        to the discrete-time LQR problem. If "pwqnn", a piecewise quadratic neural
        network is used to approximate the terminal cost. Can also be a set of multiple
        terminal costs to use, at which point these are summed together; can also be an
        empty set, in which case no terminal cost is used.
    pwq_hidden : int, optional
        The number of hidden units in the piecewise quadratic neural network, if used.
    env : ConstrainedLtiEnv, optional
        The environment to build the SCMPC for. If `None`, a new default environment is
        instantiated.

    Returns
    -------
    Mpc and PwqNN (optional)
        The single-shooting MPC controller, as well as the piecewise quadratic
        terminal cost neural net if `"pwqnn"` is in `terminal_cost`; otherwise, `None`.
    """
    if env is None:
        env = Env(0)
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
    x, _, x0 = scmpc.state("x", ns)
    u, _ = scmpc.action("u", na, lb=-a_bnd, ub=a_bnd)
    scmpc.disturbance("w", nd)
    scmpc.set_affine_dynamics(A, B, D)

    # set state constraints (the same for stage and terminal) - the initial constraint
    # is needed to penalize the current state in RL, but can be removed in other cases
    h0 = env.safety_constraints(x0)
    nc = h0.size1()
    if dcbf:
        h = env.safety_constraints(x[:, 1:])
        gamma = scmpc.parameter("gamma", (nc,))
        decays = cs.hcat(
            [cs.power(1 - gamma[i, 0], range(1, horizon + 1)) for i in range(nc)]
        )
        dcbf = h - decays.T * h0  # unrolled CBF constraints
        dcbf_lbx, dcbf_ubx = cs.vertsplit_n(dcbf, 2)
        if soft:
            _, _, slack, _ = scmpc.constraint_from_single(
                "lbx", dcbf_lbx, ">=", 0.0, soft=True
            )
            scmpc.constraint_from_single("ubx", dcbf_ubx + slack, ">=", 0.0)
        else:
            scmpc.constraint_from_single("lbx", dcbf_lbx, ">=", 0.0)
            scmpc.constraint_from_single("ubx", dcbf_ubx, ">=", 0.0)
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
    J = sum(
        cs.sum1(Q * x[:, i] ** 2) + cs.sum1(R * u[:, i] ** 2) for i in range(horizon)
    )

    # compute terminal cost
    xT = x[:, -1]
    pwqnn = None
    if "dlqr" in terminal_cost:
        P = dlqr(A, B, np.diag(Q), np.diag(R))
        J += cs.bilin(P, xT)
    if "pwqnn" in terminal_cost:
        pwqnn = PwqNN(ns, pwq_hidden)
        nnfunc = nn2function(pwqnn, "pwqnn")
        nnfunc = nnfunc.factory("F", nnfunc.name_in(), ("V", "grad:V:x", "hess:V:x:x"))
        weights = {
            n: scmpc.parameter(n, p.shape)
            for n, p in pwqnn.parameters(prefix="pwqnn", skip_none=True)
        }
        output = nnfunc(x=x0, **weights)
        dx = xT - x0
        val, jac, hess = output["V"], output["grad_V_x"], output["hess_V_x_x"]
        J += val + cs.dot(jac, dx) + 0.5 * cs.bilin(hess, dx)

    # add penalty cost (if needed) and set the solver
    if soft:
        J += env.constraint_penalty * cs.sum1(cs.sum2(slack))
    scmpc.minimize_from_single(J)
    with nostdout():
        scmpc.init_solver(SOLVER_OPTS["gurobi"], "gurobi", type="conic")
    return scmpc, pwqnn


def get_scmpc_controller(
    *args: Any,
    seed: RngType = None,
    weights: dict[str, npt.NDArray[np.floating]] | None = None,
    **kwargs: Any,
) -> tuple[
    Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]],
    dict[str, npt.NDArray[np.floating]],
]:
    """Returns the Scenario-based MPC controller as a callable function.

    Parameters
    ----------
    args, kwargs
        The arguments to pass to the `create_scmpc` method.
    seed : RngType, optional
        The seed used during creating of the PwqNN weights, if necessary.
    weights : dict of (str, array), optional
        The neural network weights to use in the controller. If `None`, the weights are
        initialized randomly.

    Returns
    -------
    callable from (array-like, ConstrainedLtiEnv) to (array-like, float)
        A controller that maps the current state + env to the desired action, and
        returns also the time it took to compute the action.
    dict of str to arrays, optional
        The numerical weights of the parametric MPC controller, if any.
    """
    # create the SCMPC
    scmpc, pwqnn = create_scmpc(*args, **kwargs)

    # group its NN parameters (if any) into a vector and assign numerical values to them
    sym_weights_, num_weights_ = {}, {}
    if weights is None:
        weights = {}
    if pwqnn is not None:
        nn_weights_ = dict(pwqnn.init_parameters(prefix="pwqnn", seed=seed))
        for k, v in nn_weights_.items():
            sym_weights_[k] = scmpc.parameters[k]
            num_weights_[k] = weights.get(k, v)
    if "gamma" in scmpc.parameters:
        sym_weights_["gamma"] = scmpc.parameters["gamma"]
        num_weights_["gamma"] = weights.get(
            "gamma", np.full(sym_weights_["gamma"].shape, DCBF_GAMMA)
        )
    sym_weights = cs.vvcat(sym_weights_.values())
    num_weights = cs.vvcat(num_weights_.values())
    disturbances = cs.vvcat(scmpc.disturbances.values())  # cannot do otherwise

    # convert the SCMPC object to a function (faster than going through csnlp)
    x0 = scmpc.initial_states["x_0"]
    u_opt = scmpc.actions["u"][:, 0]
    primals = scmpc.nlp.x
    func = scmpc.nlp.to_function(
        "scmpc", (x0, sym_weights, disturbances, primals), (u_opt, primals)
    )
    inner_solver = scmpc.nlp.solver
    last_sol = 0.0
    horizon = scmpc.prediction_horizon
    n_scenarios = scmpc.n_scenarios

    def _f(x, env: Env):
        nonlocal last_sol
        disturbances = env.sample_disturbance_profiles(n_scenarios, horizon)
        with nostdout():
            u_opt, last_sol = func(x, num_weights, disturbances.reshape(-1), last_sol)
        return u_opt.toarray().reshape(-1), inner_solver.stats()[TIME_MEAS]

    return _f, num_weights_
