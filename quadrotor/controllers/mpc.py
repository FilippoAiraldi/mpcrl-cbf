from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from csnn import ReLU, Sigmoid, init_parameters
from csnn.convex import PsdNN, PwqNN
from csnn.feedforward import Mlp
from env import QuadrotorEnv as Env
from mpcrl.util.control import rk4
from mpcrl.util.seeding import RngType
from scipy.linalg import solve_discrete_are as dlqr

from util.defaults import (
    DCBF_GAMMA,
    KAPPANN_HIDDEN,
    PSDNN_HIDDEN,
    PWQNN_HIDDEN,
    SOLVER_OPTS,
    TIME_MEAS,
)
from util.nn import nn2function


def get_discrete_time_dynamics(
    env: Env, include_disturbance: bool = False
) -> cs.Function:
    """Constructs a discrete-time dynamics function for the quadrotor env.

    Parameters
    ----------
    env : QuadrotorEnv
        Quadrotor environment.
    include_disturbance : bool, optional
        Whether to include the disturbance in the dynamics or not.

    Returns
    -------
    cs.Function
        The discrete-time dynamics function.
    """
    x, u = env.dynamics.mx_in()
    args = [x, u]
    argnames = ["x", "u"]
    if include_disturbance:
        d = cs.MX.sym("d", env.nd)
        u += d
        args.append(d)
        argnames.append("d")
    xf = cs.simplify(rk4(lambda x_: env.dynamics(x_, u), x, env.sampling_time))
    return cs.Function("dynamics", args, [xf], argnames, ["xf"], {"cse": True})
    # CasADi native RK4 integrator cannot be expanded to SX unfortunately
    # ode = {"x": x, "p": u, "ode":  env.dynamics(x, u)}
    # integrator = cs.integrator(
    #     "dyn_intg", "rk", ode, 0.0, sampling_time, {"number_of_finite_elements": 1}
    # )


def create_mpc(
    horizon: int,
    dcbf: bool,
    use_kappann: bool,
    soft: bool,
    bound_initial_state: bool,
    terminal_cost: set[Literal["dlqr", "pwqnn", "psdnn"]],
    kappann_hidden_size: Sequence[int] = KAPPANN_HIDDEN,
    pwqnn_hidden_size: int = PWQNN_HIDDEN,
    psdnn_hidden_sizes: Sequence[int] = PSDNN_HIDDEN,
    env: Env | None = None,
    *_: Any,
    **__: Any,
) -> tuple[Mpc[cs.MX], Mlp | None, PwqNN | None, PsdNN | None]:
    """Creates a nonlinear MPC controller for the `QuadrotorEnv` environment.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.
    dcbf : bool
        Whether to use discrete-time control barrier functions to enforce safety
        constraints or not.
    use_kappann : bool
        Whether to use a neural network to compute the class Kappa function for the CBF.
        If `False`, a constant is used instead. Only used when `dcbf=True`.
    soft : bool
        Whether to impose soft constraints on the states or not. If not, note that the
        optimization problem may become infeasible.
    bound_initial_state : bool
        If `False`, the initial state is excluded from the state constraints. Otherwise,
        the initial state is also constrained (useful for RL for predicting the value
        functions of the current state). Disregarded if `dcbf` is `True`.
    terminal_cost : set of {"dlqr", "pwqnn", "psdnn"}
        The type of terminal cost to use. If "dlqr", the terminal cost is the solution
        to the discrete-time LQR problem. If "pwqnn", a piecewise quadratic neural
        network is used to approximate the terminal cost. If "psdnn", a positive
        semidefinite neural network is used to approximate the terminal cost. Can also
        be a set of multiple terminal costs to use, at which point these are summed
        together; can also be an empty set, in which case no terminal cost is used.
    kappann_hidden_size : sequence of int, optional
        The number of hidden units in the multilayer perceptron used to learn the
        DCBF Kappa function, if `dcbf` is `True`.
    pwqnn_hidden_size : int, optional
        The number of hidden units in the piecewise quadratic neural network, if used.
    psdnn_hidden_sizes : sequence of int, optional
        The number of hidden units in the positive semidefinite neural network, if used.
    env : Env | None, optional
        The environment to build the MPC for. If `None`, a new default environment is
        instantiated.

    Returns
    -------
    Mpc, Mlp (optional), PwqNN (optional), and PsdNN (optional)
        The multiple-shooting nonlinear MPC controller; the multilayer perceptron if
        `dcbf` is `True`, otherwise `None`; the piecewise quadratic terminal cost neural
        net if `"pwqnn"` is in `terminal_cost`, otherwise `None`.; the positive
        semidefinite terminal cost neural net if `"psdnn"` is in `terminal_cost`,
        otherwise `None`.
    """
    if env is None:
        env = Env(0)
    ns = env.ns
    na = env.na

    # create states and actions
    mpc = Mpc(Nlp("MX"), horizon, shooting="multi")
    x, x0 = mpc.state("x", ns)
    u, _ = mpc.action("u", na, lb=env.a_lb[:, None], ub=env.a_ub[:, None])

    # set other action constraints
    u_prev = mpc.parameter("u_prev", (na,))
    du = cs.diff(cs.horzcat(u_prev[[1, 2]], u[[1, 2], :]), 1, 1)
    dtiltmax = env.dtiltmax * env.sampling_time
    mpc.constraint("max_tilt", cs.cos(u[1, :]) * cs.cos(u[2, :]), ">=", env.tiltmax)
    mpc.constraint("min_dtilt", du, ">=", -dtiltmax)
    mpc.constraint("max_dtilt", du, "<=", dtiltmax)

    # set the dynamics along the horizon
    dtdynamics = get_discrete_time_dynamics(env)
    mpc.set_nonlinear_dynamics(dtdynamics)

    # set constraints for obstacle avoidance
    no = env.n_obstacles
    pos_obs = mpc.parameter("pos_obs", (3, no))
    dir_obs = mpc.parameter("dir_obs", (3, no))
    context = cs.vvcat(Env.normalize_context(x0, pos_obs, dir_obs))
    kappann = None
    if dcbf:
        h = env.safety_constraints(x, pos_obs, dir_obs)
        if use_kappann:
            in_features = ns + no * 6  # 3 positions + 3 directions
            features = [in_features, *kappann_hidden_size, no]
            activations = [ReLU] * len(kappann_hidden_size) + [Sigmoid]
            kappann = Mlp(features, activations)
            nnfunc = nn2function(kappann, "kappann")
            weights = {
                n: mpc.parameter(n, p.shape)
                for n, p in kappann.parameters(prefix="kappann", skip_none=True)
            }
            gammas = nnfunc(x=context, **weights)["y"]
            powers = range(1, horizon + 1)
            decays = cs.hcat([cs.power(1 - gammas[i], powers) for i in range(no)])
            dcbf = h[:, 1:] - decays.T * h[:, 0]
        else:
            decays = cs.power(1 - DCBF_GAMMA, range(1, horizon + 1))
            dcbf = h[:, 1:] - h[:, 0] @ decays.T
        slack = mpc.constraint("obs", dcbf, ">=", 0.0, soft=soft)[-1]
    else:
        x_ = x if bound_initial_state else x[:, 1:]
        h = env.safety_constraints(x_, pos_obs, dir_obs)
        slack = mpc.constraint("obs", h, ">=", 0.0, soft=soft)[-1]

    # compute stage cost
    dx = x - env.xf
    Q = env.Q
    R = env.R
    J = sum(cs.bilin(Q, dx[:, i]) + cs.bilin(R, u[:, i]) for i in range(horizon))

    # compute terminal cost
    pwqnn = psdnn = None
    if "dlqr" in terminal_cost:
        ldynamics = dtdynamics.factory("dyn_lin", ("x", "u"), ("jac:xf:x", "jac:xf:u"))
        A, B = ldynamics(env.xf, env.a0)
        P = dlqr(A.toarray(), B.toarray(), Q, R)
        J += cs.bilin(P, dx[:, -1])

    if "pwqnn" in terminal_cost:
        # NOTE: the PWQ NN is not made aware of the positions and directions of the
        # obstacles (since they are not passed as inputs), so its context is limited to
        # the quadrotor's state
        pwqnn = PwqNN(ns, pwqnn_hidden_size)
        nnfunc = nn2function(pwqnn, "pwqnn")
        nnfunc = nnfunc.factory("F", nnfunc.name_in(), ("y", "grad:y:x", "hess:y:x:x"))
        weights = {
            n: mpc.parameter(n, p.shape)
            for n, p in pwqnn.parameters(prefix="pwqnn", skip_none=True)
        }
        output = nnfunc(x=x0 - env.xf, **weights)
        dxT = x[:, -1] - x0
        val, jac, hess = output["y"], output["grad_y_x"], output["hess_y_x_x"]
        J += val + cs.dot(jac, dxT) + 0.5 * cs.bilin(hess, dxT)

    if "psdnn" in terminal_cost:
        in_features = ns + no * 6  # 3 positions + 3 directions
        psdnn = PsdNN(in_features, psdnn_hidden_sizes, ns, "tril")
        nnfunc = nn2function(psdnn, "psdnn")
        weights = {
            n: mpc.parameter(n, p.shape)
            for n, p in psdnn.parameters(prefix="psdnn", skip_none=True)
        }
        L = nnfunc(x=context, **weights)["y"]
        J += cs.bilin(L @ L.T, dx[:, -1])

    # add penalty cost (if needed)
    if soft:
        J += env.constraint_penalty * cs.sum1(cs.sum2(slack))

    # set the solver
    mpc.minimize(J)
    solver = "ipopt"
    mpc.init_solver(SOLVER_OPTS[solver], solver, type="nlp")
    return mpc, kappann, pwqnn, psdnn


def get_mpc_controller(*args: Any, seed: RngType = None, **kwargs: Any) -> tuple[
    Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]],
    dict[str, npt.NDArray[np.floating]],
]:
    """Returns the MPC controller as a callable function.

    Parameters
    ----------

    args, kwargs
        The arguments to pass to the `create_mpc` method.
    seed : RngType, optional
        The seed used during creating of the neural networks parameters, if necessary.


    Returns
    -------
    callable from (array-like, QuadrotorEnv) to (array-like, float)
        A controller that maps the current state to the desired action, and returns also
        the time it took to compute the action.
    dict of str to arrays
        The numerical weights of the parametric MPC controller.
    """
    # create the MPC
    mpc, kappann, pwqnn, psdnn = create_mpc(*args, **kwargs)

    # group its NN parameters (if any) into a vector and assign numerical values to them
    sym_weights_, num_weights_ = {}, {}
    for nn, prefix in [(kappann, "kappann"), (pwqnn, "pwqnn"), (psdnn, "psdnn")]:
        if nn is None:
            continue
        if hasattr(nn, "init_parameters"):
            nn_weights = dict(nn.init_parameters(prefix=prefix, seed=seed))
        else:
            nn_weights = dict(init_parameters(nn, prefix=prefix, seed=seed))
        sym_weights_.update((k, mpc.parameters[k]) for k in nn_weights)
        num_weights_.update(nn_weights)
    sym_weights = cs.vvcat(sym_weights_.values())
    num_weights = cs.vvcat(num_weights_.values())

    # group the symbolical inputs of the MPC controller
    primals = mpc.nlp.x
    args_in = (
        primals,
        mpc.initial_states["x_0"],
        mpc.parameters["u_prev"],
        mpc.parameters["pos_obs"],
        mpc.parameters["dir_obs"],
        sym_weights,
    )

    # convert the MPC object to a function (it's faster than going through csnlp)
    func = mpc.nlp.to_function("mpc", args_in, (primals, mpc.actions["u"][:, 0]))
    last_sol = 0.0

    def _f(x, env):
        nonlocal last_sol
        last_sol, u_opt = func(
            last_sol, x, env.previous_action, env.pos_obs, env.dir_obs, num_weights
        )
        return u_opt.toarray().reshape(-1), func.stats()[TIME_MEAS]

    def reset():
        nonlocal last_sol
        last_sol = 0.0

    _f.reset = reset

    return _f, num_weights_
