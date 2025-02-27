from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from csnn import ReLU, Sigmoid, init_parameters
from csnn.convex import PsdNN
from csnn.feedforward import Mlp
from env import NORMALIZATION as N
from env import QuadrotorEnv as Env
from mpcrl.util.control import rk4
from mpcrl.util.seeding import RngType
from scipy.linalg import solve_discrete_are as dlqr

from util.defaults import DCBF_GAMMA, SOLVER_OPTS, TIME_MEAS
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
    x, u, d = env.dynamics.mx_in()
    args = [x, u]
    argnames = ["x", "u"]
    if include_disturbance:
        args.append(d)
        argnames.append("d")
    else:
        d = 0.0
    xf = cs.simplify(rk4(lambda x_: env.dynamics(x_, u, d), x, env.sampling_time))
    return cs.Function("dynamics", args, [xf], argnames, ["xf"], {"cse": True})


def create_mpc(
    horizon: int,
    dcbf: bool,
    use_kappann: bool,
    soft: bool,
    bound_initial_state: bool,
    terminal_cost: set[Literal["dlqr", "psdnn"]],
    kappann_hidden_size: Sequence[int],
    psdnn_hidden_sizes: Sequence[int],
    env: Env | None = None,
    *_: Any,
    **__: Any,
) -> tuple[Mpc[cs.MX], Mlp | None, PsdNN | None]:
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
    terminal_cost : set of {"dlqr", "psdnn"}
        The type of terminal cost to use. If "dlqr", the terminal cost is the solution
        to the discrete-time LQR problem. If "psdnn", a positive semidefinite neural
        network is used to approximate the terminal cost. Can also be a set of multiple
        terminal costs to use, at which point these are summed together; can also be an
        empty set, in which case no terminal cost is used.
    kappann_hidden_size : sequence of int
        The number of hidden units in the multilayer perceptron used to learn the
        DCBF Kappa function, if `dcbf` is `True`.
    psdnn_hidden_sizes : sequence of int
        The number of hidden units in the positive semidefinite neural network, if used.
    env : Env | None, optional
        The environment to build the MPC for. If `None`, a new default environment is
        instantiated.

    Returns
    -------
    Mpc, Mlp (optional), and PsdNN (optional)
        The multiple-shooting nonlinear MPC controller; the multilayer perceptron if
        `dcbf` is `True`, otherwise `None`; the positive semidefinite terminal cost
        neural net if `"psdnn"` is in `terminal_cost`, otherwise `None`.
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
    kappann = None
    if dcbf:
        h0 = env.safety_constraints(x0, pos_obs, dir_obs)
        h = env.safety_constraints(x[:, 1:], pos_obs, dir_obs)
        if use_kappann:
            in_features = ns + na + no * 6  # 3 positions + 3 directions
            kappann = Mlp(
                features=[in_features, *kappann_hidden_size, no],
                acts=[ReLU] * len(kappann_hidden_size) + [Sigmoid],
            )
            nnfunc = nn2function(kappann, "kappann")
            weights = {
                n: mpc.parameter(n, p.shape)
                for n, p in kappann.parameters(prefix="kappann", skip_none=True)
            }
            context = (cs.veccat(x0, u_prev, pos_obs, dir_obs) - N[0]) / N[1]
            gammas = nnfunc(x=context, **weights)["y"]
        else:
            gammas = [DCBF_GAMMA] * no
        decays = cs.hcat(
            [cs.power(1 - gammas[i], range(1, horizon + 1)) for i in range(no)]
        )
        dcbf = h - decays.T * h0
        slack = mpc.constraint("obs", dcbf, ">=", 0.0, soft=soft)[-1]
    else:
        h = env.safety_constraints(
            x if bound_initial_state else x[:, 1:], pos_obs, dir_obs
        )
        slack = mpc.constraint("obs", h, ">=", 0.0, soft=soft)[-1]

    # compute stage cost
    dx = x - env.xf
    Q = mpc.parameter("Q", (ns,))  # we square these to ensure PSD
    R = mpc.parameter("R", (na,))
    J = sum(cs.sumsqr(Q * dx[:, i]) + cs.sumsqr(R * u[:, i]) for i in range(horizon))

    # compute terminal cost
    psdnn = None
    if "dlqr" in terminal_cost:
        ldynamics = dtdynamics.factory("dyn_lin", ("x", "u"), ("jac:xf:x", "jac:xf:u"))
        A, B = ldynamics(env.xf, env.a0)
        P = dlqr(A.toarray(), B.toarray(), env.Q, env.R)
        J += cs.bilin(P, dx[:, -1])
    if "psdnn" in terminal_cost:
        in_features = ns + na + no * 6  # 3 positions + 3 directions
        psdnn = PsdNN(in_features, psdnn_hidden_sizes, ns, "tril")
        nnfunc = nn2function(psdnn, "psdnn")
        weights = {
            n: mpc.parameter(n, p.shape)
            for n, p in psdnn.parameters(prefix="psdnn", skip_none=True)
        }
        context = (cs.veccat(x0, u_prev, pos_obs, dir_obs) - N[0]) / N[1]
        J += nnfunc(x=dx[:, -1], context=context, **weights)["y"]

    # add penalty cost (if needed)
    if soft:
        J += env.constraint_penalty * cs.sum1(cs.sum2(slack))

    # set the solver
    mpc.minimize(J)
    solver = "ipopt"
    mpc.init_solver(SOLVER_OPTS[solver], solver, type="nlp")
    return mpc, kappann, psdnn


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
    mpc, kappann, psdnn = create_mpc(*args, **kwargs)

    # group its NN parameters (if any) into a dict and assign numerical values to them
    sym_weights_, num_weights_ = {}, {}
    for nn, prefix in [(kappann, "kappann"), (psdnn, "psdnn")]:
        if nn is None:
            continue
        nn_weights = dict(init_parameters(nn, prefix=prefix, seed=seed))
        sym_weights_.update((k, mpc.parameters[k]) for k in nn_weights)
        num_weights_.update(nn_weights)

    # also add the stage cost parameters
    for k in ("Q", "R"):
        sym_weights_[k] = mpc.parameters[k]
        num_weights_[k] = np.sqrt(np.diag(getattr(Env, k)))

    # group the symbolical inputs of the MPC controller
    primals = mpc.nlp.x
    sym_weights = cs.vvcat([sym_weights_[k] for k in sym_weights_])
    num_weights = cs.vvcat([num_weights_[k] for k in sym_weights_])
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
