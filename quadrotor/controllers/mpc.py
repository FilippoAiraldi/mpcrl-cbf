from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from csnn import init_parameters
from env import QuadrotorEnv as Env
from mpcrl.util.seeding import RngType
from scipy.linalg import solve_discrete_are as dlqr

from util.defaults import DCBF_GAMMA, SAFETY_BACKOFF_SQRT, SOLVER_OPTS, TIME_MEAS
from util.nn import QuadrotorNN, nn2function


def create_mpc(
    horizon: int,
    dcbf: bool,
    use_kappann: bool,
    soft: bool,
    bound_initial_state: bool,
    terminal_cost: set[Literal["dlqr", "psdnn"]],
    nn_hidden_sizes: Sequence[int],
    env: Env | None = None,
    *_: Any,
    **__: Any,
) -> tuple[Mpc[cs.MX], QuadrotorNN | None]:
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
    nn_hidden_sizes : sequence of int
        The number of hidden units in the neural network for the  positive semidefinite
        terminal cost and class Kappa function, if used.
    env : Env | None, optional
        The environment to build the MPC for. If `None`, a new default environment is
        instantiated.

    Returns
    -------
    Mpc, QuadrotorNN (optional)
        The multiple-shooting nonlinear MPC controller; the neural network used to model
        the positive semidefinite terminal cost and class Kappa function if `dcbf=True`
        and `use_kappann=True` or if `"psdnn"` is in `terminal_cost`, otherwise `None`,
        otherwise `None`.
    """
    if env is None:
        env = Env(0)
    ns, na = env.ns, env.na

    # create states and actions
    mpc = Mpc(Nlp("MX"), horizon, shooting="multi")
    x, x0 = mpc.state("x", ns, lb=env.x_lb[:, None], ub=env.x_ub[:, None])
    u, _ = mpc.action("u", na, lb=env.a_lb[:, None], ub=env.a_ub[:, None])

    # set other action constraints
    u_prev = mpc.parameter("u_prev", (na,))
    du = cs.diff(cs.horzcat(u_prev[[1, 2]], u[[1, 2], :]), 1, 1)
    dtiltmax = env.dtiltmax * env.sampling_time
    mpc.constraint("max_tilt", cs.cos(u[1, :]) * cs.cos(u[2, :]), ">=", env.tiltmax)
    mpc.constraint("min_dtilt", du, ">=", -dtiltmax)
    mpc.constraint("max_dtilt", du, "<=", dtiltmax)

    # set the dynamics along the horizon
    mpc.set_nonlinear_dynamics(lambda x_, u_: env.dtdynamics(x_, u_, 0.0))

    # create the neural network for terminal cost and Kappa function if needed
    dx = x - env.xf
    h0 = env.safety_constraint(x0)
    if (dcbf and use_kappann) or "psdnn" in terminal_cost:
        ctx_features = ns + na + 1
        net = QuadrotorNN(ctx_features, nn_hidden_sizes, ns, "tril")
        weights = {
            n: mpc.parameter(n, p.shape)
            for n, p in net.parameters(prefix="nn", skip_none=True)
        }
        func = nn2function(net, "nn")
        context = cs.veccat(x0, u_prev, h0)
        outputs = func(x=dx[:, -1], context=context, **weights)
        nn_V, nn_gamma = outputs["V"], outputs["gamma"]
    else:
        net = None

    # set constraints for obstacle avoidance
    backoff = SAFETY_BACKOFF_SQRT**2
    if dcbf:
        h = env.safety_constraint(x[:, 1:])
        gamma = nn_gamma if use_kappann else DCBF_GAMMA
        decays = cs.power(1 - gamma, range(1, horizon + 1))
        dcbf = h - decays.T * h0  # unrolled CBF constraints
        slack = mpc.constraint("obs", dcbf, ">=", backoff, soft=soft)[-1]
    else:
        h = env.safety_constraint(x if bound_initial_state else x[:, 1:])
        slack = mpc.constraint("obs", h, ">=", backoff, soft=soft)[-1]

    # compute stage cost
    Q = mpc.parameter("Q", (ns,))  # we square these to ensure PSD
    R = mpc.parameter("R", (na,))
    J = sum(cs.sumsqr(Q * dx[:, i]) + cs.sumsqr(R * u[:, i]) for i in range(horizon))

    # compute terminal cost
    if "dlqr" in terminal_cost:
        A, B = env.lindtdynamics(env.a0)
        P = dlqr(A.toarray(), B.toarray(), np.diag(env.Q), np.diag(env.R))
        J += cs.bilin(P, dx[:, -1])
    if "psdnn" in terminal_cost:
        J += nn_V

    # add penalty cost (if needed)
    if soft:
        J += env.constraint_penalty * cs.sum1(cs.sum2(slack))

    # set the solver
    mpc.minimize(J)
    solver = "ipopt"
    mpc.init_solver(SOLVER_OPTS[solver], solver, type="nlp")
    return mpc, net


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
        A controller that maps the current state + env to the desired action, and
        returns also the time it took to compute the action.
    dict of str to arrays
        The numerical weights of the parametric MPC controller, if any.
    """
    # create the MPC
    mpc, net = create_mpc(*args, **kwargs)

    # group its NN parameters (if any) into a dict and assign numerical values to them,
    # as well as for the stage cost parameters
    sym_weights_, num_weights_ = {}, {}
    if net is not None:
        nn_weights = dict(init_parameters(net, prefix="nn", seed=seed))
        sym_weights_.update((k, mpc.parameters[k]) for k in nn_weights)
        num_weights_.update(nn_weights)
    for k in ("Q", "R"):
        sym_weights_[k] = mpc.parameters[k]
        num_weights_[k] = np.sqrt(getattr(Env, k))

    # group the symbolical inputs of the MPC controller
    primals = mpc.nlp.x
    sym_weights = cs.vvcat([sym_weights_[k] for k in sym_weights_])
    num_weights = cs.vvcat([num_weights_[k] for k in sym_weights_])
    args_in = (
        primals,
        mpc.initial_states["x_0"],
        mpc.parameters["u_prev"],
        sym_weights,
    )

    # convert the MPC object to a function (it's faster than going through csnlp)
    func = mpc.nlp.to_function("mpc", args_in, (primals, mpc.actions["u"][:, 0]))
    last_sol = 0.0

    def _f(x, env):
        nonlocal last_sol
        last_sol, u_opt = func(last_sol, x, env.previous_action, num_weights)
        return u_opt.toarray().reshape(-1), func.stats()[TIME_MEAS]

    def reset():
        nonlocal last_sol
        last_sol = 0.0

    _f.reset = reset

    return _f, num_weights_
