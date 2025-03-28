from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import ScenarioBasedMpc
from csnn import init_parameters
from env import QuadrotorEnv as Env
from mpcrl.util.seeding import RngType
from scipy.linalg import solve_discrete_are as dlqr

from util.defaults import DCBF_GAMMA, SAFETY_BACKOFF_SQRT, SOLVER_OPTS, TIME_MEAS
from util.nn import QuadrotorNN, nn2function


def create_scmpc(
    horizon: int,
    scenarios: int,
    dcbf: bool,
    use_kappann: bool,
    soft: bool,
    bound_initial_state: bool,
    terminal_cost: set[Literal["dlqr", "psdnn"]],
    nn_hidden_sizes: Sequence[int],
    env: Env | None = None,
    *_: Any,
    **__: Any,
) -> tuple[ScenarioBasedMpc[cs.MX], QuadrotorNN | None]:
    """Creates a nonlinear MPC controller for the `QuadrotorEnv` environment.

    Parameters
    ----------
    horizon : int
        The prediction horizon of the controller.
    scenarios : int
        The number of scenarios to consider in the SCMPC controller.
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
    ScenarioBasedMpc, QuadrotorNN (optional)
        The multiple-shooting nonlinear scenario-based MPC controller; the neural
        network used to model the positive semidefinite terminal cost and class Kappa
        function if `dcbf=True` and `use_kappann=True` or if `"psdnn"` is in
        `terminal_cost`, otherwise `None`, otherwise `None`.
    """
    if env is None:
        env = Env(0)
    ns, na, nd = env.ns, env.na, env.nd

    # create states, actions and disturbances
    scmpc = ScenarioBasedMpc(Nlp("MX"), scenarios, horizon, shooting="multi")
    x, _, x0 = scmpc.state("x", ns, lb=env.x_lb[:, None], ub=env.x_ub[:, None])
    u, _ = scmpc.action("u", na, lb=env.a_lb[:, None], ub=env.a_ub[:, None])
    scmpc.disturbance("w", nd)

    # set other action constraints
    u_prev = scmpc.parameter("u_prev", (na,))
    du = cs.diff(cs.horzcat(u_prev[[1, 2]], u[[1, 2], :]), 1, 1)
    dtiltmax = env.dtiltmax * env.sampling_time
    scmpc.constraint("max_tilt", cs.cos(u[1, :]) * cs.cos(u[2, :]), ">=", env.tiltmax)
    scmpc.constraint("min_dtilt", du, ">=", -dtiltmax)
    scmpc.constraint("max_dtilt", du, "<=", dtiltmax)

    # set the dynamics along the horizon
    scmpc.set_nonlinear_dynamics(env.dtdynamics)

    # create the neural network for terminal cost and Kappa function if needed
    dx = x - env.xf
    h0 = env.safety_constraint(x0)
    if (dcbf and use_kappann) or "psdnn" in terminal_cost:
        ctx_features = ns + na + 1
        net = QuadrotorNN(ctx_features, nn_hidden_sizes, ns, "tril")
        weights = {
            n: scmpc.parameter(n, p.shape)
            for n, p in net.parameters(prefix="nn", skip_none=True)
        }
        func = nn2function(net, "nn")
        context = cs.veccat(x0, u_prev, h0)
        outputs = func(x=dx[:, -1], context=context, **weights)
        nn_V, nn_gamma = outputs["V"], outputs["gamma"]
    else:
        net = None

    # set constraints for obstacle avoidance
    backoff = scmpc.parameter("backoff", (1,)) ** 2
    if dcbf:
        h = env.safety_constraint(x[:, 1:])
        gamma = nn_gamma if use_kappann else DCBF_GAMMA
        decays = cs.power(1 - gamma, range(1, horizon + 1))
        dcbf = h - decays.T * h0  # unrolled CBF constraints
        slack = scmpc.constraint_from_single("obs", dcbf, ">=", backoff, soft=soft)[-2]
    else:
        h = env.safety_constraint(x if bound_initial_state else x[:, 1:])
        slack = scmpc.constraint_from_single("obs", h, ">=", backoff, soft=soft)[-2]

    # compute stage cost
    Q = scmpc.parameter("Q", (ns,))  # we square these to ensure PSD
    R = scmpc.parameter("R", (na,))
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
    scmpc.minimize_from_single(J)
    solver = "ipopt"
    scmpc.init_solver(SOLVER_OPTS[solver], solver, type="nlp")
    return scmpc, net


def get_scmpc_controller(
    *args: Any,
    seed: RngType = None,
    weights: dict[str, npt.NDArray[np.floating]] | None = None,
    **kwargs: Any,
) -> tuple[
    Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]],
    dict[str, npt.NDArray[np.floating]],
]:
    """Returns the scenario-based MPC controller as a callable function.

    Parameters
    ----------

    args, kwargs
        The arguments to pass to the `create_mpc` method.
    seed : RngType, optional
        The seed used during creating of the neural networks parameters, if necessary.
    weights : dict of (str, array), optional
        The MPC weights to use in the controller. If `None`, the weights are initialized
        randomly or to default values.

    Returns
    -------
    callable from (array-like, QuadrotorEnv) to (array-like, float)
        A controller that maps the current state to the desired action, and returns also
        the time it took to compute the action.
    dict of str to arrays, optional
        The numerical weights of the neural network used to learn the DCBF Kappa
        function (used only for saving to disk for plotting).
    """
    # create the MPC
    scmpc, net = create_scmpc(*args, **kwargs)

    # group its parameters (if any) into a dict and assign numerical values to them by
    # either pulling from the given weights or initializing them
    sym_weights_, num_weights_ = {}, {}
    if weights is None:
        weights = {}
    if net is not None:
        nn_weights_ = dict(init_parameters(net, prefix="nn", seed=seed))
        for k, v in nn_weights_.items():
            sym_weights_[k] = scmpc.parameters[k]
            num_weights_[k] = weights.get(k, v)
    for k in ("Q", "R"):
        sym_weights_[k] = scmpc.parameters[k]
        num_weights_[k] = weights.get(k, np.sqrt(getattr(Env, k)))
    sym_weights_["backoff"] = scmpc.parameters["backoff"]
    num_weights_["backoff"] = weights.get("backoff", SAFETY_BACKOFF_SQRT)

    # group the symbolical inputs of the MPC controller
    primals = scmpc.nlp.x
    sym_weights = cs.vvcat([sym_weights_[k] for k in sym_weights_])
    num_weights = cs.vvcat([num_weights_[k] for k in sym_weights_])
    disturbances = cs.vvcat(scmpc.disturbances.values())
    args_in = (
        primals,
        scmpc.initial_states["x_0"],
        scmpc.parameters["u_prev"],
        sym_weights,
        disturbances,
    )

    # convert the MPC object to a function (it's faster than going through csnlp)
    func = scmpc.nlp.to_function("scmpc", args_in, (primals, scmpc.actions["u"][:, 0]))
    last_sol = 0.0
    horizon, n_scenarios = scmpc.prediction_horizon, scmpc.n_scenarios

    def _f(x, env: Env):
        nonlocal last_sol
        d = env.sample_disturbance_profiles(n_scenarios, horizon).reshape(-1)
        last_sol, u_opt = func(last_sol, x, env.previous_action, num_weights, d)
        return u_opt.toarray().reshape(-1), func.stats()[TIME_MEAS]

    def reset():
        nonlocal last_sol
        last_sol = 0.0

    _f.reset = reset

    return _f, num_weights_
