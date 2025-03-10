from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from controllers.mpc import get_discrete_time_dynamics
from csnlp import Nlp
from csnlp.wrappers import ScenarioBasedMpc
from csnn import ReLU, Sigmoid, init_parameters
from csnn.convex import PsdNN
from csnn.feedforward import Mlp
from env import NORMALIZATION as N
from env import QuadrotorEnv as Env
from mpcrl.util.seeding import RngType
from scipy.linalg import solve_discrete_are as dlqr

from util.defaults import DCBF_GAMMA, SOLVER_OPTS, TIME_MEAS
from util.nn import nn2function


def create_scmpc(
    horizon: int,
    scenarios: int,
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
) -> tuple[ScenarioBasedMpc[cs.MX], Mlp | None, PsdNN | None]:
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
    ns, na, nd = env.ns, env.na, env.nd

    # create states, actions and disturbances
    scmpc = ScenarioBasedMpc(Nlp("MX"), scenarios, horizon, shooting="multi")
    x_lb = np.full((ns, 1), -10.0)
    x_ub = np.reshape([20.0, 20.0, 20.0, 10.0, 10.0, 10.0], (ns, 1))
    x, _, x0 = scmpc.state("x", ns, lb=x_lb, ub=x_ub)
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
    dtdynamics = get_discrete_time_dynamics(env, include_disturbance=True)
    scmpc.set_nonlinear_dynamics(dtdynamics)

    # set constraints for obstacle avoidance
    h0 = env.safety_constraint(x0)
    ctx_features = ns + na + 1
    context = (cs.veccat(x0, u_prev, h0) - N[0]) / N[1]
    kappann = None
    if dcbf:
        h = env.safety_constraint(x[:, 1:])
        if use_kappann:
            kappann = Mlp(
                features=[ctx_features, *kappann_hidden_size, 1],
                acts=[ReLU] * len(kappann_hidden_size) + [Sigmoid],
            )
            nnfunc = nn2function(kappann, "kappann")
            weights = {
                n: scmpc.parameter(n, p.shape)
                for n, p in kappann.parameters(prefix="kappann", skip_none=True)
            }
            gamma = nnfunc(x=context, **weights)["y"]
        else:
            gamma = DCBF_GAMMA
        decays = cs.power(1 - gamma, range(1, horizon + 1))
        dcbf = h - decays.T * h0
        slack = scmpc.constraint_from_single("obs", dcbf, ">=", 0.0, soft=soft)[-2]
    else:
        h = env.safety_constraint(x if bound_initial_state else x[:, 1:])
        slack = scmpc.constraint_from_single("obs", h, ">=", 0.0, soft=soft)[-2]

    # compute stage cost
    dx = x - env.xf
    Q = scmpc.parameter("Q", (ns,))  # we square these to ensure PSD
    R = scmpc.parameter("R", (na,))
    J = sum(cs.sumsqr(Q * dx[:, i]) + cs.sumsqr(R * u[:, i]) for i in range(horizon))

    # compute terminal cost
    psdnn = None
    if "dlqr" in terminal_cost:
        ldynamics = dtdynamics.factory("dyn_lin", ("x", "u"), ("jac:xf:x", "jac:xf:u"))
        A, B = ldynamics(env.xf, env.a0)
        P = dlqr(A.toarray(), B.toarray(), env.Q, env.R)
        J += cs.bilin(P, dx[:, -1])
    if "psdnn" in terminal_cost:
        psdnn = PsdNN(ctx_features, psdnn_hidden_sizes, ns, "tril")
        nnfunc = nn2function(psdnn, "psdnn")
        weights = {
            n: scmpc.parameter(n, p.shape)
            for n, p in psdnn.parameters(prefix="psdnn", skip_none=True)
        }
        J += nnfunc(x=dx[:, -1], context=context, **weights)["y"]

    # add penalty cost (if needed)
    if soft:
        J += env.constraint_penalty * cs.sum1(cs.sum2(slack))

    # set the solver
    scmpc.minimize_from_single(J)
    solver = "ipopt"
    scmpc.init_solver(SOLVER_OPTS[solver], solver, type="nlp")
    return scmpc, kappann, psdnn


def get_scmpc_controller(
    *args: Any,
    seed: RngType = None,
    weights: dict[str, npt.NDArray[np.floating]] | None = None,
    **kwargs: Any,
) -> tuple[
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
    weights : dict of (str, array), optional
        The MPC weights to use in the controller. If `None`, the weights are initialized
        randomly.

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
    scmpc, kappann, psdnn = create_scmpc(*args, **kwargs)

    # group its parameters (if any) into a dict and assign numerical values to them
    sym_weights_, num_weights_ = {}, {}
    if weights is not None:
        # load numerical values from given weights
        for n, weight in weights.items():
            if n not in scmpc.parameters:
                continue
            sym_weight = scmpc.parameters[n]
            if sym_weight.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{n}'. Expected "
                    f"{sym_weight.shape}; got {weight.shape} instead."
                )
            sym_weights_[n] = sym_weight
            num_weights_[n] = weight
    else:
        # initialize NN weights randomly
        for nn, prefix in [(kappann, "kappann"), (psdnn, "psdnn")]:
            if nn is None:
                continue
            nn_weights_ = dict(init_parameters(nn, prefix=prefix, seed=seed))
            sym_weights_.update((k, scmpc.parameters[k]) for k in nn_weights_)
            num_weights_.update(nn_weights_)

        # also initialize the stage cost parameters
        for k in ("Q", "R"):
            sym_weights_[k] = scmpc.parameters[k]
            num_weights_[k] = np.sqrt(np.diag(getattr(Env, k)))

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
