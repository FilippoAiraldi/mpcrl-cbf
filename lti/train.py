import argparse
import sys
from datetime import datetime
from logging import DEBUG
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from csnn import set_sym_type
from joblib import Parallel, delayed
from mpcrl import LearnableParameter, LearnableParametersDict, RlLearningAgent
from mpcrl.util.seeding import RngType
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

sys.path.append(str(Path(__file__).resolve().parents[1]))

from controllers.scmpc import create_scmpc
from env import ConstrainedLtiEnv as Env

from util.wrappers import RecordSolverTime, SuppressOutput


def get_agent(
    algorithm: Literal["lstd-ql", "lstd-dpg"], *args: Any, **kwargs: Any
) -> RlLearningAgent:
    """Returns the agent given its algorithm.

    Parameters
    ----------
    algorithm : {"lstd-ql", "lstd-dpg"}
        The agent's algorithm to construct.
    args, kwargs
        The arguments to pass to the agent constructor.

    Returns
    -------
    RlLearningAgent
        An instance of the learning agent for the given algorithm.

    Raises
    ------
    ValueError
        Raises an error if the algorithm is not recognized.
    """
    if algorithm == "lstd-ql":
        from agents.lstd_ql import get_lstd_qlearning_agent as func
    elif algorithm == "lstd-dpg":
        from agents.lstd_dpg import get_lstd_dpg_agent as func
    else:
        raise ValueError(f"Unknown agent: {algorithm}")
    return func(*args, **kwargs)


def train_one_agent(
    algorithm: Literal["lstd-ql", "lstd-dpg"],
    scmpc_kwargs: dict[str, Any],
    learning_rate: float,
    exploration_epsilon: tuple[float, float],
    exploration_strength: tuple[float, float],
    episodes: int,
    timesteps: int,
    reset_kwargs: dict[str, Any],
    seed: RngType,
    n: int,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    dict[str, npt.NDArray[np.floating]],
    npt.NDArray[np.floating],
]:
    """Trains an agent with the specified algorithm on the constrained LTI environment.

    Parameters
    ----------
    algorithm : {"lstd-ql", "lstd-dpg"}
        The algorithm to use for training the RL agent.
    scmpc_kwargs : dict
        The arguments to pass to the controller instantiation.
    learning_rate : float
        The learning rate of the agent.
    exploration_epsilon : (float, float)
        A tuple containing the initial exploration probability and its episodic decay
        rate.
    exploration_strength : (float, float)
        A tuple containing the initial exploration strength and its episodic decay rate.
    episodes : int
        The number of episodes to train the agent for
    timesteps : int
        The number of timesteps to run each evaluation for.
    reset_kwargs : dict of str to any
        Optional arguments to pass to the environment's reset method.
    seed : int
        The seed for the random number generator.
    n : int
        The identifying number/index of this agent.

    Returns
    -------
    4 float arrays, 1 dict of float arrays, 1 float array
        Returns, in this order, the total cost for each training episode, two arrays
        containing the actions and states trajectories respectively, an array of
        solution computation times (for `V(s)` and `Q(s,a)`, respectively), a dictionary
        containing the history of updates to the learnable parameters, and an array of
        TD errors (only for the `"lstd-ql"` algorithm).
    """
    # instantiate the environment
    rng = np.random.default_rng(seed)
    env = Env(timesteps)

    # create the SCMPC controller
    set_sym_type("MX")
    scmpc, pwqnn = create_scmpc(**scmpc_kwargs, env=env)
    scmpc = RecordSolverTime(scmpc)
    if scmpc.unwrapped._solver_plugin == "qpoases":
        scmpc = SuppressOutput(scmpc)

    # initialize learnable parameters
    sym_pars = scmpc.parameters
    learnable_pars_ = []
    if pwqnn is not None:
        for name, weight in pwqnn.init_parameters(prefix="pwqnn", seed=rng):
            if name.endswith("dot.weight"):
                lb, ub = 0.0, np.inf  # force convexity of nn function
            elif name.endswith("input_layer.bias"):
                lb, ub = -np.inf, 0.0  # force value at origin to be close to zero
            else:
                lb, ub = -np.inf, np.inf
            learnable_pars_.append(
                LearnableParameter(name, weight.shape, weight, lb, ub, sym_pars[name])
            )
    learnable_pars = LearnableParametersDict(learnable_pars_)

    # instantiate and wrap the agent
    agent = get_agent(
        algorithm,
        scmpc=scmpc,
        learnable_parameters=learnable_pars,
        batch_size=timesteps,
        learning_rate=learning_rate,
        exploration_epsilon=exploration_epsilon,
        exploration_strength=exploration_strength,
        name=f"{algorithm}_{n}",
    )
    agent = RecordUpdates(agent)
    agent = Log(agent, level=DEBUG, log_frequencies={"on_episode_end": 100})

    # launch training
    env = MonitorEpisodes(env, deque_size=episodes)
    R = agent.train(env, episodes, rng, True, reset_kwargs)
    agent.detach_wrapper(True)  # prevents joblib from getting stuck

    # extract and return the data from the environment and the agent
    U = np.asarray(env.actions).squeeze(-1)
    X = np.asarray(env.observations)
    updates_history = {n: np.asarray(par) for n, par in agent.updates_history.items()}
    others = []
    if algorithm == "lstd-ql":
        sol_times = np.stack(
            (
                np.reshape(agent.V.solver_time, (episodes, timesteps + 1))[:, 1:],
                np.reshape(agent.Q.solver_time, (episodes, timesteps)),
            ),
            axis=-1,
        )
        others.append(np.reshape(agent.td_errors, (episodes, timesteps)))
    elif algorithm == "lstd-dpg":
        sol_times = np.reshape(agent.V.solver_time, (episodes, timesteps))
        others.append(
            np.reshape(agent.policy_gradients, (episodes, learnable_pars.size)),
        )
    return R, U, X, sol_times, updates_history, *others


if __name__ == "__main__":
    # parse script arguments
    default_save = f"train_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    parser = argparse.ArgumentParser(
        description="Training of SCMPC-RL agents on the constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Choice of RL algorithm")
    group.add_argument(
        "algorithm",
        choices=("lstd-ql", "lstd-dpg"),
        help="The algorithm to use for training the RL agents.",
    )
    group = parser.add_argument_group("RL algorithm options")
    group.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate of the RL algorithm.",
    )
    group.add_argument(
        "--exploration-eps",
        type=float,
        default=(1.0, 0.997),
        nargs=2,
        help="The chance (epsilon) of exploration (probability and episodic decay).",
    )
    group.add_argument(
        "--exploration-str",
        type=float,
        default=(1.0, 0.997),
        nargs=2,
        help="The strength of exploration (strength and episodic decay).",
    )
    group = parser.add_argument_group("Scenario MPC (SCMPC) options")
    group.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="The horizon of the MPC controller.",
    )
    group.add_argument(
        "--scenarios",
        type=int,
        default=32,
        help="The number of scenarios to use in the SCMPC controller.",
    )
    group.add_argument(
        "--dcbf",
        action="store_true",
        help="Whether to use discrete-time CBF constraints in the SCMPC controller.",
    )
    group.add_argument(
        "--soft",
        action="store_true",
        help="Whether to use soft constraints in the SCMPC controller.",
    )
    group.add_argument(
        "--bound-initial-state",
        action="store_true",
        help="Whether to bound the initial state in the SCMPC controller.",
    )
    group.add_argument(
        "--terminal-cost",
        choices=("dlqr", "pwqnn"),
        nargs="*",
        default={"pwqnn"},
        help="Which type of terminal cost to use in the SCMPC controller.",
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument(
        "--n-agents", type=int, default=10, help="Number of agents to train."
    )
    group.add_argument(
        "--n-episodes",
        type=int,
        default=1000,
        help="Number of training episodes per agent.",
    )
    group.add_argument(
        "--timesteps",
        type=int,
        default=30,
        help="Number of timesteps per each training episode.",
    )
    group.add_argument(
        "--ic",
        choices=("contour", "interior", "box"),
        default="contour",
        help="Sets whether the initial conditions (i.e., initial state) of the "
        "environment is drawn from the contour or interior of the max. invariant set, "
        "or its bounding box.",
    )
    group = parser.add_argument_group("Storing and plotting options")
    group.add_argument(
        "--save",
        type=str,
        default=default_save,
        help="Saves results with this filename. If not set, a default name is given.",
    )
    group.add_argument(
        "--plot",
        action="store_true",
        help="Shows results in a plot at the end of the simulation.",
    )
    group = parser.add_argument_group("Computational options")
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    args = parser.parse_args()
    args.terminal_cost = set(args.terminal_cost)
    print(f"Args: {args}\n")

    # prepare arguments to the training simulation
    algo = args.algorithm
    scmpc_kwargs = {
        "horizon": args.horizon,
        "dcbf": args.dcbf,
        "soft": args.soft,
        "bound_initial_state": args.bound_initial_state,
        "terminal_cost": args.terminal_cost,
        "scenarios": args.scenarios,
    }
    lr = args.lr
    eps = args.exploration_eps
    strength = (args.exploration_str[0] * 2 * Env.a_bound, args.exploration_str[1])
    n_episodes = args.n_episodes
    ts = args.timesteps
    reset_kwargs = {"ic": args.ic}
    seeds = np.random.SeedSequence(args.seed).spawn(args.n_agents)

    # run the simulations (possibly in parallel asynchronously)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(train_one_agent)(
            algo, scmpc_kwargs, lr, eps, strength, n_episodes, ts, reset_kwargs, seed, i
        )
        for i, seed in enumerate(seeds)
    )

    # congregate data all together - updates_history is a dictionary, so requires
    # further attention
    keys = ["cost", "actions", "states", "sol_times", "updates_history"]
    if algo == "lstd-ql":
        keys.append("td_errors")
    elif algo == "lstd-dpg":
        keys.append("policy_gradients")
    data_dict = dict(zip(keys, map(np.asarray, zip(*data))))
    par_names = data_dict["updates_history"][0].keys()
    data_dict["updates_history"] = {
        n: np.asarray([d[n] for d in data_dict["updates_history"]]) for n in par_names
    }

    # finally, store and plot the results. If no filepath is passed, always plot
    if args.save:
        from csnlp.util.io import save

        save(args.save, **data_dict, args=args.__dict__, compression="lzma")
    if args.plot or not args.save:
        import matplotlib.pyplot as plt
        from plot import (
            plot_returns,
            plot_solver_times,
            plot_states_and_actions,
            plot_terminal_cost_evolution,
            plot_training,
        )

        data, args = [data_dict], [args.__dict__]
        plot_states_and_actions(data)
        plot_returns(data)
        plot_solver_times(data)
        plot_training(data)
        plot_terminal_cost_evolution(data, args)
        plt.show()
