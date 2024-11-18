import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from controllers.dclf_dcbf import get_dclf_dcbf_controller
from controllers.dlqr import get_dlqr_controller
from csnlp.util.io import save
from env import ConstrainedLtiEnv as Env
from gymnasium.wrappers import TimeLimit
from joblib import Parallel, delayed
from mpcrl.wrappers.envs import MonitorEpisodes
from plot import plot_states_and_actions_and_return


def simulate_controller_once(
    controller: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    timesteps: int,
    seed: int,
    **reset_kwargs: Any,
) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Simulates one episode of the constrained LTI environment using the given
    controller.

    Parameters
    ----------
    controller : callable from array-like to array-like
        A controller that maps the current state to the desired action.
    timesteps : int
        The number of timesteps to simulate.
    seed : int
        The seed for the random number generator.
    reset_kwargs : any
        Optional arguments to pass to the environment's reset method.

    Returns
    -------
    float, and tuple of two arrays
        Returns the total cost of the episode and a tuple of two arrays containing
        action and state trajectories, respectively.
    """
    if reset_kwargs is None:
        reset_kwargs = {}
    env = MonitorEpisodes(TimeLimit(Env(), timesteps), deque_size=1)
    x, _ = env.reset(seed=seed, options=reset_kwargs)
    terminated = truncated = False
    while not (terminated or truncated):
        controller(x)


# def simulate_controller_once(
#     controller_type: Literal["dlqr", "dclf-dcbf"],
#     timesteps: int,
#     seed: int,
#     **reset_kwargs: Any,
# ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
#     """Simulates one episode of the constrained LTI environment using the given
#     controller.

#     Parameters
#     ----------
#     controller : {"dqrl", "dclf-dcbf"}
#         The controller to use for the simulation.
#     timesteps : int
#         The number of timesteps to simulate.
#     seed : int
#         The seed for the random number generator.
#     reset_kwargs : any
#         Optional arguments to pass to the environment's reset method.

#     Returns
#     -------
#     float, and tuple of two arrays
#         Returns the total cost of the episode and a tuple of two arrays containing
#         action and state trajectories, respectively.
#     """
#     if reset_kwargs is None:
#         reset_kwargs = {}

#     if controller_type == "dlqr":
#         controller = get_dlqr_controller()
#     else:
#         controller = get_dclf_dcbf_controller()
#     env = MonitorEpisodes(TimeLimit(Env(), timesteps), deque_size=1)
#     x, _ = env.reset(seed=seed, options=reset_kwargs)
#     terminated = truncated = False
#     while not (terminated or truncated):
#         u = controller(x)
#         x, _, terminated, truncated, _ = env.step(u)
#     R = env.rewards[0].sum()
#     U = env.actions[0]
#     X = env.observations[0]
#     return R, U, X


# def simulate_mpc_agent_once(
#     timesteps: int, seed: int, **reset_kwargs: Any
# ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
#     """Simulates one episode of the constrained LTI environment using an MPC-based
#     control agent.

#     Parameters
#     ----------
#     controller : {"dqrl", "dclf-dcbf"}
#         The controller to use for the simulation.
#     timesteps : int
#         The number of timesteps to simulate.
#     seed : int
#         The seed for the random number generator.
#     reset_kwargs : any
#         Optional arguments to pass to the environment's reset method.

#     Returns
#     -------
#     float, and tuple of two arrays
#         Returns the total cost of the episode and a tuple of two arrays containing
#         action and state trajectories, respectively.
#     """
#     if reset_kwargs is None:
#         reset_kwargs = {}
#     mpc = create_mpc_controller(20)
#     agent = InfeasibleAgent(mpc)
#     env = MonitorEpisodes(TimeLimit(Env(), timesteps), deque_size=1)
#     agent.evaluate(
#         env, 1, seed=seed, env_reset_options=reset_kwargs, terminate_on_infeas=True
#     )
#     if len(env.rewards) == 0:
#         env.force_episode_end()
#     R = env.rewards[0].sum()
#     U = env.actions[0].reshape((-1, env.get_wrapper_attr("na")))
#     X = env.observations[0].reshape((-1, env.get_wrapper_attr("ns")))
#     return R, U, X


if __name__ == "__main__":
    # parse script arguments
    parser = argparse.ArgumentParser(
        description="Simulation of the constrained LTI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Controller options")
    group.add_argument(
        "controller",
        choices=["dlqr", "dclf-dcbf"],
        help="The controller to use for the simulation.",
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument("--n-sim", type=int, default=100, help="Number of simulations.")
    group.add_argument(
        "--timesteps",
        type=int,
        default=30,
        help="Number of timesteps per each simulation.",
    )
    group.add_argument(
        "--init-conditions",
        choices=["contour", "interior"],
        default="contour",
        help="Sets whether the initial state in the environment is on the contour or in"
        " the interior of the safe set.",
    )
    group = parser.add_argument_group("Storing and plotting options")
    group.add_argument(
        "--save",
        type=str,
        default="",
        help="Save results under controllers/data. If not set, no data is saved.",
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

    # get the controller
    controller_name = args.controller
    if controller_name == "dlqr":
        controller = get_dlqr_controller()
    elif controller_name == "dclf-dcbf":
        controller = get_dclf_dcbf_controller()
    else:
        raise RuntimeError(f"Unknown controller: {controller_name}")

    # run the simulations (possibly in parallel asynchronously)
    ic_on_contour = args.init_conditions == "contour"
    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_sim)
    data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
        delayed(simulate_controller_once)(
            controller, args.timesteps, int(seed), contour=ic_on_contour
        )
        for seed in seeds
    )
    # # set up the controller function
    # controller_name = args.controller
    # if controller_name == "mpc":
    #     func = simulate_mpc_agent_once
    # else:
    #     func = partial(simulate_controller_once, controller_name)

    # # run the simulations (possibly in parallel asynchronously)
    # ic_on_contour = args.init_conditions == "contour"
    # seeds = np.random.SeedSequence(args.seed).generate_state(args.n_sim)
    # data = Parallel(n_jobs=args.n_jobs, verbose=10, return_as="generator_unordered")(
    #     delayed(func)(args.timesteps, int(seed), contour=ic_on_contour)
    #     for seed in seeds
    # )

    # finally, store and plot the results. If no filepath is passed, always plot
    if args.save:
        path = Path("data")
        if not path.is_dir():
            path = "lti_example" / path
        path.mkdir(parents=True, exist_ok=True)
        save(str(path / args.save), **data_dict, args=args, compression="lzma")
    if args.plot or not args.save:
        plot_states_and_actions_and_return([data_dict])
        plt.show()
