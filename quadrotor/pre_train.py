import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import chain, pairwise
from math import ceil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from joblib import Parallel, delayed
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_dir))

from env import QuadrotorEnv as Env

from quadrotor.eval import get_controller
from util.defaults import PSDNN_HIDDEN

torch.set_default_dtype(torch.float32)
torch.set_default_device(torch.device("cpu"))
DTYPE = torch.get_default_dtype()
DEVICE = torch.get_default_device()


def simulate_controller_once(
    controller: Callable, n_ep: int, timesteps: int, seeds: Sequence[int]
) -> tuple[npt.NDArray[np.floating], ...]:
    """Simulates one episode of the quadrotor environment using the given controller.

    Parameters
    ----------
    controller : callable
        A controller that takes the current state and the environment as input and
        returns the control input for the next timestep. Also has a `reset` method.
    n_ep : int
        The number of episodes to run for this controller.
    timesteps : int
        The number of timesteps to run each episode for.
    seeds : int
        The seeds for the random number generator for every episode.

    Returns
    -------
    3 float arrays
        Returns the following objects:
         - an array containing the rewards for each episode's timestep
         - an array containing the state trajectories
         - an array containing the obstacles positions and direction for each episode
    """
    # similar to quadrotor/eval.py:simulate_controller_once, but since the controllers
    # are needed multiple times, we don't re-instantiate them every time
    env = Env(timesteps)
    R = np.empty((n_ep, timesteps))
    X = np.empty((n_ep, timesteps, env.ns))
    obstacles = np.empty((n_ep, 2, *env.safety_constraints.size_in(1)))
    for e, s in enumerate(seeds):
        controller.reset()
        x, _ = env.reset(seed=int(s))
        obstacles[e] = env.pos_obs, env.dir_obs
        X[e, 0] = x
        for t in range(timesteps):
            u, _ = controller(x, env)
            x, r, _, _, _ = env.step(u)
            R[e, t] = r
            if t < timesteps - 1:
                X[e, t + 1] = x
    return R, X, obstacles


def generate_dataset(
    controllers: Sequence[Callable],
    n_episodes: int,
    timesteps: int,
    parallel: Parallel,
    seed: int,
) -> TensorDataset:
    """Generates a dataset for supervised learning of predictions of the cost-to-go from
    quadrotor env's context (state + obstacles).

    Parameters
    ----------
    controllers : sequence of callable
        A sequence of controllers to use for generating the data.
    n_ctrl : int
        Number of controllers in parallel to simulate to generate the data.
    n_episodes : int
        Number of episodes to generate for each controller.
    timesteps : int
        The number of timesteps to run each episode for.
    parallel : Parallel
        Parallel object to use for parallel computation.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    TensorDataset
        A dataset containing the context and cost-to-go for each episode and each
        timestep.
    """
    # first, generate the data by playing out the episodes in parallel
    n_ctrl = len(controllers)
    all_seeds = (  # generate all seeds at once to ensure reproducibility
        np.random.SeedSequence(seed)
        .generate_state(n_ctrl * n_episodes)
        .reshape(n_ctrl, -1)
    )
    data = parallel(
        delayed(simulate_controller_once)(controller, n_episodes, timesteps, seeds)
        for controller, seeds in zip(controllers, all_seeds)
    )
    R, X, obstacles = map(np.vstack, zip(*data))

    # compute input context, i.e., state and flattened+repeated obstacle data
    X_ = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
    N = n_ctrl * n_episodes
    obstacles_ = torch.as_tensor(  # reshape in numpy because torch doesn't support F
        obstacles.reshape(N, 2, 1, -1, order="F"), dtype=DTYPE, device=DEVICE
    ).expand(-1, -1, timesteps, -1)
    pos_obs, dir_obs = obstacles_[:, 0], obstacles_[:, 1]
    context = torch.concatenate((X_, pos_obs, dir_obs), 2)

    # and compute output, i.e., cost-to-go for each state (do so in torch to avoid issue
    # with negative strides during numpy-to-torch conversion)
    R_ = torch.as_tensor(R, dtype=DTYPE, device=DEVICE)
    G = R_.fliplr().cumsum(1).fliplr()

    # flatten 1st and 2nd dimensions (episodes and timesteps) into a single dimension
    return TensorDataset(context.reshape(N * timesteps, -1), G.reshape(-1))


class TorchPsdNN(nn.Module):
    """Network that outputs the Cholesky decomposition of a positive semi-definite
    matrix (PSD) for any input. The decomposition is returned as lower triangular.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : sequence of ints
        Number of features in each hidden linear layer.
    out_mat_size : int
        Size of the output PSD square matrix (i.e., its side length).
    act : type of activation function, optional
        Class of the activation function. By default, `ReLU` is used.

    Raises
    ------
    ValueError
        Raises if the number of hidden layers is less than 1.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_mat_size: int,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if len(hidden_features) < 1:
            raise ValueError("Psdnn must have at least one hidden layer")
        super().__init__()
        self.state_batchnorm = nn.BatchNorm1d(in_features, affine=False)
        # create the layers
        features = chain([in_features], hidden_features)
        out_features = (out_mat_size * (out_mat_size + 1)) // 2
        inner_layers = chain.from_iterable(
            (nn.Linear(i, j), act()) for i, j in pairwise(features)
        )
        last_layer = nn.Linear(hidden_features[-1], out_features)
        self.layers = nn.Sequential(*inner_layers, last_layer)
        # for reshaping the output to a lower triangular matrix
        self._mat_size = (out_mat_size, out_mat_size)
        self._tril_idx = torch.tril_indices(out_mat_size, out_mat_size)
        #
        self._xf = torch.as_tensor(Env.xf, dtype=DTYPE, device=DEVICE)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        out_size = x.shape[:-1] + self._mat_size
        out = torch.zeros(out_size, dtype=y.dtype, layout=y.layout, device=y.device)
        out[..., self._tril_idx[0], self._tril_idx[1]] = y
        return out

    def predict_state_value(self, context: torch.Tensor) -> torch.Tensor:
        """Computes the predicted state value for a context (state + obstacles data)."""
        x = context[..., : Env.ns, None]
        dx = x - self._xf
        L = self.forward(context)
        y = L.mT.bmm(dx)
        return y.mT.bmm(y).squeeze((-1, -2))


def train_loop(
    dl: DataLoader, model: TorchPsdNN, loss_fn: Callable, optim: torch.optim.Optimizer
) -> float:
    """Trains the model on the given dataloader.

    Parameters
    ----------
    dl : DataLoader
        Data loader to use for training.
    model : TorchPsdNN
        Model to train.
    loss_fn : Callable
        Loss function to use for training.
    optim : torch.optim.Optimizer
        Optimizer to use for training.

    Returns
    -------
    float
        The average loss over the training dataset.
    """
    model.train()
    train_loss = 0.0
    for context, G in dl:
        pred = model.predict_state_value(context)
        loss = loss_fn(pred, G)
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_loss += loss.item()
    train_loss /= len(dl)
    return train_loss


def eval_loop(dl: DataLoader, model: TorchPsdNN, loss_fn: Callable) -> float:
    """Evaluates the model on the given dataloader.

    Parameters
    ----------
    dl : DataLoader
        Data loader to use for evaluation.
    model : TorchPsdNN
        Model to evaluate.
    loss : Callable
        Loss function to use for evaluation.

    Returns
    -------
    float
        The average loss over the evaluation dataset.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for context, G in dl:
            pred = model.predict_state_value(context)
            test_loss += loss_fn(pred, G).item()
    test_loss /= len(dl)
    return test_loss


if __name__ == "__main__":
    # parse script arguments - similar to eval.py, but we don't allow learning-based
    # controllers, and we add some additional arguments for supervised training of NNs
    default_save = f"pre_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    parser = ArgumentParser(
        description="Evaluation of controllers on the quadrotor environment.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Choice of controller")
    group.add_argument(
        "controller",
        choices=("mpc", "scmpc"),
        help="The controller to use for the simulation.",
    )
    group = parser.add_argument_group("MPC options")
    group.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="The horizon of the MPC controller.",
    )
    group.add_argument(
        "--soft",
        action="store_true",
        help="Whether to use soft constraints in the MPC controller.",
    )
    group = parser.add_argument_group(
        "Scenario MPC (SCMPC) options (used only when `controller=scmpc`)"
    )
    group.add_argument(
        "--scenarios",
        type=int,
        default=32,
        help="The number of scenarios to use in the SCMPC controller.",
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument(
        "--timesteps",
        type=int,
        default=125,
        help="Number of timesteps per each simulation.",
    )
    group = parser.add_argument_group("Supervised learning options")
    group.add_argument(
        "--n-epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )
    group.add_argument(
        "--n-episodes-per-epoch",
        type=int,
        default=20,
        help="Number of training/validation episodes per epoch.",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=2**7,
        help="Batch size (i.e., number of different initial states).",
    )
    group.add_argument(
        "--psdnn-hidden",
        type=int,
        default=PSDNN_HIDDEN,
        nargs=2,
        help="The number of hidden units in the PSDNN terminal cost.",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate of the supervised learning algorithm.",
    )
    group = parser.add_argument_group("Storing options")
    group.add_argument(
        "--save",
        type=str,
        default=default_save,
        help="Saves results with this filename. If not set, a default name is given.",
    )
    group = parser.add_argument_group("Computational options")
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    args = parser.parse_args()
    print(f"Args: {args}\n")

    # prepare arguments to the simulation
    controller = args.controller
    controller_kwargs = {
        "horizon": args.horizon,
        "soft": args.soft,
        "scenarios": args.scenarios,
        # kwargs below are default
        "dcbf": False,
        "use_kappann": False,
        "bound_initial_state": False,
        "terminal_cost": set(),
        "kappann_hidden_size": [],
        "pwqnn_hidden_size": 0,
        "psdnn_hidden_sizes": [],
    }
    ts = args.timesteps
    n_jobs = args.n_jobs
    n_ep_per_job = ceil(args.n_episodes_per_epoch / n_jobs)  # round up
    batch_size = args.batch_size
    seed = args.seed
    save_filename = args.save
    torch.manual_seed(seed)
    logging.basicConfig(
        filename=f"{save_filename}_log.csv",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )

    # build the controllers only once
    controllers = [
        get_controller(controller, **controller_kwargs)[0] for _ in range(n_jobs)
    ]

    # define model, loss function, and optimizer
    mdl = TorchPsdNN(Env.ns + 2 * 3 * Env.n_obstacles, args.psdnn_hidden, Env.ns)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(mdl.parameters(), args.lr, weight_decay=1e-4)

    # start the training loop - for validation we generate a static dataset once at the
    # beginning to better evaluate training convergence - for training a new dataset is
    # generated every epoch on the fly
    n_epochs = args.n_epochs
    logging.info("epoch,train_loss,val_loss")
    with Parallel(n_jobs, verbose=10, return_as="generator_unordered") as parallel:
        valid_dataset = generate_dataset(controllers, n_ep_per_job, ts, parallel, seed)
        valid_dataloader = DataLoader(valid_dataset, batch_size)

        best_eval_loss = float("inf")
        for t in range(n_epochs):
            train_dataset = generate_dataset(
                controllers, n_ep_per_job, ts, parallel, seed + t
            )
            train_dataloader = DataLoader(train_dataset, batch_size)

            train_loss = train_loop(train_dataloader, mdl, loss_fn, optimizer)
            eval_loss = eval_loop(valid_dataloader, mdl, loss_fn)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                checkpoint = {
                    "args": args.__dict__,
                    "epoch": t,
                    "model_state_dict": mdl.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(checkpoint, f"{save_filename}_best.pt")
            if t % (n_epochs // 10) == 0:
                print(f"Epoch {t:>5d}/{n_epochs:>5d}")
            logging.info(f"{t},{train_loss:.15e},{eval_loss:.15e}")
