import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from itertools import chain, pairwise
from pathlib import Path
from time import perf_counter

import torch
from csnlp.util.io import load
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torcheval.metrics import MeanSquaredError, R2Score

repo_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_dir))

from quadrotor.env import QuadrotorEnv as Env
from util.defaults import QUADROTOR_NN_HIDDEN

DTYPE = torch.float64
DEVICE = torch.device("cpu")
AS_TENSOR = partial(torch.as_tensor, dtype=DTYPE, device=DEVICE)


def load_dataset(
    filename: str,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, Tensor], float]:
    """Loads the dataset and splits it into training, evaluation, and testing datasets.
    Also normalizes the input data."""
    # load the dataset
    dataset = load(filename)
    cost_to_go = AS_TENSOR(dataset["cost_to_go"])
    cost_to_go_ptp = (cost_to_go.max() - cost_to_go.min()).item()
    states = AS_TENSOR(dataset["states"])
    prev_actions = AS_TENSOR(dataset["previous_actions"])

    # split the dataset indices into training, evaluation, and testing
    rates = (0.6, 0.2, 0.2)
    train_idx, eval_idx, test_idx = random_split(torch.arange(states.shape[0]), rates)

    # add distance to obstacles to the dataset
    pos_ = states[..., :3]
    diff_pos_ = pos_ - AS_TENSOR(Env.pos_obs)
    dir_obs = AS_TENSOR(Env.dir_obs).view(1, 1, -1)
    dist = (
        torch.linalg.cross(diff_pos_, dir_obs).square().sum(-1, keepdims=True)
        - (Env.radius_obs + Env.radius_quadrotor) ** 2
    )

    # normalize input data via the training means and stds
    x_std, x_mean = torch.std_mean(states[train_idx], (0, 1))
    up_std, up_mean = torch.std_mean(prev_actions[train_idx], (0, 1))
    dist_std, dist_mean = dist.amax((0, 1)) - dist.amin((0, 1)), torch.zeros(
        (1,), dtype=DTYPE, device=DEVICE  # preserve sign
    )
    normalizations = {
        "state_mean": x_mean.cpu(),
        "state_std": x_std.cpu(),
        "action_mean": up_mean.cpu(),
        "action_std": up_std.cpu(),
        "dist_mean": dist_mean.cpu(),
        "dist_std": dist_std.cpu(),
    }
    norm_states = (states - x_mean) / x_std
    norm_prev_actions = (prev_actions - up_mean) / up_std
    norm_dist = (dist - dist_mean) / dist_std

    # add the previous state to the dataset
    norm_prev_states = torch.cat((norm_states[:, 0, None], norm_states[:, :-1]), dim=1)

    # split the normalized data into training, evaluation, and testing
    train_ds, eval_ds, test_ds = (
        TensorDataset(
            norm_states[idx].reshape(-1, states.shape[-1]),
            norm_prev_states[idx].reshape(-1, states.shape[-1]),
            norm_prev_actions[idx].reshape(-1, prev_actions.shape[-1]),
            norm_dist[idx].reshape(-1, norm_dist.shape[-1]),
            cost_to_go[idx].reshape(-1),
        )
        for idx in (train_idx, eval_idx, test_idx)
    )

    return train_ds, eval_ds, test_ds, normalizations, cost_to_go_ptp


class QuadrotorNN(nn.Module):
    """PyTorch implementation of the `QuadrotorEnv` class.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : sequence of ints
        Number of features in each hidden linear layer.
    out_size : int
        Size of the output  matrix (i.e., the size of the state space).
    act : type of activation function, optional
        Class of the activation function. By default, `ReLU` is used.
    eps : float, optional
        Regularization term to ensure the matrix is positive semi-definite. Set it to
        `<= 0.0` to disable regularization. By default, `1e-4`.

    Raises
    ------
    ValueError
        Raises if the number of hidden layers is less than 1.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_size: int,
        act: type[nn.Module] = nn.ReLU,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if len(hidden_features) < 1:
            raise ValueError("The network must have at least one hidden layer")
        super().__init__()
        features = chain([in_features], hidden_features)
        self.hidden_layers = nn.Sequential(
            *chain.from_iterable(
                (nn.Linear(i, j), act()) for i, j in pairwise(features)
            )
        )
        self.mat_head = nn.Linear(hidden_features[-1], (out_size * (out_size + 1)) // 2)
        self.ref_head = nn.Linear(hidden_features[-1], out_size)
        self.cbf_head = nn.Linear(hidden_features[-1], 1)  # NOTE: does not get trained
        self._mat_size = (out_size, out_size)
        self._tril_idx = torch.tril_indices(out_size, out_size)
        self._xf = AS_TENSOR(Env.xf)
        self._eps = eps

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        h = self.hidden_layers(context)

        ref = self.ref_head(h)

        elems = self.mat_head(h)
        mat_size = context.shape[:-1] + self._mat_size
        mat = torch.zeros(
            mat_size, dtype=elems.dtype, layout=elems.layout, device=elems.device
        )
        mat[..., self._tril_idx[0], self._tril_idx[1]] = elems
        return mat, ref

    def predict_state_value(
        self, state: Tensor, prev_state: Tensor, prev_action: Tensor, dist: Tensor
    ) -> Tensor:
        """Predicts the value of the given state and context (prev state + prev action +
        distance to obstacles)."""
        batches = torch.broadcast_shapes(
            state.shape[:-1],
            prev_state.shape[:-1],
            prev_action.shape[:-1],
            dist.shape[:-1],
        )
        x = torch.broadcast_to(state, batches + state.shape[-1:])
        xp = torch.broadcast_to(prev_state, batches + prev_state.shape[-1:])
        up = torch.broadcast_to(prev_action, batches + prev_action.shape[-1:])
        d = torch.broadcast_to(dist, batches + dist.shape[-1:])
        context = torch.cat((xp, up, d), dim=-1)

        L, ref = self.forward(context)
        dx = (x - self._xf - ref).unsqueeze(-1)
        y = L.mT.bmm(dx)
        out = y.mT.bmm(y).squeeze((-1, -2))
        if self._eps > 0.0:
            out += self._eps * dx.mT.bmm(dx).squeeze((-1, -2))
        return out


def train(
    dl: DataLoader, model: QuadrotorNN, loss_fn: Callable, optim: torch.optim.Optimizer
) -> tuple[float, float, float]:
    """Trains the model on the given dataloader.

    Parameters
    ----------
    dl : DataLoader
        Data loader to use for training.
    model : TorchQuadrotorNN
        Model to train.
    loss_fn : Callable
        Loss function to use for training.
    optim : torch.optim.Optimizer
        Optimizer to use for training.

    Returns
    -------
    3 floats
        Loss function, RMSE and R2 score on the training dataset.
    """
    tot_loss = 0.0
    mse = MeanSquaredError(device=DEVICE)
    r2score = R2Score(device=DEVICE)

    model.train()
    for x, x_prev, u_prev, dist, G in dl:
        pred = model.predict_state_value(x, x_prev, u_prev, dist)
        loss = loss_fn(pred, G)
        loss.backward()
        optim.step()
        optim.zero_grad()

        tot_loss += loss.item()
        mse.update(pred, G)
        r2score.update(pred, G)

    return tot_loss / len(dl), mse.compute().sqrt().item(), r2score.compute().item()


def test(
    dl: DataLoader, model: QuadrotorNN, loss_fn: Callable
) -> tuple[float, float, float]:
    """Tests the model on the given dataloader.

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
    3 floats
        Loss function, RMSE and R2 score on the evluation dataset.
    """
    tot_loss = 0.0
    mse = MeanSquaredError(device=DEVICE)
    r2score = R2Score(device=DEVICE)

    model.eval()
    with torch.no_grad():
        for x, x_prev, u_prev, dist, G in dl:
            pred = model.predict_state_value(x, x_prev, u_prev, dist)
            tot_loss += loss_fn(pred, G).item()
            mse.update(pred, G)
            r2score.update(pred, G)

    return tot_loss / len(dl), mse.compute().sqrt().item(), r2score.compute().item()


if __name__ == "__main__":
    # parse script arguments - similar to eval.py, but we don't allow learning-based
    # controllers, and we add some additional arguments for supervised training of NNs
    default_save = f"pre_train_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    parser = ArgumentParser(
        description="Pre-training of terminal cost-to-go approximation.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Choice of controller")
    group.add_argument(
        "dataset",
        type=str,
        help="Filename of the dataset to be used for pre-training.",
    )
    group = parser.add_argument_group("Supervised learning options")
    group.add_argument(
        "--n-epochs",
        type=int,
        default=2000,
        help="Number of pre-training epochs.",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=2**7,
        help="Batch size.",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate of the supervised learning algorithm.",
    )
    group = parser.add_argument_group("Neural topology options")
    group.add_argument(
        "--nn-hidden",
        type=int,
        default=QUADROTOR_NN_HIDDEN,
        nargs="+",
        help="The number of hidden units per layer in the NN.",
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
    args = parser.parse_args()
    print(f"Args: {args}\n")

    # prepare arguments to the simulation
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    save_filename = args.save
    torch.manual_seed(args.seed)

    # load the dataset and get the dataloaders
    train_ds, eval_ds, test_ds, norm, cost_to_go_ptp = load_dataset(args.dataset)
    train_dl = DataLoader(train_ds, batch_size, True)
    eval_dl = DataLoader(eval_ds, batch_size)
    test_dl = DataLoader(test_ds, batch_size)

    # define model, loss function, and optimizer
    context_size = Env.ns + Env.na + 1
    mdl = QuadrotorNN(context_size, args.nn_hidden, Env.ns).to(DEVICE, DTYPE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(mdl.parameters(), args.lr, weight_decay=1e-3)

    # train the model
    logging.basicConfig(
        filename=f"{save_filename}_log.csv",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    logging.info("epoch,train_loss,train_nrmse,train_r2,eval_loss,eval_nrmse,eval_r2")
    best_eval_loss = float("inf")
    start = perf_counter()
    for t in range(n_epochs):
        t_loss, t_rmse, t_r2 = train(train_dl, mdl, loss_fn, optimizer)
        e_loss, e_rmse, e_r2 = test(eval_dl, mdl, loss_fn)
        t_rmse /= cost_to_go_ptp
        e_rmse /= cost_to_go_ptp

        if e_loss < best_eval_loss:
            best_eval_loss = e_loss
            checkpoint = {
                "args": args.__dict__,
                "epoch": t,
                "model_state_dict": mdl.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalization": norm,
            }
            torch.save(checkpoint, f"{save_filename}_best.pt")
            msg = ", better model found!"
        else:
            msg = ""

        elapsed_time = perf_counter() - start
        print(f"Epoch: {t:>4d}/{n_epochs:>4d}, elapsed time: {elapsed_time:.2f}s" + msg)
        logging.info(f"{t},{t_loss},{t_rmse},{t_r2},{e_loss},{e_rmse},{e_r2}")

    # test the best model
    checkpoint = torch.load(f"{save_filename}_best.pt")
    mdl.load_state_dict(checkpoint["model_state_dict"])
    t_loss, t_rmse, t_r2 = test(test_dl, mdl, loss_fn)
    t_rmse /= cost_to_go_ptp
    logging.info(f"-1,{t_loss},{t_rmse},{t_r2},nan,nan,nan")  # special csv row
