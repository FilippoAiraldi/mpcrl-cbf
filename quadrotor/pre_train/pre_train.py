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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torcheval.metrics import MeanSquaredError, R2Score

repo_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_dir))

from quadrotor.env import QuadrotorEnv as Env
from util.defaults import PSDNN_HIDDEN

DTYPE = torch.float64
DEVICE = torch.device("cpu")
AS_TENSOR = partial(torch.as_tensor, dtype=DTYPE, device=DEVICE)


class TorchPsdNN(nn.Module):
    """Network that predicts a terminal cost as a quadratic form. The quadratic matrix
    is a positive semi-definite matrix (PSD) and is output as a Cholesky decomposition.
    Also the reference point is learnable.

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
        eps: float = 1e-2,
    ) -> None:
        super().__init__()
        if len(hidden_features) < 1:
            raise ValueError("Psdnn must have at least one hidden layer")
        super().__init__()
        self.context_norm = nn.BatchNorm1d(in_features, affine=False)
        features = chain([in_features], hidden_features)
        self.hidden_layers = nn.Sequential(
            *chain.from_iterable(
                (nn.Linear(i, j), act()) for i, j in pairwise(features)
            )
        )
        self.mat_head = nn.Linear(hidden_features[-1], (out_size * (out_size + 1)) // 2)
        self.ref_head = nn.Linear(hidden_features[-1], out_size)
        self._mat_size = (out_size, out_size)
        self._tril_idx = torch.tril_indices(out_size, out_size)
        self._xf = AS_TENSOR(Env.xf)
        self._eps = eps

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden_layers(self.context_norm(context))

        ref = self.ref_head(h)

        elems = self.mat_head(h)
        mat_size = context.shape[:-1] + self._mat_size
        mat = torch.zeros(
            mat_size, dtype=elems.dtype, layout=elems.layout, device=elems.device
        )
        mat[..., self._tril_idx[0], self._tril_idx[1]] = elems
        return mat, ref

    def predict_state_value(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        pos_obs: torch.Tensor,
        dir_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the predicted state value for a context (state + prev action +
        obstacles data)."""
        batches = torch.broadcast_shapes(
            state.shape[:-1],
            prev_action.shape[:-1],
            pos_obs.shape[:-1],
            dir_obs.shape[:-1],
        )
        x = torch.broadcast_to(state, batches + state.shape[-1:])
        up = torch.broadcast_to(prev_action, batches + prev_action.shape[-1:])
        po = torch.broadcast_to(pos_obs, batches + pos_obs.shape[-1:])
        do = torch.broadcast_to(dir_obs, batches + dir_obs.shape[-1:])
        context = torch.cat((x, up, po, do), dim=-1)

        L, ref = self.forward(context)
        mat = L.bmm(L.mT)
        mat.diagonal(dim1=-1, dim2=-2).add_(self._eps)
        dx = (x - self._xf - ref).unsqueeze(-1)
        return dx.mT.bmm(mat.bmm(dx)).squeeze((-1, -2))


def train(
    dl: DataLoader, model: TorchPsdNN, loss_fn: Callable, optim: torch.optim.Optimizer
) -> tuple[float, float, float]:
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
    3 floats
        Loss function, RMSE and R2 score on the training dataset.
    """
    tot_loss = 0.0
    mse = MeanSquaredError(device=DEVICE)
    r2score = R2Score(device=DEVICE)

    model.train()
    for x, u_prev, pos_obs, dir_obs, G in dl:
        pred = model.predict_state_value(x, u_prev, pos_obs, dir_obs)
        loss = loss_fn(pred, G)
        loss.backward()
        optim.step()
        optim.zero_grad()

        tot_loss += loss.item()
        mse.update(pred, G)
        r2score.update(pred, G)

    return tot_loss / len(dl), mse.compute().sqrt().item(), r2score.compute().item()


def test(
    dl: DataLoader, model: TorchPsdNN, loss_fn: Callable
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
        for x, u_prev, pos_obs, dir_obs, G in dl:
            pred = model.predict_state_value(x, u_prev, pos_obs, dir_obs)
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
        "--psdnn-hidden",
        type=int,
        default=PSDNN_HIDDEN,
        nargs="+",
        help="The number of hidden units per layer in the PSDNN terminal cost.",
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

    # load the dataset
    dataset = load(args.dataset)
    cost_to_go = AS_TENSOR(dataset["cost_to_go"])
    cost_to_go_ptp = (cost_to_go.max() - cost_to_go.min()).item()
    states = AS_TENSOR(dataset["states"])
    prev_actions = AS_TENSOR(dataset["previous_actions"])
    obstacles = AS_TENSOR(dataset["obstacles"])

    # expand the obstacle data to be broadcastable with the rest
    n_ep, timesteps = states.shape[:2]
    obstacles_ = obstacles.unsqueeze(1).expand(-1, timesteps, -1, -1)
    pos_obstacles, dir_obstacles = obstacles_[..., 0, :], obstacles_[..., 1, :]

    # split the dataset episode indices into training, evaluation, and testing
    indices = torch.arange(n_ep)
    train_dl, eval_dl, test_dl = (
        DataLoader(
            TensorDataset(
                states[idx].reshape(-1, states.shape[-1]),
                prev_actions[idx].reshape(-1, prev_actions.shape[-1]),
                pos_obstacles[idx].reshape(-1, pos_obstacles.shape[-1]),
                dir_obstacles[idx].reshape(-1, dir_obstacles.shape[-1]),
                cost_to_go[idx].reshape(-1),
            ),
            batch_size,
        )
        for idx in random_split(indices, (0.6, 0.2, 0.2))
    )

    # define model, loss function, and optimizer
    context_size = Env.ns + Env.na + 2 * 3 * Env.n_obstacles
    mdl = TorchPsdNN(context_size, args.psdnn_hidden, Env.ns).to(DEVICE, DTYPE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(mdl.parameters(), args.lr, weight_decay=1e-4)

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
