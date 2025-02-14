"""Plots the progress of pre-training sessions from the corresponding CSV log files."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def do_plotting(filenames: Sequence[str], log: bool) -> None:
    """Reads the CSV files and performs plotting."""
    data = [np.loadtxt(fn, delimiter=",", skiprows=1) for fn in filenames]

    nrows = len(data)
    _, axs_all = plt.subplots(nrows, 3, sharex=True, constrained_layout=True)
    axs_all = np.atleast_2d(axs_all)
    for i, (fn, datum, axs) in enumerate(zip(filenames, data, axs_all)):
        # if the last row contains an epoch == -1, it reports the testing results!
        if datum[-1, 0] == -1:
            test_loss, test_nrmse, test_r2 = datum[-1, 1:4]
            print(
                f"{fn}: loss={test_loss:.6f}, NRMSE={test_nrmse:.6f}, R2={test_r2:.6f}"
            )
            datum = datum[:-1]  # remove that row before the rest of the script
        epochs, train_loss, train_nrmse, train_r2, val_loss, val_nrmse, val_r2 = datum.T

        min_idx = np.argmin(val_loss)

        method = getattr(axs[0], "semilogy" if log else "plot")
        method(epochs, train_loss, label="train")
        method(epochs, val_loss, label="eval")
        method(epochs[min_idx], val_loss[min_idx], "ro", label="best")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        if i == 0:
            axs[0].legend()

        method = getattr(axs[1], "semilogy" if log else "plot")
        method(epochs, train_nrmse)
        method(epochs, val_nrmse)
        method(epochs[min_idx], val_nrmse[min_idx], "ro")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("NRMSE")
        axs[1].set_title(rf"$\bf{{{fn.replace("_", r"\_")}}}$")

        axs[2].plot(epochs, train_r2)
        axs[2].plot(epochs, val_r2)
        axs[2].plot(epochs[min_idx], val_r2[min_idx], "ro")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("$R^2$")

    for ax in axs_all.flat:
        ax._label_outer_xaxis(skip_non_rectangular_axes=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Plots pre-training CSV log files.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filenames", nargs="+", type=str, help="CSV log files to plot")
    parser.add_argument(
        "--log", action="store_true", help="Losses are reported in log values."
    )
    args = parser.parse_args()
    do_plotting(set(args.filenames), args.log)
    plt.show()
