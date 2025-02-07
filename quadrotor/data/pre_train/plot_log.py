import sys

import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python script.py <argument1> <argument2> ...")
    sys.exit(1)

filenames = sys.argv[1:]

epochs = []
train_losses = []
eval_losses = []
for fn in filenames:
    epochs_, train_losses_, eval_losses_ = np.loadtxt(fn, delimiter=",", skiprows=1).T
    epochs.append(epochs_)
    train_losses.append(train_losses_)
    eval_losses.append(eval_losses_)

ncols = int(np.round(np.sqrt(len(filenames))))
nrows = int(np.ceil(len(filenames) / ncols))
_, axs = plt.subplots(ncols, nrows, sharex=True, sharey=True, constrained_layout=True)
axs = np.atleast_1d(axs).flatten()

for ax, fn, epochs_, train_losses_, eval_losses_ in zip(
    axs, filenames, epochs, train_losses, eval_losses
):
    ax.semilogy(epochs_, train_losses_, label="train")
    ax.semilogy(epochs_, eval_losses_, label="eval")

    min_idx = np.argmin(eval_losses_)
    ax.semilogy(epochs_[min_idx], eval_losses_[min_idx], "ro")

    ax.set_title(fn)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax._label_outer_xaxis(skip_non_rectangular_axes=False)
    ax._label_outer_yaxis(skip_non_rectangular_axes=False)
    ax.legend()

for i in range(len(filenames), len(axs)):
    axs[i].set_axis_off()

plt.show()
