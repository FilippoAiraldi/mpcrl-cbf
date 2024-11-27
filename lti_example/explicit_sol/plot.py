"""Plots the optimal infinite-horizon value function and policy for the constrained LTI
environment."""

import argparse
from collections.abc import Collection

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm, colors


def do_plot(data: Collection[dict[str, npt.NDArray[np.floating]]], log: bool) -> None:
    """Plots the optimal infinite-horizon value function and policy for the constrained
    LTI environment.

    Parameters
    ----------
    data : collection of dicts of arrays
        The data to plot. Each dictionary must contain the keys "grid", "V", and "U".
    log : bool
        Whether to plot the value function in log scale.
    """
    # compute the ranges of the value function and policy
    vf_min, vf_max = np.inf, -np.inf
    pol_min, pol_max = np.inf, -np.inf
    for datum in data:
        value_func = datum["V"]
        policy = datum["U"]
        vf_min = min(vf_min, np.nanmin(value_func))
        vf_max = max(vf_max, np.nanmax(value_func))
        pol_min = min(pol_min, np.nanmin(policy))
        pol_max = max(pol_max, np.nanmax(policy))

    # create the figure and kwargs
    N = len(data)
    fig = plt.figure(constrained_layout=True)
    cmap = "RdBu_r"
    if not log:
        vf_norm = colors.Normalize(vmin=vf_min, vmax=vf_max)
    else:
        vf_norm = colors.LogNorm(vmin=vf_min, vmax=vf_max)
    pol_norm = colors.Normalize(vmin=pol_min, vmax=pol_max)
    vf_kwargs = {"norm": vf_norm, "cmap": cmap}
    pol_kwargs = {"norm": pol_norm, "cmap": cmap}

    # loop over the data and plot
    idx = 1
    for i, datum in enumerate(data):
        ax1 = fig.add_subplot(N, 4, idx, projection="3d")
        ax2 = fig.add_subplot(N, 4, idx + 1)
        ax3 = fig.add_subplot(N, 4, idx + 2, projection="3d")
        ax4 = fig.add_subplot(N, 4, idx + 3, projection="3d")
        idx += 4
        grid = datum["grid"]
        V = datum["V"]
        U = datum["U"]

        X1, X2 = np.meshgrid(grid, grid)
        ax1.plot_surface(X1, X2, V, **vf_kwargs)
        ax2.contourf(X1, X2, V, **vf_kwargs)
        ax3.plot_surface(X1, X2, U[..., 0], **pol_kwargs)
        ax4.plot_surface(X1, X2, U[..., 1], **pol_kwargs)

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
        ax2.set_aspect("equal")
        ax1.set_zlabel("$V(x)$")
        ax3.set_zlabel("$u_1(x)$")
        ax4.set_zlabel("$u_2(x)$")
        if i == 0:
            ax1.set_title("Value Function")
            ax2.set_title("Value Function (top)")
            ax3.set_title("Policy 1")
            ax4.set_title("Policy 2")

    for norm, axs in ((vf_norm, (ax1, ax2)), (pol_norm, (ax3, ax4))):
        fig.colorbar(cm.ScalarMappable(norm, cmap), ax=axs, orientation="horizontal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots optimal value function and policy for the constrained LTI "
        "environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Explicit solutions to plot")
    group.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Filenames containing the explicit solutions (.npz).",
    )
    group = parser.add_argument_group("Other options")
    group.add_argument("--log", action="store_true", help="Plots in log scale.")
    args = parser.parse_args()
    data = list(map(np.load, args.filenames))
    do_plot(data, args.log)
    plt.show()
