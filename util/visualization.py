from collections.abc import Collection
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.util.io import load
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import sem


def load_file(
    filename: str,
) -> tuple[dict[str, Any], dict[str, npt.NDArray[np.floating]]]:
    """Loads the data from a single file on disk, separating the simulation arguments
    from the rest of the data.

    Parameters
    ----------
    filename : str
        The filename of the file to load.

    Returns
    -------
    tuple of two dictionaries
        Returns the arguments used to run the simualtion script, as well as the data
        itselft. Both of these are dictionaries.
    """
    data = load(filename)
    args = data.pop("args")
    return args, data


def plot_population(
    ax: Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | tuple[int, ...],
    use_quartiles: bool = False,
    use_sem: bool = False,
    log: bool = False,
    clip_min: float | None = None,
    marker: str | None = None,
    ls: str | None = None,
    color: str | None = None,
    alpha: float = 0.25,
) -> None:
    """Plots a quantity repeated across a population. The mean/median is plotted as a
    line, and the area between the mean ± std or median ± quartiles is filled.

    Parameters
    ----------
    ax : Axes
        The axis to plot on.
    x : array-like of shape `(n,)`
        The x-axis values.
    y : array-like of shape `(..., n, ...)`
        The y-axis values. This contains the values for each member of the population.
        It can have any shape, as long as it contains the `n` dimension. See `axis` in
        order to reduce the population to a single value.
    axis : int | tuple[int, int], optional
        Axis along which to compute the mean/median and std/quartiles. After reduction,
        the shape must be `(n,)` in order to be plotted.
    use_quartiles : bool, optional
        If `True`, the median and quartiles are used instead of the mean and std, by
        default `False`.
    use_sem : bool, optional
        If `True`, the standard error of the mean is computed instead of the std, by
        default `False`.
    log : bool, optional
        If `True`, the y-axis is plotted in log scale, by default `False`.
    clip_min : float, optional
        If given, clips the plot to this minimum value, by default `None`.
    marker : str, optional
        The marker to use in the plot, by default `None`.
    ls : str, optional
        The line style to use in the plot, by default `None
    color : str, optional
        The color to use in the plot, by default `None`.
    alpha : float, optional
        The transparency of the filled area, by default `0.25`.
    """
    assert not (use_quartiles and use_sem), "Cannot plot both quartiles and sem."
    if use_quartiles:
        lower, middle, upper = np.nanquantile(y, [0.25, 0.5, 0.75], axis)
    else:
        middle = np.nanmean(y, axis)
        std = np.nanstd(y, axis) if not use_sem else sem(y, axis, nan_policy="omit")
        lower = middle - std
        upper = middle + std
    if clip_min is not None:
        lower = np.maximum(lower, clip_min)
    method = ax.semilogy if log else ax.plot
    c = method(x, middle, label=None, marker=marker, ls=ls, color=color)[0].get_color()
    if alpha > 0:
        ax.fill_between(x, lower, upper, alpha=alpha, color=c, label=None)


def plot_single_violin(
    ax: Axes,
    position: float,
    data: npt.ArrayLike,
    scatter: bool = True,
    color: str | None = None,
    alpha: float = 0.25,
    vert: bool = True,
    side: Literal["both", "low", "high"] = "both",
    gap: float = 0.0,
    **other_violin_kwargs: Any,
):
    """Plots a single violin plot.

    Parameters
    ----------
    ax : Axes
        The axis to plot on.
    position : float
        The position of the violin plot on the x-axis, if `vert`, or on the y-axis.
    data : 1d array-like
        The data to plot.
    color : str, optional
        The color of the scattered and violin data, by default `None`.
    alpha : float, optional
        The transparency of the violin plot, by default `0.25`.
    vert : bool, optional
        If `True`, the violin plot is vertical, by default `True`.
    side : {"both", "low", "high"}, optional
        The side of the violin plot to plot, by default `"both"`.
    gap : float, optional
        Positive gap between sided violin plots, by default `0.0`.
    violin_kwargs : dict
        Other keywords arguments to pass to the violin plot.
    """
    data = np.asarray(data)
    if side == "low":
        position -= gap
    elif side == "high":
        position += gap
    violin_data = ax.violinplot(
        data, [position], vert, side=side, **other_violin_kwargs
    )

    # plot scattered data
    if scatter:
        if side == "low":
            lb, ub = -0.01, 0.0
        elif side == "high":
            lb, ub = 0.0, 0.01
        else:
            lb, ub = -0.01, 0.01
        positions = position + np.random.uniform(lb, ub, size=data.size)
        ax.scatter(positions, data, s=10, facecolor="none", edgecolors=color)

    # the rest of the method adjustes the length of the violin lines to match the
    # distribution and adjust the color
    bodies = violin_data["bodies"]
    assert len(bodies) == 1, "only one violin plot is supported! Something went wrong."
    body = bodies[0]
    body.set_facecolor(color)
    body.set_alpha(alpha)
    if color:
        for k in ("cmeans", "cmedians", "cquantiles"):
            if k in violin_data:
                linecollection = violin_data[k]
                linecollection.set_color(color)

    # violin_vertices = body.get_paths()[0].vertices
    # N = (violin_vertices.shape[0] - 1) // 2
    # if side in ("low", "both"):
    #     X = violin_vertices[1:N, 0]
    #     Y = violin_vertices[1:N, 1]
    # else:
    #     X = violin_vertices[-2 : -N - 1 : -1, 0]
    #     Y = violin_vertices[-2 : -N - 1 : -1, 1]
    # if vert:
    #     X, Y = Y, X

    # from scipy.interpolate import CubicSpline

    # interp = CubicSpline(X, Y)
    # int_vert = int(vert)
    # int_not_vert = int(not vert)

    # for k in ("cmeans", "cmedians", "cquantiles"):
    #     if k not in violin_data:
    #         continue

    #     linecollection = violin_data[k]
    #     if color:
    #         linecollection.set_color(color)

    #     for line in linecollection.get_paths():
    #         value = line.vertices[0, int_vert]
    #         vertex = interp(value)
    #         if side == "low":
    #             line.vertices[0, int_not_vert] = vertex
    #         elif side == "high":
    #             line.vertices[1, int_not_vert] = vertex
    #         else:
    #             line.vertices[:, int_not_vert] = [vertex, 2 * position - vertex]


def plot_cylinder(
    ax: Axes3D,
    radius: float,
    height: float,
    center: npt.NDArray[np.floating],
    direction: npt.NDArray[np.floating],
    resolution: int = 100,
    color: str | None = None,
    alpha: float | None = None,
) -> None:
    """Plots a cylinder in 3D as a surface.

    Parameters
    ----------
    ax : Axes3D
        The axis to plot on.
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder.
    center : array of 3 floats
        The center of the cylinder.
    direction : array of 3 floats
        The direction of the cylinder axis.
    resolution : int, optional
        The resolution of the cylinder, by default `100`.
    color : str | None, optional
        The color of the cylinder, by default `None`.
    alpha : float | None, optional
        The transparency of the cylinder, by default `None`.
    """
    # generate cylinder in local coordinates
    theta = np.linspace(0, 2 * np.pi, resolution)  # angular coordinates
    z = np.linspace(0, height, resolution)  # height coordinates
    theta, z = np.meshgrid(theta, z)  # meshgrid for cylinder surface
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = z

    # normalize direction vector
    direction = direction / np.linalg.norm(direction)

    # find two perpendicular vectors to direction
    if direction[0] == 0 and direction[1] == 0:  # Special case for vertical direction
        perp1 = np.asarray([1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 0, 1])
        perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    # Create the rotation matrix
    rotation_matrix = np.asarray([perp1, perp2, direction]).T

    # Rotate and translate the cylinder, and reshape to original grid shape
    x_flat, y_flat, z_flat = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    cylinder_coords = np.dot(rotation_matrix, np.asarray([x_flat, y_flat, z_flat]))
    x_global = (cylinder_coords[0, :] + center[0]).reshape(x.shape)
    y_global = (cylinder_coords[1, :] + center[1]).reshape(y.shape)
    z_global = (cylinder_coords[2, :] + center[2]).reshape(z.shape)

    # plot
    ax.plot_surface(x_global, y_global, z_global, color=color, alpha=alpha)


def plot_returns(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    names: Collection[str] | None = None,
) -> None:
    """Plots the returns of the simulations (normalized by the length of each
    episode).

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the keys `"cost"`
        and `"states"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    _, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

    for i, datum in enumerate(data):
        c = f"C{i}"
        timesteps = datum["states"].shape[2] - 1
        returns = datum["cost"] / timesteps  # n_agents x n_ep
        episodes = np.arange(returns.shape[1])
        # print(f"cost: {np.mean(returns)} ± {np.std(returns)} | {np.median(returns)}")
        # in the first axis, flatten the first two axes as we do not distinguish between
        # different agents
        returns_flat = returns.reshape(-1)
        plot_single_violin(
            ax1,
            i,
            returns_flat,
            showextrema=False,
            quantiles=[0.25, 0.5, 0.75],
            color=c,
        )
        plot_population(ax2, episodes, returns, axis=0, color=c, clip_min=0)

    ax1.set_xticks(range(len(data)))
    if names is not None:
        ax1.set_xticklabels(names)
    ax1.set_ylabel("Return (normalized)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Return (normalized)")
    # ax1.set_yscale("log")
    # ax2.set_yscale("log")


def plot_solver_times(
    data: Collection[dict[str, npt.NDArray[np.floating]]],
    names: Collection[str] | None = None,
) -> None:
    """Plots the solver times of the simulations. For agents with a solver for the
    state and action value functions, their computation times are plotted separately.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each containing the key
        `"sol_times"`.
    names : collection of str, optional
        The names of the simulations to use in the plot.
    """
    _, ax = plt.subplots(1, 1, constrained_layout=True)

    for i, datum in enumerate(data):
        c = f"C{i}"
        sol_times = datum["sol_times"]  # n_agents x n_ep x timestep (x 2, optional)
        kw = {"showextrema": False, "quantiles": [0.25, 0.5, 0.75], "color": c}

        # flatten the first three axes as we do not distinguish between different
        # agents, episodes or time steps
        sol_times_flat = sol_times.reshape(-1, *sol_times.shape[3:])
        # print(f"sol. time: {np.mean(st):e} ± {np.std(st):e} | {np.median(st):e}")
        if sol_times_flat.ndim == 1:
            plot_single_violin(ax, i, sol_times_flat, **kw)
        else:
            assert sol_times_flat.ndim == 2, "Unexpected shape of `sol_times_flat`"
            sol_times_V, sol_times_Q = sol_times_flat.T
            plot_single_violin(ax, i, sol_times_V, side="low", **kw)
            plot_single_violin(ax, i, sol_times_Q, side="high", **kw)

    ax.set_xticks(range(len(data)))
    if names is not None:
        ax.set_xticklabels(names)
    ax.set_ylabel("Solver time [s]")
    ax.set_yscale("log")


def plot_training(
    data: Collection[dict[str, npt.NDArray[np.floating]]], *_: Any, **__: Any
) -> None:
    """Plots the training results of the simulations. This plot does not show anything
    if the data does not contain training results.

    Parameters
    ----------
    data : collection of dictionaries (str, arrays)
        The dictionaries from different simulations, each potentially containing the
        keys `"updates_history"`, `"td_errors"`, or `"policy_performances"`, or
        `"evals"`.
    """
    param_names = set()
    for datum in data:
        if "updates_history" in datum:
            param_names.update(datum["updates_history"].keys())
    n_params = len(param_names)
    any_td = any("td_errors" in d for d in data)
    any_pg = any("policy_gradients" in d for d in data)
    any_eval = any("evals" in d for d in data)
    if n_params == 0 and not any_td and not any_pg and not any_eval:
        return

    fig = plt.figure(constrained_layout=True)
    ncols = max(1, int(np.round(np.sqrt(n_params))))
    nrows = int(np.ceil(n_params / ncols))
    gs = GridSpec(int(any_td or any_pg) + int(any_eval) + nrows, ncols, fig)
    offset = 0
    if any_td or any_pg:
        if any_td and any_pg:
            ax_td = fig.add_subplot(gs[offset, :])
            ax_pg = ax_td.twinx()
        elif any_td:
            ax_td = fig.add_subplot(gs[offset, :])
        else:
            ax_pg = fig.add_subplot(gs[offset, :])
        offset += 1
    if any_eval:
        ax_eval = fig.add_subplot(gs[offset, :])
        offset += 1
    if n_params > 0:
        param_names = sorted(param_names)
        start = nrows * ncols - n_params
        ax_par_first = fig.add_subplot(gs[start // ncols + offset, start % ncols])
        ax_pars = {param_names[0]: ax_par_first}
        for i, name in enumerate(param_names[1:], start=start + 1):
            ax_pars[name] = fig.add_subplot(
                gs[i // ncols + offset, i % ncols], sharex=ax_par_first
            )

    for i, datum in enumerate(data):
        c = f"C{i}"

        if "td_errors" in datum:
            td_errors = datum["td_errors"]  # n_agents x n_ep x timestep
            td_errors = np.nansum(np.square(td_errors), 2)
            episodes = np.arange(td_errors.shape[1])
            plot_population(ax_td, episodes, td_errors, axis=0, color=c)
            # nans = np.isnan(td_errors).sum(1)
            # timesteps = td_errors.shape[2]
            # print(f"NaN TD errors: {np.mean(nans)} +/- {np.std(nans)} / {timesteps}")
        elif "policy_gradients" in datum:
            n_ep = datum["cost"].shape[1]
            grads = datum["policy_gradients"]  # n_agents x n_up x n_pars
            norms = np.linalg.norm(grads, axis=2)
            updates = np.linspace(0, n_ep - 1, grads.shape[1])
            plot_population(ax_pg, updates, norms, axis=0, color=c, log=True)

        if "evals" in datum:
            eval_returns = datum["evals"]  # n_agents x n_evals x n_eval_ep
            n_ep = datum["cost"].shape[1]  # n_agents x n_ep
            episodes = np.linspace(0, n_ep - 1, eval_returns.shape[1])
            plot_population(ax_eval, episodes, eval_returns.mean(2), 0, use_sem=True)

        if "updates_history" in datum:
            for name, param in datum["updates_history"].items():
                updates = np.arange(param.shape[1])
                param = param.reshape(*param.shape[:2], -1)  # n_agents x n_up x ...
                ax = ax_pars[name]
                n_params = param.shape[2]
                alpha = 0.25 if n_params == 1 else 0
                for idx in range(param.shape[2]):
                    plot_population(
                        ax, updates, param[..., idx], axis=0, color=c, alpha=alpha
                    )

    if any_td:
        ax_td.set_xlabel("Episode")
        ax_td.set_ylabel(r"$\sum{\delta^2}$")
    if any_pg:
        ax_pg.set_xlabel("Episode")
        ax_pg.set_ylabel(r"$\| \nabla_\theta J(\pi_\theta) \|_2$")
    if any_eval:
        ax_eval.set_xlabel("Episode")
        ax_eval.set_ylabel("Eval. Return")
    if n_params > 0:
        for name, ax in ax_pars.items():
            ax.set_xlabel("Episode")
            ax.set_ylabel(name)
            ax._label_outer_xaxis(skip_non_rectangular_axes=False)
