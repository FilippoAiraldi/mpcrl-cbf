from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.interpolate import CubicSpline


def plot_population(
    ax: Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    axis: int | tuple[int, ...],
    use_quartiles: bool = False,
    log: bool = False,
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
    log : bool, optional
        If `True`, the y-axis is plotted in log scale, by default `False`.
    marker : str, optional
        The marker to use in the plot, by default `None`.
    ls : str, optional
        The line style to use in the plot, by default `None
    color : str, optional
        The color to use in the plot, by default `None`.
    alpha : float, optional
        The transparency of the filled area, by default `0.25`.
    """
    if use_quartiles:
        lower, middle, upper = np.nanquantile(y, [0.25, 0.5, 0.75], axis)
    else:
        middle = np.nanmean(y, axis)
        std = np.nanstd(y, axis)
        lower = middle - std
        upper = middle + std
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

    violin_vertices = body.get_paths()[0].vertices
    N = (violin_vertices.shape[0] - 1) // 2
    if side in ("low", "both"):
        X = violin_vertices[1:N, 0]
        Y = violin_vertices[1:N, 1]
    else:
        X = violin_vertices[-2 : -N - 1 : -1, 0]
        Y = violin_vertices[-2 : -N - 1 : -1, 1]
    if vert:
        X, Y = Y, X
    interp = CubicSpline(X, Y)

    int_vert = int(vert)
    int_not_vert = int(not vert)

    for k in ("cmeans", "cmedians", "cquantiles"):
        if k not in violin_data:
            continue

        linecollection = violin_data[k]
        if color:
            linecollection.set_color(color)

        for line in linecollection.get_paths():
            value = line.vertices[0, int_vert]
            vertex = interp(value)
            if side == "low":
                line.vertices[0, int_not_vert] = vertex
            elif side == "high":
                line.vertices[1, int_not_vert] = vertex
            else:
                line.vertices[:, int_not_vert] = [vertex, 2 * position - vertex]
