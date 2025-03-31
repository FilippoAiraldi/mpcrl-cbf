from collections.abc import Callable
from time import perf_counter, process_time
from typing import Any

import numpy as np
import numpy.typing as npt
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dlqr

from util.defaults import TIME_MEAS


def get_dlqr_controller(*_: Any, **__: Any) -> tuple[
    Callable[[npt.NDArray[np.floating], Env], tuple[npt.NDArray[np.floating], float]],
    dict[str, npt.NDArray[np.floating]],
]:
    """Returns the discrete-time LQR controller with action saturation.

    Returns
    -------
    callable from (array-like, ConstrainedLtiEnv) to (array-like, float)
        A controller that maps the current state + env to the desired action, and
        returns also the time it took to compute the action.
    dict of str to arrays
        The numerical weights of the parametric MPC controller, if any.
    """
    K, _ = dlqr(Env.A, Env.B, np.diag(Env.Q), np.diag(Env.R))
    a_min = Env.a_bound
    timer_func = perf_counter if "wall" in TIME_MEAS else process_time

    def _f(x, _):
        t0 = timer_func()
        u = np.clip(-np.dot(K, x), -a_min, a_min)
        return u, timer_func() - t0

    return _f, {}
