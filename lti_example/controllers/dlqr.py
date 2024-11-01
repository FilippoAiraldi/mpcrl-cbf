from collections.abc import Callable
from time import process_time

import numpy as np
import numpy.typing as npt
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dlqr


def get_dlqr_controller() -> (
    Callable[[npt.NDArray[np.floating]], tuple[npt.NDArray[np.floating], float]]
):
    """Returns the discrete-time LQR controller with action saturation.

    Returns
    -------
    callable from array-like to (array-like, float)
        A controller that maps the current state to the desired action, and returns also
        the time it took to compute the action.
    """
    K, _ = dlqr(Env.A, Env.B, Env.Q, Env.R)
    a_min = Env.a_bound

    def _f(x):
        t0 = process_time()
        u = np.clip(-np.dot(K, x), -a_min, a_min)
        return u, process_time() - t0

    return _f
