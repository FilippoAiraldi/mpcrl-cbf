from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from env import ConstrainedLtiEnv as Env
from mpcrl.util.control import dlqr


def get_dlqr_controller() -> (
    Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
):
    """Returns the discrete-time LQR controller with action saturation.

    Returns
    -------
    callable from array-like to array-like
        A controller that maps the current state to the desired action.
    """
    K, _ = dlqr(Env.A, Env.B, Env.Q, Env.R)
    a_min = Env.a_bound

    def _f(x):
        return np.clip(-np.dot(K, x), -a_min, a_min)

    return _f
