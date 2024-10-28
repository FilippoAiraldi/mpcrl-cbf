from typing import Any

import numpy as np
from gymnasium import logger
from gymnasium.spaces import Box


class LooseBox(Box):
    """A box space, where the bounds in the method `contains` are checked up to some
    tolerance.

    Parameters
    ----------
    args, kwargs : any
        Arguments and keyword arguments to pass to the `Box` constructor.
    rtol : float, optional
        The relative tolerance parameter for `numpy.isclose`, by default `1e-8`.
    atol : float, optional
        The absolute tolerance parameter for `numpy.isclose`, by default `1e-8`.
    """

    def __init__(
        self, *args: Any, rtol: float = 1e-8, atol: float = 1e-8, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rtol = rtol
        self.atol = atol

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            logger.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(
                (x >= self.low) | (np.isclose(x, self.low, self.rtol, self.atol))
            )
            and np.all(
                (x <= self.high) | (np.isclose(x, self.high, self.rtol, self.atol))
            )
        )

    def __repr__(self) -> str:
        return super().__repr__().replace(Box.__name__, self.__class__.__name__)
