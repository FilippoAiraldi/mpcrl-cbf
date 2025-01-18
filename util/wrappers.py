import sys
from contextlib import contextmanager
from typing import Any, TypeVar

import casadi as cs
from csnlp import Nlp, wrappers
from mpcrl import LearningAgent

from util.defaults import TIME_MEAS

SymType = TypeVar("SysType", cs.MX, cs.SX)


class DummyFile:
    """A dummy file object that does nothing."""

    def write(self, x):
        pass


DUMMY_FILE = DummyFile()


@contextmanager
def nostdout():
    """Suppresses the standard output."""
    save_stdout = sys.stdout
    try:
        sys.stdout = DUMMY_FILE
        yield
    finally:
        sys.stdout = save_stdout


class SuppressOutput(wrappers.Wrapper[SymType]):
    """A wrapper class that suppresses the output of the solver to console."""

    def init_solver(self, *args: Any, **kwargs: Any) -> Any:
        with nostdout():
            return self.nlp.init_solver(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with nostdout():
            return self.nlp.__call__(*args, **kwargs)


class RecordSolverTime(wrappers.Wrapper[SymType]):
    """A wrapper class of that records the time taken by the solver."""

    def __init__(self, nlp: Nlp[SymType]) -> None:
        super().__init__(nlp)
        self.solver_time: list[float] = []
        self._agent: LearningAgent | None = None

    def set_learning_agent(self, agent: LearningAgent) -> None:
        """Optionally set the agent to record times only when training."""
        self._agent = agent

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sol = self.nlp.__call__(*args, **kwargs)
        if self._agent is None or self._agent.unwrapped._is_training:
            self.solver_time.append(sol.stats[TIME_MEAS])
        return sol
