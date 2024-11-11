import sys
from contextlib import contextmanager
from typing import Any, TypeVar

import casadi as cs
from csnlp import Nlp, wrappers

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sol = self.nlp.__call__(*args, **kwargs)
        self.solver_time.append(sol.stats["t_proc_total"])
        return sol
