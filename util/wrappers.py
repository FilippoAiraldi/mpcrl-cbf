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

    def init_solver(self, *args: Any, **kwds: Any) -> Any:
        with nostdout():
            return super().init_solver(*args, **kwds)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with nostdout():
            return super().__call__(*args, **kwds)


class RecordSolverTime(wrappers.Wrapper[SymType]):
    """A wrapper class of that records the time taken by the solver."""

    def __init__(self, nlp: Nlp[SymType]) -> None:
        super().__init__(nlp)
        self.solver_time: list[float] = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        sol = super().__call__(*args, **kwds)
        self.solver_time.append(sol.stats["t_proc_total"])
        return sol
