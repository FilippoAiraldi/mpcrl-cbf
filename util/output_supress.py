import sys
from contextlib import contextmanager


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
