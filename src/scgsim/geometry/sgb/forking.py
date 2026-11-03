"""Process forking for the per-tile workers.

Owns the two platform facts the runtime depends on: whether the ``fork`` start
method exists (workers inherit their read-only input through it, so nothing is
pickled), and silencing the interpreter's generic multi-threaded-fork warning for
the pure stage functions that run in the workers.
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.context import BaseContext


def fork_context() -> BaseContext | None:
    """The ``fork`` start method, or ``None`` where the platform has none (stages then run in-process)."""
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


@contextmanager
def quiet_fork_warning() -> Iterator[None]:
    """Workers are forked from a process that already runs pool-management threads.

    The stage functions are pure and never touch those threads' locks, so the
    interpreter's multi-threaded-fork warning does not apply to them.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*multi-threaded.*fork.*", category=DeprecationWarning
        )
        yield
