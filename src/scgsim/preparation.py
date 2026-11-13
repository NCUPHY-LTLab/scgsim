"""Explicit, run-scoped resource ownership for geometry preparation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from threading import RLock
from typing import TypeVar

from scgsim.refusal import internal, refused

_WHERE_PREPARATION_CALL = "the call to preparation_resources"

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass
class _PreparationState:
    """One context-local worker budget and its lazily created shared pool."""

    max_workers: int
    executor: ThreadPoolExecutor | None = None
    # One ordered map or native stage owns this budget at a time.
    lock: RLock = field(default_factory=RLock)


# Ambient by design: the worker budget only decides how many threads run a map whose outputs keep input order,
# so no result or message depends on it.
_state: ContextVar[_PreparationState | None] = ContextVar(
    "scgsim_preparation", default=None
)
# Ambient by design: the nesting depth only makes a map inside a worker run serially; it changes no result.
_worker_depth: ContextVar[int] = ContextVar("scgsim_preparation_depth", default=0)


def current_preparation_workers() -> int | None:
    """Return the explicit outer budget, or None for ordinary serial preparation."""
    state = _state.get()
    return None if state is None else state.max_workers


@contextmanager
def preparation_resources(*, max_workers: int) -> Iterator[None]:
    """Share one lazy thread pool across preparation calls in this scope.

    Nested contexts reuse the outer budget. Workers execute nested maps serially;
    outputs and exception delivery follow input order. This context changes no
    native environment settings or scientific input/output payloads.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError(
            refused(
                "max_workers",
                f"must be a positive integer; got {max_workers!r}",
                _WHERE_PREPARATION_CALL,
            )
        )
    if max_workers <= 0:
        raise ValueError(
            refused(
                "max_workers",
                f"must be a positive integer; got {max_workers!r}",
                _WHERE_PREPARATION_CALL,
            )
        )
    if _state.get() is not None:
        yield
        return
    state = _PreparationState(max_workers)
    token = _state.set(state)
    primary: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            with state.lock:
                if state.executor is not None:
                    state.executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                "Preparation executor cleanup failed: " + type(error).__name__
            )
        finally:
            _state.reset(token)


def _in_worker(function: Callable[[_T], _R], item: _T) -> _R:
    """Mark a task as a worker so any nested map runs serially."""
    token = _worker_depth.set(_worker_depth.get() + 1)
    try:
        return function(item)
    finally:
        _worker_depth.reset(token)


def ordered_map(function: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
    """Apply independent tasks with bounded submission and deterministic merge."""
    state = _state.get()
    if state is None or state.max_workers == 1 or _worker_depth.get():
        return [function(item) for item in items]
    with state.lock:
        if state.executor is None:
            state.executor = ThreadPoolExecutor(
                max_workers=state.max_workers, thread_name_prefix="scgsim-preparation"
            )
        pending: deque[Future[_R]] = deque()
        iterator = iter(items)
        results: list[_R] = []
        try:
            for _ in range(state.max_workers):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                context = copy_context()
                pending.append(
                    state.executor.submit(context.run, _in_worker, function, item)
                )
            while pending:
                results.append(pending.popleft().result())
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                context = copy_context()
                pending.append(
                    state.executor.submit(context.run, _in_worker, function, item)
                )
        except BaseException:
            for future in pending:
                future.cancel()
            # Running tasks own contextual state until they settle; never return early.
            for future in pending:
                try:
                    future.result()
                except BaseException:  # noqa: BLE001 -- settle every owned future before returning the original failure
                    pass
            raise
        return results


@contextmanager
def native_preparation() -> Iterator[None]:
    """Hold the preparation budget while native meshing runs in child processes.

    Meshing is single-threaded (the meshing module fixes one Gmsh thread for reproducibility), so the
    children need no share of the worker budget; the guard only keeps native meshing out of a preparation
    worker and serialises it with other budgeted stages.
    """
    state = _state.get()
    if state is None:
        yield
        return
    if _worker_depth.get():
        raise RuntimeError(
            internal("Native meshing cannot run inside a preparation worker")
        )
    with state.lock:
        token = _worker_depth.set(1)
        try:
            yield
        finally:
            _worker_depth.reset(token)
