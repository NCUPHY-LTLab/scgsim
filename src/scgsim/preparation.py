"""Explicit, run-scoped resource ownership for geometry preparation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from threading import RLock
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass
class _PreparationState:
    max_workers: int
    executor: ThreadPoolExecutor | None = None
    # One ordered map or native stage owns this budget at a time.
    lock: RLock = field(default_factory=RLock)


_state: ContextVar[_PreparationState | None] = ContextVar(
    "scgsim_preparation", default=None
)
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
        raise TypeError("max_workers must be a positive integer.")
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer.")
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
    token = _worker_depth.set(_worker_depth.get() + 1)
    try:
        return function(item)
    finally:
        _worker_depth.reset(token)


def _ordered_map(function: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
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
        results = []
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
                except BaseException:
                    pass
            raise
        return results


@contextmanager
def _native_preparation(mesh_controls: Mapping[str, object]) -> Iterator[None]:
    """Reserve the pool budget only while the native mesh child is running."""
    state = _state.get()
    if state is None:
        yield
        return
    if _worker_depth.get():
        raise RuntimeError("Native meshing cannot run inside a preparation worker.")
    with state.lock:
        for name in ("threads", "surface_threads"):
            value = mesh_controls[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Scoped Gmsh thread counts must be positive integers.")
            if value <= 0 or value > state.max_workers:
                raise ValueError(
                    "Gmsh threads and surface_threads must be positive and fit "
                    "preparation_resources max_workers; lower the requested mesh "
                    "threads or increase the explicit preparation budget."
                )
        token = _worker_depth.set(1)
        try:
            yield
        finally:
            _worker_depth.reset(token)
