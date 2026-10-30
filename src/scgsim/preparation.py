"""Explicit, run-scoped resource ownership for geometry preparation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from threading import RLock
from typing import TypeVar, cast

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
        pending: dict[Future[_R], int] = {}
        iterator = iter(items)
        results: list[_R | None] = []
        failures: dict[int, BaseException] = {}
        exhausted = False

        def refill() -> None:
            nonlocal exhausted
            while not exhausted and len(pending) < state.max_workers:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                context = copy_context()
                future = state.executor.submit(context.run, _in_worker, function, item)
                pending[future] = len(results)
                results.append(None)

        def collect(future: Future[_R]) -> None:
            index = pending.pop(future)
            if future.cancelled():
                return
            try:
                results[index] = future.result()
            except BaseException as error:
                failures[index] = error

        def drain(*, after: int) -> None:
            for future, index in pending.items():
                if index > after:
                    future.cancel()
            # Earlier inputs must settle to select the first input-order error.
            # All running tasks own contextual state until they settle.
            for future in list(pending):
                collect(future)

        try:
            refill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    collect(future)
                if failures:
                    break
                refill()
        except BaseException:
            # Iterator and scheduling errors retain their original precedence.
            drain(after=-1)
            raise
        if failures:
            drain(after=min(failures))
            raise failures[min(failures)]
        return cast(list[_R], results)


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
