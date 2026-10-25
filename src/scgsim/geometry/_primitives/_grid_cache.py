"""Call-local reuse of pure subdivision work; invocation audits stay outside."""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Protocol, TypeVar, cast


class _Operand(Protocol):
    operand_id: str
    exterior: tuple
    holes: tuple
    physical: bool
    contact: bool


class _Context(Protocol):
    grid_size_um: float
    coordinate_frame: str


@dataclass
class _Cache:
    values: dict[tuple, object] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


_ACTIVE: ContextVar[_Cache | None] = ContextVar("scgsim_grid_work_cache", default=None)
_T = TypeVar("_T")


def _number(value):
    # Preserve signed zero and integer/float representation without quantizing.
    return (type(value), value.hex() if isinstance(value, float) else value)


def _key(operands, context, apply_grid):
    def ring(points):
        return tuple(tuple(_number(v) for v in point) for point in points)

    return (
        tuple(
            (
                v.operand_id,
                ring(v.exterior),
                tuple(ring(h) for h in v.holes),
                v.physical,
                v.contact,
            )
            for v in operands
        ),
        (_number(context.grid_size_um), context.coordinate_frame),
        apply_grid,
    )


@contextmanager
def grid_work_cache_scope():
    """Own one fresh outer scope; copied executor contexts share only its cache."""
    if _ACTIVE.get() is not None:
        yield
        return
    cache = _Cache()
    token = _ACTIVE.set(cache)
    try:
        yield
    finally:
        with cache.lock:
            cache.values.clear()
        _ACTIVE.reset(token)


def with_grid_work_cache(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        with grid_work_cache_scope():
            return function(*args, **kwargs)

    return scoped


def cached_subdivision(
    operands: Iterable[_Operand],
    context: _Context,
    *,
    apply_grid: bool,
    build: Callable[[], _T],
) -> _T:
    """Reuse successful immutable subdivisions, never failures or caller audits.

    Concurrent misses may compute twice. No worker waits for another worker's
    future, and the lock never spans geometry computation or nested maps.
    """
    cache = _ACTIVE.get()
    if cache is None:
        return build()
    key = _key(operands, context, apply_grid)
    missing = object()
    with cache.lock:
        value = cache.values.get(key, missing)
    if value is not missing:
        return cast(_T, value)
    result = build()
    with cache.lock:
        cache.values.setdefault(key, result)
    return result
