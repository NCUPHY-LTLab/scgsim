"""Geometry-mesh controls, independent of Palace's finite-element order."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

MESH_CONTROL_KEYS = (
    "size_policy",
    "algorithm_3d",
    "threads",
    "surface_threads",
    "geometry_order",
    "high_order_optimize",
)


@dataclass(frozen=True)
class MeshSizing:
    """One normalized sizing policy shared by preparation and the fresh worker."""

    size_policy: str
    refined_mesh_size: float | None
    max_mesh_size: float | None

    def __post_init__(self) -> None:
        _validate_size_policy(self.size_policy)
        if self.size_policy == "gmsh_default":
            if self.refined_mesh_size is not None or self.max_mesh_size is not None:
                raise ValueError("gmsh_default conflicts with explicit mesh sizes.")
            return
        for name in ("refined_mesh_size", "max_mesh_size"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and > 0.")
            object.__setattr__(self, name, float(value))
        if self.max_mesh_size < self.refined_mesh_size:
            raise ValueError("max_mesh_size must be >= refined_mesh_size.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "size_policy": self.size_policy,
            "refined_mesh_size": self.refined_mesh_size,
            "max_mesh_size": self.max_mesh_size,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MeshSizing:
        return cls(
            payload["size_policy"], payload["refined_mesh_size"], payload["max_mesh_size"]
        )


def _validate_size_policy(size_policy: str) -> None:
    if size_policy not in ("scgsim_default", "gmsh_default"):
        raise ValueError("size_policy must be 'scgsim_default' or 'gmsh_default'.")


def normalize_mesh_sizing(
    *,
    size_policy: str = "scgsim_default",
    refined_mesh_size: float | None = None,
    max_mesh_size: float | None = None,
) -> MeshSizing:
    _validate_size_policy(size_policy)
    if size_policy == "scgsim_default":
        refined_mesh_size = 5.0 if refined_mesh_size is None else refined_mesh_size
        max_mesh_size = 300.0 if max_mesh_size is None else max_mesh_size
    return MeshSizing(size_policy, refined_mesh_size, max_mesh_size)


def update_mesh_sizing(
    current: MeshSizing,
    *,
    size_policy: str | None = None,
    refined_mesh_size: float | None = None,
    max_mesh_size: float | None = None,
) -> MeshSizing:
    policy = current.size_policy if size_policy is None else size_policy
    _validate_size_policy(policy)
    if policy == "scgsim_default" and current.size_policy == policy:
        if refined_mesh_size is None:
            refined_mesh_size = current.refined_mesh_size
        if max_mesh_size is None:
            max_mesh_size = current.max_mesh_size
    return normalize_mesh_sizing(
        size_policy=policy,
        refined_mesh_size=refined_mesh_size,
        max_mesh_size=max_mesh_size,
    )


def mesh_controls(
    *,
    size_policy: str = "scgsim_default",
    algorithm_3d: str = "Delaunay",
    threads: int = 1,
    surface_threads: int = 1,
    geometry_order: int = 1,
    high_order_optimize: bool = True,
) -> dict[str, Any]:
    """Normalize supported native controls without imposing a research envelope."""
    _validate_size_policy(size_policy)
    if algorithm_3d not in {"Delaunay", "HXT"}:
        raise ValueError("algorithm_3d must be 'Delaunay' or 'HXT'.")
    for name, value in (("threads", threads), ("surface_threads", surface_threads)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer native Gmsh thread setting.")
    if (
        isinstance(geometry_order, bool)
        or not isinstance(geometry_order, int)
        or geometry_order < 1
    ):
        raise ValueError("geometry_order must be a positive integer.")
    if not isinstance(high_order_optimize, bool):
        raise TypeError("high_order_optimize must be a bool.")
    return dict(
        size_policy=size_policy,
        algorithm_3d=algorithm_3d,
        threads=threads,
        surface_threads=surface_threads,
        geometry_order=geometry_order,
        high_order_optimize=high_order_optimize,
    )
