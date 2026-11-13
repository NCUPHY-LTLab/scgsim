"""Canonical interfaces, surfaces and metal-to-metal contacts for geometry plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import (
    DEFAULT_INTERFACE_SOLVER_USE,
    InterfaceKindLiteral,
    RouteLiteral,
    SolverUseLiteral,
    Vector3D,
)
from scgsim.geometry.models.refusals import internal


@dataclass(frozen=True)
class InterfacePlanRecord:
    """Pre-construction declaration that two semantic owners share a surface.

    `interface_id` is the stable semantic id and must start with the interface
    kind directly, such as `MM__`, `MS__`, `MA__`, `SA__`, `SS__`, or `AA__`.
    There is no extra `IF__` prefix.

    Every interface must be recognized before OCC geometry creation. This
    record only states the semantic fact: which two owners meet, what kind of
    interface it is, and which recognition rule proved it. Backend-live
    geometry is created only from `SurfacePlanRecord`.

    Required metadata for bottom-up OCC planning should be attached here as
    soon as it is recognized: `contact_plane` for the physical plane of contact,
    `footprint` for projected contact geometry, and `loop_geometry_ref` when the
    interface is already known as an explicit loop source.

    Route A surface sheets may need side-aware semantics on the same geometric
    loop. Use metadata such as `surface_side`, `adjacent_domain_id`,
    `excluded_owner_semantic_id`, and `interface_kinds` to say which side is
    MA/MS/MM without creating overlapping backend surfaces.
    """

    interface_id: str
    kind: InterfaceKindLiteral
    owner_semantic_ids: tuple[str, str]
    recognition_rule: str
    source_polygon_ids: tuple[str, ...] = ()
    host_domain_id: str | None = None
    solver_use: SolverUseLiteral | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.interface_id.startswith(f"{self.kind}__"):
            raise ValueError(
                internal(
                    f"interface {self.interface_id!r} of kind {self.kind!r} has an id "
                    f"that does not start with {self.kind}__"
                )
            )
        if self.solver_use is None:
            object.__setattr__(
                self,
                "solver_use",
                DEFAULT_INTERFACE_SOLVER_USE[self.kind],
            )


@dataclass(frozen=True)
class SurfacePlanRecord:
    """One backend-live planned surface to create before volume assembly.

    Shared interfaces are represented by one `surface_id`, and both owning
    volumes must reference that same id. Partitioned interfaces are already
    expanded into child `SurfacePlanRecord`s before backend lowering. The
    backend must create each child surface directly from its loops and hole
    loops; it must not cut a parent surface later.

    v1 backend lowering should consume `outer_loop_ref` and `hole_loop_refs`.
    `geometry_ref` remains as audit/source metadata: it may carry `plane` or
    `contact_plane`, `outer_loop`, optional `hole_loops`, optional `footprint`,
    and optional `loop_geometry_ref`. A surface that cannot be tied to canonical
    loop refs must fail during canonicalization rather than falling back to
    global fragment-based discovery.

    `metadata` may carry side/exposure semantics such as `interface_kinds`,
    `surface_side`, `owner_semantic_ids`, `adjacent_domain_id`,
    and `excluded_owner_semantic_id`. Route A surface-sheet conductors must be
    represented by these interface surfaces, not by extra standalone sheet
    surfaces.
    When a surface belongs to more than one geometry interface kind, physical
    names must include the joined interface kinds, for example `MM_MS__...`.
    """

    surface_id: str
    owner_semantic_id: str
    surface_role: str
    geometry_ref: Mapping[str, Any]
    outer_loop_ref: str | None = None
    hole_loop_refs: tuple[str, ...] = ()
    interface_id: str | None = None
    parent_surface_id: str | None = None
    partition_label: str | None = None
    normal_hint: Vector3D | None = None
    valid_routes: tuple[RouteLiteral, ...] = ()
    solver_use: SolverUseLiteral = "solver_active"
    construction_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MMContactRecord:
    """Hidden finite-area same-net conductor contact provenance.

    Contacts are topology records, never physical groups.  They preserve the
    original faces/fragments that were normalized into one PEC component while
    keeping Route A/B solver geometry free of an internal contact boundary.

    Polygon contacts keep their outer loop and plane normal. Curved contacts
    carry a detached canonical graph in geometry_ref; their polygon loop is
    absent, and a curved wall has no fictitious constant normal.
    """

    contact_id: str
    lower_entity_id: str
    upper_entity_id: str
    lower_source_face_id: str
    upper_source_face_id: str
    lower_source_fragment_ids: tuple[str, ...]
    upper_source_fragment_ids: tuple[str, ...]
    outer_loop: tuple[tuple[float, float], ...] | None
    area_um2: float
    normal: Vector3D | None
    conductor_component_id: str
    net_id: str | None
    equipotential_id: str | None = None
    layer_provenance: Mapping[str, Any] = field(default_factory=dict)
    material_provenance: Mapping[str, Any] = field(default_factory=dict)
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    geometry_ref: Mapping[str, Any] = field(default_factory=dict)
