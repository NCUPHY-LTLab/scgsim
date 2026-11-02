"""Probe existing lumped-port mesh incidence against exact planned terminals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from scgsim.geometry.construction.native_geometry.reports import failure_record
from scgsim.geometry.construction.native_types import GmshModelApi
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.primitives.port_contract import route_b_port_sheet_binding_matches


@dataclass
class LumpedPortMeshProbe:
    """Share a disposable mesh and exact port-terminal inventories across owners."""

    plan: ConstructionPlanRecord
    gmsh: GmshModelApi
    loops_by_id: Mapping[str, SurfaceLoopRecord]
    curve_tags: Mapping[str, int]
    surface_tag_by_id: Mapping[str, int]
    failures: list[dict[str, Any]]
    records: list[dict[str, Any]]

    def append(self) -> None:
        """Append the existing port evidence and always clear the disposable mesh."""
        if self.plan.route not in {"A", "B"}:
            return
        port_surfaces = tuple(
            surface
            for surface in self.plan.surfaces
            if surface.surface_role == "lumped_port"
        )
        if not port_surfaces:
            return
        try:
            self.gmsh.model.mesh.generate(2)
            for port_surface in port_surfaces:
                self.port_surface = port_surface
                self.append_port()
        except Exception as exc:  # noqa: BLE001 - report a failed disposable mesh probe
            for self.port_surface in port_surfaces:
                self.failures.append(
                    failure_record(
                        "lumped_port_mesh_generation_failed",
                        [self.port_surface.surface_id],
                        f"Route {self.plan.route} lumped-port mesh probe failed: {exc}",
                    )
                )
        finally:
            self.gmsh.model.mesh.clear()

    def append_port(self):
        """Resolve one retained port's mesh nodes and ordered terminal owners."""
        self.port_tag = self.surface_tag_by_id.get(self.port_surface.surface_id)
        if self.port_tag is None:
            return
        self.port_nodes = _surface_mesh_node_tags(self.gmsh, self.port_tag)
        owners = tuple(
            str(owner) for owner in self.port_surface.metadata["owner_semantic_ids"]
        )
        self.port_curve_ids = {
            curve_ref.curve_id
            for loop_id in (
                self.port_surface.outer_loop_ref,
                *self.port_surface.hole_loop_refs,
            )
            for curve_ref in self.loops_by_id[cast(str, loop_id)].curve_refs
        }
        for owner_id in owners:
            self.append_owner(owner_id)

    def append_owner(self, owner_id: str):
        """Compare exact credited terminal curves and shared native mesh nodes."""
        overlap = _route_b_port_overlap(self.port_surface, owner_id)
        overlap_id = (
            str(overlap.get("overlap_id")) if isinstance(overlap, Mapping) else None
        )
        owner_pec_surfaces = tuple(
            surface
            for surface in self.plan.surfaces
            if surface.owner_semantic_id == owner_id
            and (
                (
                    self.plan.route == "A"
                    and surface.metadata.get("representation") == "surface_sheet"
                )
                or (
                    self.plan.route == "B"
                    and surface.surface_role == "cutout_boundary_shell"
                    and str(surface.geometry_ref.get("shell_part", "")).startswith(
                        "sidewall"
                    )
                    and surface.metadata.get("route_b_port_sheet_sidewall_partition")
                    and route_b_port_sheet_binding_matches(
                        surface,
                        port_surface_id=self.port_surface.surface_id,
                        overlap_id=overlap_id,
                        host_semantic_id=owner_id,
                    )
                )
            )
        )
        sidewall_tags = tuple(
            self.surface_tag_by_id[surface.surface_id]
            for surface in owner_pec_surfaces
            if surface.surface_id in self.surface_tag_by_id
        )
        owner_surface_ids = {surface.surface_id for surface in owner_pec_surfaces}
        credited_terminal_curve_ids = tuple(
            curve.curve_id
            for curve in self.plan.curves
            if curve.curve_id in self.port_curve_ids
            and owner_surface_ids.intersection(curve.used_by_surface_ids)
        )
        owner_sidewall_nodes: set[int] = set()
        for sidewall_tag in sidewall_tags:
            owner_sidewall_nodes.update(
                _surface_mesh_node_tags(self.gmsh, sidewall_tag)
            )
        shared_nodes = self.port_nodes & owner_sidewall_nodes
        expected_terminal_curve_ids = (
            _route_b_expected_terminal_curve_ids(
                self.plan,
                port_curve_ids=self.port_curve_ids,
                overlap=overlap,
            )
            if self.plan.route == "B"
            else credited_terminal_curve_ids
        )
        terminal_curve_node_counts = {
            curve_id: len(
                _curve_mesh_node_tags(self.gmsh, self.curve_tags[curve_id])
                & self.port_nodes
                & owner_sidewall_nodes
            )
            for curve_id in expected_terminal_curve_ids
            if curve_id in self.curve_tags
        }
        exact_terminal_coverage = set(credited_terminal_curve_ids) == set(
            expected_terminal_curve_ids
        )
        status = (
            "pass"
            if (
                shared_nodes
                and exact_terminal_coverage
                and len(terminal_curve_node_counts) == len(expected_terminal_curve_ids)
                and all(terminal_curve_node_counts.values())
            )
            else "fail"
        )
        if status == "fail":
            self.failures.append(
                failure_record(
                    "lumped_port_missing_shared_mesh_nodes",
                    [self.port_surface.surface_id, owner_id],
                    f"Route {self.plan.route} lumped-port sheet must share exact "
                    "terminal curves and mesh nodes with each owner PEC "
                    "surface.",
                )
            )
        self.records.append(
            {
                "source_record_kind": (
                    f"route_{self.plan.route.lower()}_lumped_port_mesh"
                ),
                "source_record_id": self.port_surface.surface_id,
                "owner_semantic_id": owner_id,
                "overlap_id": overlap_id,
                "backend_port_tag": self.port_tag,
                "backend_owner_pec_surface_tags": sorted(sidewall_tags),
                "bound_owner_pec_surface_ids": sorted(owner_surface_ids),
                "expected_terminal_curve_ids": sorted(expected_terminal_curve_ids),
                "credited_terminal_curve_ids": sorted(credited_terminal_curve_ids),
                "per_terminal_curve_shared_node_counts": (terminal_curve_node_counts),
                "planned_shared_curve_ids": sorted(credited_terminal_curve_ids),
                "backend_shared_curve_tags": sorted(
                    self.curve_tags[curve_id]
                    for curve_id in credited_terminal_curve_ids
                    if curve_id in self.curve_tags
                ),
                "shared_mesh_node_count": len(shared_nodes),
                "status": status,
            }
        )


def _surface_mesh_node_tags(gmsh: GmshModelApi, surface_tag: int) -> set[int]:
    node_tags, _, _ = gmsh.model.mesh.getNodes(
        2,
        surface_tag,
        includeBoundary=True,
    )
    return {int(node_tag) for node_tag in node_tags}


def _curve_mesh_node_tags(gmsh: GmshModelApi, curve_tag: int) -> set[int]:
    node_tags, _, _ = gmsh.model.mesh.getNodes(
        1,
        curve_tag,
        includeBoundary=True,
    )
    return {int(node_tag) for node_tag in node_tags}


def _route_b_port_overlap(
    port_surface: SurfacePlanRecord, owner_id: str
) -> Mapping[str, Any] | None:
    source_provenance = port_surface.metadata.get("source_provenance", {})
    if not isinstance(source_provenance, Mapping):
        return None
    for overlap in source_provenance.get("overlaps", ()):
        if isinstance(overlap, Mapping) and overlap.get("host_semantic_id") == owner_id:
            return overlap
    return None


def _route_b_expected_terminal_curve_ids(
    plan: ConstructionPlanRecord,
    *,
    port_curve_ids: set[str],
    overlap: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if overlap is None:
        return ()
    if "native_terminal_curve_ids" in overlap:
        return tuple(overlap["native_terminal_curve_ids"])
    overlap_loop = overlap.get("overlap_loop")
    if not isinstance(overlap_loop, Sequence):
        return ()
    points_by_id = {point.point_id: point.coordinate for point in plan.points}
    curves_by_id = {curve.curve_id: curve for curve in plan.curves}
    return tuple(
        sorted(
            curve_id
            for curve_id in port_curve_ids
            if curve_id in curves_by_id
            and _curve_matches_overlap(
                curves_by_id[curve_id],
                points_by_id=points_by_id,
                overlap_loop=overlap_loop,
            )
        )
    )


def _curve_matches_overlap(
    curve: CurvePlanRecord,
    *,
    points_by_id: Mapping[str, tuple[float, float, float]],
    overlap_loop: Sequence[Any],
) -> bool:
    start = points_by_id.get(curve.start_point_id)
    end = points_by_id.get(curve.end_point_id)
    if start is None or end is None:
        return False
    return any(
        _segments_overlap(start[:2], end[:2], overlap_start, overlap_end)
        for overlap_start, overlap_end in _ring_edges(overlap_loop)
    )


def _ring_edges(
    loop: Sequence[Any],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    return tuple(
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _segments_overlap(
    start: tuple[float, float],
    end: tuple[float, float],
    other_start: tuple[float, float],
    other_end: tuple[float, float],
) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return False
    cross_start = (other_start[0] - start[0]) * dy - (other_start[1] - start[1]) * dx
    cross_end = (other_end[0] - start[0]) * dy - (other_end[1] - start[1]) * dx
    if abs(cross_start) > 1e-9 or abs(cross_end) > 1e-9:
        return False
    first = (
        (other_start[0] - start[0]) * dx + (other_start[1] - start[1]) * dy
    ) / length_sq
    second = (
        (other_end[0] - start[0]) * dx + (other_end[1] - start[1]) * dy
    ) / length_sq
    return min(1.0, max(first, second)) - max(0.0, min(first, second)) > 1e-9
