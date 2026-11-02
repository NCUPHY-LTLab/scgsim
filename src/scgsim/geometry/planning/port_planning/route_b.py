"""Partition Route B port sidewalls and bind their final topology and volume hosts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.regions import PortSheetOverlapRecord
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.domain import solution_entity_xy_region
from scgsim.geometry.primitives.entities import entity_z_range_um
from scgsim.geometry.primitives.grid import boolean_polygon_region
from scgsim.geometry.primitives.loops import gdstk_surface_region
from scgsim.geometry.primitives.planar_loops import clean_loop, ring_edges
from scgsim.geometry.primitives.port_contract import route_b_port_sheet_binding_matches
from scgsim.geometry.primitives.spatial import (
    geometry_ref_surface_z_um,
    same_coordinate_z,
    segment_overlap_interval,
)


def bind_route_b_port_volume_plans(
    build_input: GeometryBuildInput,
    surfaces: tuple[SurfacePlanRecord, ...],
    volumes: tuple[VolumePlanRecord, ...],
) -> tuple[SurfacePlanRecord, ...]:
    """Bind active port remainders to final geometric vacuum pieces.

    Material physical groups may aggregate these pieces; their names cannot
    identify a native mesh embedding host.
    """
    import gdstk  # noqa: PLC0415 -- load the optional polygon engine only at this native geometry boundary.

    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result = []
    for surface in surfaces:
        if surface.surface_role != "lumped_port":
            result.append(surface)
            continue
        z = geometry_ref_surface_z_um(surface.geometry_ref)
        remainder = gdstk_surface_region(surface.geometry_ref)
        hosts = []
        for volume in volumes:
            entity = entities[volume.owner_semantic_id]
            if entity.material_kind != "vacuum":
                continue
            z_min, z_max = entity_z_range_um(entity)
            if not z_min < z < z_max:
                continue
            if boolean_polygon_region(
                gdstk, remainder, solution_entity_xy_region(gdstk, entity), "and"
            ):
                hosts.append(volume.volume_id)
        if not hosts:
            raise ValueError(
                f"{surface.surface_id} active port has no final vacuum volume host"
            )
        result.append(
            replace(
                surface,
                metadata={**surface.metadata, "embedded_volume_plan_ids": tuple(hosts)},
            )
        )
    return tuple(result)


def partition_route_b_port_sheet_sidewalls(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
    remainder: Mapping[str, Any],
    overlaps_by_host: Mapping[str, Any],
) -> list[SurfacePlanRecord]:
    """Split only the two overlap-bound Route-B PEC sidewalls at the port plane."""
    return _RouteBPortPartitioner(
        surfaces, port_surface, remainder, overlaps_by_host
    ).build()


@dataclass
class _RouteBPortPartitioner:
    """Carry exact terminal segments and the accumulated Route B sidewall children."""

    surfaces: Sequence[SurfacePlanRecord]
    port_surface: SurfacePlanRecord
    remainder: Mapping[str, Any]
    overlaps_by_host: Mapping[str, PortSheetOverlapRecord]
    port_z_um: float = field(init=False)
    terminal_segments_by_host: dict[
        str, tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    ] = field(init=False)
    result: list[SurfacePlanRecord] = field(init=False)
    split_hosts: set[str] = field(init=False)

    def build(self) -> list[SurfacePlanRecord]:
        """Partition each selected sidewall and require both bound hosts."""
        self.initialize()
        for surface in self.surfaces:
            self.partition_surface(surface)
        if set(self.overlaps_by_host) != self.split_hosts:
            missing = tuple(
                (
                    host_id
                    for host_id in self.overlaps_by_host
                    if host_id not in self.split_hosts
                )
            )
            raise ValueError(
                f"{self.port_surface.surface_id} requires finite Route B sidewalls for {missing!r}"
            )
        return self.result

    def initialize(self) -> None:
        """Read the finite terminal segments before modifying any surface."""
        self.port_z_um = geometry_ref_surface_z_um(self.port_surface.geometry_ref)
        self.terminal_segments_by_host = {
            host_id: tuple(
                (
                    (start, end)
                    for boundary in (
                        self.remainder["outer_loop"],
                        *self.remainder["hole_loops"],
                    )
                    for start, end in ring_edges(clean_loop(boundary))
                    if any(
                        (
                            segment_overlap_interval(
                                start, end, overlap_start, overlap_end
                            )
                            is not None
                            for overlap_start, overlap_end in ring_edges(
                                clean_loop(overlap.overlap_loop)
                            )
                        )
                    )
                )
            )
            for host_id, overlap in self.overlaps_by_host.items()
        }
        if any((not segments for segments in self.terminal_segments_by_host.values())):
            missing = tuple(
                (
                    host_id
                    for host_id, segments in self.terminal_segments_by_host.items()
                    if not segments
                )
            )
            raise ValueError(
                f"{self.port_surface.surface_id} has no finite terminal segment for {missing!r}"
            )
        self.result: list[SurfacePlanRecord] = []
        self.split_hosts: set[str] = set()

    def partition_surface(self, surface: SurfacePlanRecord) -> None:
        """Retain or split one source sidewall at the exact port plane."""
        shell_part = str(surface.geometry_ref.get("shell_part", ""))
        if (
            surface.surface_role != "cutout_boundary_shell"
            or surface.owner_semantic_id not in self.overlaps_by_host
            or (not shell_part.startswith("sidewall_"))
        ):
            self.result.append(surface)
            return None
        points = tuple(
            cast(
                Sequence[tuple[float, float, float]],
                surface.geometry_ref.get("quad_points", ()),
            )
        )
        if len(points) != 4:
            raise ValueError(
                f"{surface.surface_id} Route B port host sidewall requires a quad"
            )
        quad = tuple(
            ((float(point[0]), float(point[1]), float(point[2])) for point in points)
        )
        if not any(
            (
                segment_overlap_interval(quad[0][:2], quad[1][:2], start, end)
                is not None
                for start, end in self.terminal_segments_by_host[
                    surface.owner_semantic_id
                ]
            )
        ):
            self.result.append(surface)
            return None
        z_min_um = min((point[2] for point in quad))
        z_max_um = max((point[2] for point in quad))
        if not z_min_um < z_max_um:
            raise ValueError(
                f"{surface.surface_id} Route B port host sidewall has empty height"
            )
        overlap = self.overlaps_by_host[surface.owner_semantic_id]
        binding = {
            "port_surface_id": self.port_surface.surface_id,
            "overlap_id": str(overlap.overlap_id),
            "host_semantic_id": surface.owner_semantic_id,
        }
        if same_coordinate_z(self.port_z_um, z_min_um) or same_coordinate_z(
            self.port_z_um, z_max_um
        ):
            if not surface.metadata.get("route_b_port_sheet_bindings"):
                raise ValueError(
                    f"{self.port_surface.surface_id} requires an imprinted sidewall child for {surface.surface_id}"
                )
            self.result.append(
                replace(
                    surface,
                    metadata=_route_b_port_sheet_partition_metadata(
                        surface.metadata, binding=binding, port_z_um=self.port_z_um
                    ),
                )
            )
            self.split_hosts.add(surface.owner_semantic_id)
            return None
        if not z_min_um < self.port_z_um < z_max_um:
            raise ValueError(
                f"{self.port_surface.surface_id} must cut {surface.surface_id} at its finite sidewall interior"
            )
        lower_quad = (
            quad[0],
            quad[1],
            (quad[2][0], quad[2][1], self.port_z_um),
            (quad[3][0], quad[3][1], self.port_z_um),
        )
        upper_quad = (
            (quad[0][0], quad[0][1], self.port_z_um),
            (quad[1][0], quad[1][1], self.port_z_um),
            quad[2],
            quad[3],
        )
        parent_surface_id = surface.parent_surface_id or surface.surface_id
        partition_metadata = _route_b_port_sheet_partition_metadata(
            surface.metadata, binding=binding, port_z_um=self.port_z_um
        )
        self.result.extend(
            (
                replace(
                    surface,
                    surface_id=f"{surface.surface_id}__{self.port_surface.surface_id}__LOWER",
                    parent_surface_id=parent_surface_id,
                    partition_label="port_lower",
                    geometry_ref={
                        **dict(surface.geometry_ref),
                        "quad_points": lower_quad,
                    },
                    metadata=partition_metadata,
                ),
                replace(
                    surface,
                    surface_id=f"{surface.surface_id}__{self.port_surface.surface_id}__UPPER",
                    parent_surface_id=parent_surface_id,
                    partition_label="port_upper",
                    geometry_ref={
                        **dict(surface.geometry_ref),
                        "quad_points": upper_quad,
                    },
                    metadata=partition_metadata,
                ),
            )
        )
        self.split_hosts.add(surface.owner_semantic_id)


def _route_b_port_sheet_partition_metadata(
    metadata: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    port_z_um: float,
) -> dict[str, Any]:
    bindings = tuple(
        dict(existing)
        for existing in metadata.get("route_b_port_sheet_bindings", ())
        if isinstance(existing, Mapping)
    )
    if len(bindings) != len(metadata.get("route_b_port_sheet_bindings", ())):
        raise ValueError("Route B port sidewall bindings must be structured records")
    if dict(binding) not in bindings:
        bindings = (*bindings, dict(binding))
    return {
        **dict(metadata),
        "route_b_port_sheet_sidewall_partition": True,
        "route_b_port_sheet_plane_z_um": port_z_um,
        "route_b_port_sheet_bindings": bindings,
    }


def validate_route_b_port_sheet_sidewall_topology(
    *,
    route: RouteLiteral,
    points: Sequence[PointPlanRecord],
    curves: Sequence[CurvePlanRecord],
    surface_loops: Sequence[SurfaceLoopRecord],
    surfaces: Sequence[SurfacePlanRecord],
) -> None:
    """Require each Route-B port overlap to reuse just its bound PEC curve set."""
    if route != "B":
        return
    loops_by_id = {loop.loop_id: loop for loop in surface_loops}
    surfaces_by_id = {surface.surface_id: surface for surface in surfaces}
    points_by_id = {point.point_id: point.coordinate for point in points}
    curves_by_id = {curve.curve_id: curve for curve in curves}
    for port_surface in (
        surface for surface in surfaces if surface.surface_role == "lumped_port"
    ):
        owners = tuple(
            str(owner) for owner in port_surface.metadata["owner_semantic_ids"]
        )
        overlaps_by_host = {
            str(overlap["host_semantic_id"]): overlap
            for overlap in port_surface.metadata["source_provenance"]["overlaps"]
            if isinstance(overlap, Mapping)
        }
        port_curve_ids = {
            curve_ref.curve_id
            for loop_id in (port_surface.outer_loop_ref, *port_surface.hole_loop_refs)
            for curve_ref in loops_by_id[cast(str, loop_id)].curve_refs
        }
        if not port_curve_ids:
            raise ValueError(
                f"{port_surface.surface_id} has no canonical terminal curves"
            )
        for host_id in owners:
            overlap = overlaps_by_host.get(host_id)
            if overlap is None:
                raise ValueError(
                    f"{port_surface.surface_id} lacks exact overlap provenance "
                    f"for {host_id}"
                )
            expected_terminal_curve_ids = {
                curve_id
                for curve_id in port_curve_ids
                if _route_b_port_terminal_curve_matches_overlap(
                    curves_by_id[curve_id],
                    points_by_id=points_by_id,
                    overlap_loop=overlap["overlap_loop"],
                )
            }
            if not expected_terminal_curve_ids:
                raise ValueError(
                    f"{port_surface.surface_id} has no expected Route B terminal "
                    f"curves for {host_id}"
                )
            credited_terminal_curve_ids = {
                curve.curve_id
                for curve in curves
                if curve.curve_id in port_curve_ids
                and any(
                    candidate.surface_role == "cutout_boundary_shell"
                    and candidate.owner_semantic_id == host_id
                    and str(candidate.geometry_ref.get("shell_part", "")).startswith(
                        "sidewall_"
                    )
                    and candidate.metadata.get("route_b_port_sheet_sidewall_partition")
                    and route_b_port_sheet_binding_matches(
                        candidate,
                        port_surface_id=port_surface.surface_id,
                        overlap_id=str(overlap["overlap_id"]),
                        host_semantic_id=host_id,
                    )
                    for candidate in (
                        surfaces_by_id[surface_id]
                        for surface_id in curve.used_by_surface_ids
                        if surface_id in surfaces_by_id
                    )
                )
            }
            if credited_terminal_curve_ids != expected_terminal_curve_ids:
                missing = sorted(
                    expected_terminal_curve_ids - credited_terminal_curve_ids
                )
                extra = sorted(
                    credited_terminal_curve_ids - expected_terminal_curve_ids
                )
                raise ValueError(
                    f"{port_surface.surface_id} Route B terminal curves for "
                    f"{host_id} PEC sidewall differ; missing={missing!r}, "
                    f"extra={extra!r}"
                )


def _route_b_port_terminal_curve_matches_overlap(
    curve: CurvePlanRecord,
    *,
    points_by_id: Mapping[str, tuple[float, float, float]],
    overlap_loop: Any,
) -> bool:
    start = points_by_id.get(curve.start_point_id)
    end = points_by_id.get(curve.end_point_id)
    if start is None or end is None:
        return False
    return any(
        segment_overlap_interval(start[:2], end[:2], overlap_start, overlap_end)
        is not None
        for overlap_start, overlap_end in ring_edges(clean_loop(overlap_loop))
    )
