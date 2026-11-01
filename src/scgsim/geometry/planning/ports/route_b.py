"""Route B port sheets: partition cutout sidewalls and bind port volumes.

In Route B the port lands on a host-owned cutout shell, so the shell sidewalls are
partitioned at the port and validated for closed topology. See ``docs/geometry-sgb.qmd``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.solutions.occupancy import solution_entity_xy_region
from scgsim.geometry.primitives.entities import entity_z_range_um
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    gdstk_surface_region,
    normalized_loop,
    ring_edges,
)
from scgsim.geometry.primitives.port_contract import (
    route_b_port_sheet_binding_matches,
)
from scgsim.geometry.primitives.spatial import (
    geometry_ref_surface_z_um,
    is_same_z,
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
    import gdstk

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
            if boolean_gdstk_region(
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
    """Split only the two overlap-bound Route-B PEC sidewalls at the port plane.

    The port is a flat sheet between two host conductors. Each host's PEC sidewall that
    touches the port must be cut at the port's z, so the port edge lies on an existing
    curve and the mesh stays conformal.
    """
    port_z_um = geometry_ref_surface_z_um(port_surface.geometry_ref)
    terminal_segments_by_host = _terminal_segments_by_host(
        port_surface, remainder, overlaps_by_host
    )
    result: list[SurfacePlanRecord] = []
    split_hosts: set[str] = set()
    for surface in surfaces:
        pieces, was_split = _partition_sidewall_at_port(
            surface,
            port_surface=port_surface,
            port_z_um=port_z_um,
            overlaps_by_host=overlaps_by_host,
            terminal_segments_by_host=terminal_segments_by_host,
        )
        result.extend(pieces)
        if was_split:
            split_hosts.add(surface.owner_semantic_id)
    if set(overlaps_by_host) != split_hosts:
        missing = tuple(
            host_id for host_id in overlaps_by_host if host_id not in split_hosts
        )
        raise ValueError(
            f"{port_surface.surface_id} requires finite Route B sidewalls "
            f"for {missing!r}"
        )
    return result


def _terminal_segments_by_host(
    port_surface: SurfacePlanRecord,
    remainder: Mapping[str, Any],
    overlaps_by_host: Mapping[str, Any],
) -> dict[str, tuple[Any, ...]]:
    """Find the port edges that lie along each host's overlap; every host needs one."""
    terminal_segments_by_host = {
        host_id: tuple(
            (start, end)
            for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
            for start, end in ring_edges(normalized_loop(boundary))
            if any(
                segment_overlap_interval(start, end, overlap_start, overlap_end)
                is not None
                for overlap_start, overlap_end in ring_edges(
                    normalized_loop(overlap.overlap_loop)
                )
            )
        )
        for host_id, overlap in overlaps_by_host.items()
    }
    if any(not segments for segments in terminal_segments_by_host.values()):
        missing = tuple(
            host_id
            for host_id, segments in terminal_segments_by_host.items()
            if not segments
        )
        raise ValueError(
            f"{port_surface.surface_id} has no finite terminal segment for {missing!r}"
        )
    return terminal_segments_by_host


def _partition_sidewall_at_port(
    surface: SurfacePlanRecord,
    *,
    port_surface: SurfacePlanRecord,
    port_z_um: float,
    overlaps_by_host: Mapping[str, Any],
    terminal_segments_by_host: Mapping[str, Sequence[Any]],
) -> tuple[list[SurfacePlanRecord], bool]:
    """Return the surfaces that replace one surface, and whether it was a port sidewall.

    Surfaces that are not a touching host sidewall come back unchanged. A sidewall whose
    top or bottom already lies at the port plane is only tagged; one that the port cuts
    through its interior is split into a lower and an upper quad.
    """
    shell_part = str(surface.geometry_ref.get("shell_part", ""))
    if (
        surface.surface_role != "cutout_boundary_shell"
        or surface.owner_semantic_id not in overlaps_by_host
        or not shell_part.startswith("sidewall_")
    ):
        return [surface], False
    points = tuple(surface.geometry_ref.get("quad_points", ()))
    if len(points) != 4:
        raise ValueError(
            f"{surface.surface_id} Route B port host sidewall requires a quad"
        )
    quad = tuple(
        (float(point[0]), float(point[1]), float(point[2])) for point in points
    )
    if not any(
        segment_overlap_interval(quad[0][:2], quad[1][:2], start, end) is not None
        for start, end in terminal_segments_by_host[surface.owner_semantic_id]
    ):
        return [surface], False
    z_min_um = min(point[2] for point in quad)
    z_max_um = max(point[2] for point in quad)
    if not z_min_um < z_max_um:
        raise ValueError(
            f"{surface.surface_id} Route B port host sidewall has empty height"
        )
    overlap = overlaps_by_host[surface.owner_semantic_id]
    binding = {
        "port_surface_id": port_surface.surface_id,
        "overlap_id": str(overlap.overlap_id),
        "host_semantic_id": surface.owner_semantic_id,
    }
    if is_same_z(port_z_um, z_min_um) or is_same_z(port_z_um, z_max_um):
        # The sidewall already ends at the port plane, so nothing is cut here. It must
        # already be an imprinted child that carries port bindings; this adds one.
        if not surface.metadata.get("route_b_port_sheet_bindings"):
            raise ValueError(
                f"{port_surface.surface_id} requires an imprinted "
                f"sidewall child for {surface.surface_id}"
            )
        return [
            replace(
                surface,
                metadata=_route_b_port_sheet_partition_metadata(
                    surface.metadata,
                    binding=binding,
                    port_z_um=port_z_um,
                ),
            )
        ], True
    if not z_min_um < port_z_um < z_max_um:
        raise ValueError(
            f"{port_surface.surface_id} must cut {surface.surface_id} at its "
            "finite sidewall interior"
        )
    return _split_sidewall_quad(
        surface,
        port_surface=port_surface,
        quad=quad,
        port_z_um=port_z_um,
        binding=binding,
    ), True


def _split_sidewall_quad(
    surface: SurfacePlanRecord,
    *,
    port_surface: SurfacePlanRecord,
    quad: Sequence[tuple[float, float, float]],
    port_z_um: float,
    binding: Mapping[str, str],
) -> list[SurfacePlanRecord]:
    """Cut a sidewall quad at the port plane into a lower and an upper child surface."""
    lower_quad = (
        quad[0],
        quad[1],
        (quad[2][0], quad[2][1], port_z_um),
        (quad[3][0], quad[3][1], port_z_um),
    )
    upper_quad = (
        (quad[0][0], quad[0][1], port_z_um),
        (quad[1][0], quad[1][1], port_z_um),
        quad[2],
        quad[3],
    )
    parent_surface_id = surface.parent_surface_id or surface.surface_id
    partition_metadata = _route_b_port_sheet_partition_metadata(
        surface.metadata,
        binding=binding,
        port_z_um=port_z_um,
    )
    return [
        replace(
            surface,
            surface_id=f"{surface.surface_id}__{port_surface.surface_id}__LOWER",
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
            surface_id=f"{surface.surface_id}__{port_surface.surface_id}__UPPER",
            parent_surface_id=parent_surface_id,
            partition_label="port_upper",
            geometry_ref={
                **dict(surface.geometry_ref),
                "quad_points": upper_quad,
            },
            metadata=partition_metadata,
        ),
    ]


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
            for curve_ref in loops_by_id[loop_id].curve_refs
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
        for overlap_start, overlap_end in ring_edges(normalized_loop(overlap_loop))
    )
