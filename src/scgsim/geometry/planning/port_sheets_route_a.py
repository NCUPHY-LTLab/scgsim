"""Route A port sheets: carve the port sheet out of its host plane.

In Route A the port is an embedded sheet, so the host surface it sits on is cut around
it and the remainder keeps the host's ownership. See ``docs/geometry-sgb.qmd``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any

from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.entity_validation import (
    _is_solution_entity,
    _is_vacuum_solution_entity,
)
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _gdstk_surface_region,
    _geometry_refs_from_gdstk_region,
    _polygon_area,
)
from scgsim.geometry._primitives.spatial import _geometry_ref_surface_z_um, _same_z
from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.topology import _surface_owner_ids


def _route_a_port_sheet_sheet_contract(
    host_ids: Sequence[str],
    planned_surfaces: Sequence[SurfacePlanRecord],
    overlaps: Sequence[Any],
) -> tuple[float, tuple[str, str]]:
    """Select the local Route-A child intersecting each exact host footprint."""
    import gdstk

    host_planes: list[float] = []
    boundary_volume_ids: tuple[str, str] | None = None
    for host_id in host_ids:
        host_overlaps = tuple(
            overlap for overlap in overlaps if overlap.host_semantic_id == host_id
        )
        if len(host_overlaps) != 1:
            raise ValueError(
                f"{host_id} requires exactly one Route A port overlap record"
            )
        overlap_region = (gdstk.Polygon(_clean_loop(host_overlaps[0].overlap_loop)),)
        candidates = tuple(
            surface
            for surface in planned_surfaces
            if surface.metadata.get("representation") == "surface_sheet"
            # Route-A contact caps are per bump/face contact and bound a
            # single solution volume.  A layout junction sheet replaces the
            # actual two-volume face interface, never one of those caps.
            and surface.surface_role == "A_planned_interface"
            and host_id in _surface_owner_ids(surface)
        )
        host_sheets = tuple(
            surface
            for surface in candidates
            if _boolean_gdstk_region(
                gdstk,
                overlap_region,
                _gdstk_surface_region(surface.geometry_ref),
                "and",
            )
        )
        if len(host_sheets) != 1:
            raise ValueError(
                f"{host_id} port overlap intersects {len(host_sheets)} local "
                "Route A surface_sheet patches"
            )
        if _boolean_gdstk_region(
            gdstk,
            overlap_region,
            _gdstk_surface_region(host_sheets[0].geometry_ref),
            "not",
        ):
            raise ValueError(
                f"{host_id} port overlap crosses incompatible Route A domains"
            )
        plane_z_um = _geometry_ref_surface_z_um(host_sheets[0].geometry_ref)
        if not isfinite(plane_z_um):
            raise ValueError(f"{host_id} Route A surface_sheet plane must be finite")
        host_planes.append(plane_z_um)
        raw_boundary_ids = host_sheets[0].metadata.get("boundary_volume_ids")
        if isinstance(raw_boundary_ids, str):
            candidate_boundary_ids = ()
        else:
            candidate_boundary_ids = tuple(str(value) for value in raw_boundary_ids)
        if len(candidate_boundary_ids) != 2 or len(set(candidate_boundary_ids)) != 2:
            raise ValueError(
                f"{host_id} Route A surface_sheet requires two boundary volumes"
            )
        if boundary_volume_ids is None:
            boundary_volume_ids = candidate_boundary_ids
        elif boundary_volume_ids != candidate_boundary_ids:
            raise ValueError(
                "Route A lumped-port hosts require common boundary volumes"
            )
    if not _same_z(host_planes[0], host_planes[1]):
        raise ValueError("Route A lumped-port hosts require one common sheet plane")
    if boundary_volume_ids is None:
        raise ValueError("Route A lumped-port hosts require boundary volumes")
    return host_planes[0], boundary_volume_ids


def _route_a_port_sheet_vacuum_volume_id(
    boundary_volume_ids: Sequence[str],
    entities: Mapping[str, SemanticEntitySpec],
) -> str:
    """Select the typed vacuum member of the actual Route-A sheet adjacency."""
    boundary_entities: list[SemanticEntitySpec] = []
    for volume_id in boundary_volume_ids:
        entity = entities.get(volume_id)
        if entity is None or not _is_solution_entity(entity):
            raise ValueError(
                "Route A lumped-port boundary volumes must be typed solution regions"
            )
        boundary_entities.append(entity)
    vacuum_ids = tuple(
        volume_id
        for volume_id, entity in zip(
            boundary_volume_ids, boundary_entities, strict=True
        )
        if _is_vacuum_solution_entity(entity)
    )
    if len(vacuum_ids) != 1:
        raise ValueError(
            "Route A lumped-port boundary volumes require exactly one vacuum "
            "and one non-vacuum solution region"
        )
    return vacuum_ids[0]


def _carve_route_a_port_sheet_from_host_plane(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
) -> list[SurfacePlanRecord]:
    """Replace the co-planar solution face under a Route-A port with its hole."""
    import gdstk

    port_region = _gdstk_surface_region(port_surface.geometry_ref)
    port_z_um = _geometry_ref_surface_z_um(port_surface.geometry_ref)
    result: list[SurfacePlanRecord] = []
    for surface in surfaces:
        if (
            surface.construction_only
            or surface.metadata.get("representation") == "surface_sheet"
            or "outer_loop" not in surface.geometry_ref
            or not _same_z(_geometry_ref_surface_z_um(surface.geometry_ref), port_z_um)
        ):
            result.append(surface)
            continue
        overlap = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            port_region,
            "and",
        )
        if not overlap:
            result.append(surface)
            continue
        remainder = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            port_region,
            "not",
        )
        refs = _geometry_refs_from_gdstk_region(surface.geometry_ref, remainder)
        parent_surface_id = surface.parent_surface_id or surface.surface_id
        for index, geometry_ref in enumerate(refs):
            result.append(
                replace(
                    surface,
                    surface_id=(
                        surface.surface_id
                        if len(refs) == 1
                        else f"{surface.surface_id}__PORT_REMAINDER_{index:04d}"
                    ),
                    geometry_ref=geometry_ref,
                    parent_surface_id=(
                        surface.parent_surface_id
                        if len(refs) == 1
                        else parent_surface_id
                    ),
                    partition_label=(
                        surface.partition_label
                        if len(refs) == 1
                        else f"PORT_REMAINDER_{index:04d}"
                    ),
                )
            )
    return result


def _port_sheet_remainder_geometry(region: PortSheetRegionRecord) -> dict[str, Any]:
    import gdstk

    authored = _gdstk_surface_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}
    )
    overlaps = tuple(
        gdstk.Polygon(_clean_loop(overlap.overlap_loop)) for overlap in region.overlaps
    )
    remainder = _boolean_gdstk_region(gdstk, authored, overlaps, "not")
    refs = _geometry_refs_from_gdstk_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}, remainder
    )
    if len(refs) != 1:
        raise ValueError(
            f"{region.port_sheet_id} requires exactly one finite active remainder"
        )
    result = refs[0]
    area = abs(_polygon_area(_clean_loop(result["outer_loop"]))) - sum(
        abs(_polygon_area(_clean_loop(hole))) for hole in result["hole_loops"]
    )
    if not isfinite(area) or area <= _TOPOLOGY_EPS_UM:
        raise ValueError(f"{region.port_sheet_id} active remainder must be finite")
    return result
