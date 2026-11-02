"""Resolve Route A sheet adjacency and carve the port from its host plane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any, cast

from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.topology import surface_owner_ids
from scgsim.geometry.primitives.entity_validation import (
    is_solution_entity,
    is_vacuum_solution_entity,
)
from scgsim.geometry.primitives.grid import boolean_polygon_region
from scgsim.geometry.primitives.loops import (
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
)
from scgsim.geometry.primitives.planar_loops import clean_loop
from scgsim.geometry.primitives.spatial import (
    geometry_ref_surface_z_um,
    same_coordinate_z,
)


def route_a_port_sheet_sheet_contract(
    host_ids: Sequence[str],
    planned_surfaces: Sequence[SurfacePlanRecord],
    overlaps: Sequence[Any],
) -> tuple[float, tuple[str, str]]:
    """Select the local Route-A child intersecting each exact host footprint."""
    import gdstk  # noqa: PLC0415 -- load the optional polygon engine only at this native geometry boundary.

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
        overlap_region = (gdstk.Polygon(clean_loop(host_overlaps[0].overlap_loop)),)
        candidates = tuple(
            surface
            for surface in planned_surfaces
            if surface.metadata.get("representation") == "surface_sheet"
            # Route-A contact caps are per bump/face contact and bound a
            # single solution volume.  A layout junction sheet replaces the
            # actual two-volume face interface, never one of those caps.
            and surface.surface_role == "A_planned_interface"
            and host_id in surface_owner_ids(surface)
        )
        host_sheets = tuple(
            surface
            for surface in candidates
            if boolean_polygon_region(
                gdstk,
                overlap_region,
                gdstk_surface_region(surface.geometry_ref),
                "and",
            )
        )
        if len(host_sheets) != 1:
            raise ValueError(
                f"{host_id} port overlap intersects {len(host_sheets)} local "
                "Route A surface_sheet patches"
            )
        if boolean_polygon_region(
            gdstk,
            overlap_region,
            gdstk_surface_region(host_sheets[0].geometry_ref),
            "not",
        ):
            raise ValueError(
                f"{host_id} port overlap crosses incompatible Route A domains"
            )
        plane_z_um = geometry_ref_surface_z_um(host_sheets[0].geometry_ref)
        if not isfinite(plane_z_um):
            raise ValueError(f"{host_id} Route A surface_sheet plane must be finite")
        host_planes.append(plane_z_um)
        raw_boundary_ids = cast(
            Sequence[str] | str, host_sheets[0].metadata.get("boundary_volume_ids")
        )
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
                "Route A lumped-port hosts require common boundary volumes: "
                f"{boundary_volume_ids!r} != {candidate_boundary_ids!r}"
            )
    if not same_coordinate_z(host_planes[0], host_planes[1]):
        raise ValueError("Route A lumped-port hosts require one common sheet plane")
    if boundary_volume_ids is None:
        raise ValueError("Route A lumped-port hosts require boundary volumes")
    return host_planes[0], boundary_volume_ids


def route_a_port_sheet_vacuum_volume_id(
    boundary_volume_ids: Sequence[str],
    entities: Mapping[str, SemanticEntitySpec],
) -> str:
    """Select the typed vacuum member of the actual Route-A sheet adjacency."""
    boundary_entities: list[SemanticEntitySpec] = []
    for volume_id in boundary_volume_ids:
        entity = entities.get(volume_id)
        if entity is None or not is_solution_entity(entity):
            raise ValueError(
                "Route A lumped-port boundary volumes must be typed solution regions"
            )
        boundary_entities.append(entity)
    vacuum_ids = tuple(
        volume_id
        for volume_id, entity in zip(
            boundary_volume_ids, boundary_entities, strict=True
        )
        if is_vacuum_solution_entity(entity)
    )
    if len(vacuum_ids) != 1:
        raise ValueError(
            "Route A lumped-port boundary volumes require exactly one vacuum "
            "and one non-vacuum solution region"
        )
    return vacuum_ids[0]


def carve_route_a_port_sheet_from_host_plane(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
) -> list[SurfacePlanRecord]:
    """Replace the co-planar solution face under a Route-A port with its hole."""
    import gdstk  # noqa: PLC0415 -- load the optional polygon engine only at this native geometry boundary.

    port_region = gdstk_surface_region(port_surface.geometry_ref)
    port_z_um = geometry_ref_surface_z_um(port_surface.geometry_ref)
    result: list[SurfacePlanRecord] = []
    for surface in surfaces:
        if (
            surface.construction_only
            or surface.metadata.get("representation") == "surface_sheet"
            or "outer_loop" not in surface.geometry_ref
            or not same_coordinate_z(
                geometry_ref_surface_z_um(surface.geometry_ref), port_z_um
            )
        ):
            result.append(surface)
            continue
        overlap = boolean_polygon_region(
            gdstk,
            gdstk_surface_region(surface.geometry_ref),
            port_region,
            "and",
        )
        if not overlap:
            result.append(surface)
            continue
        remainder = boolean_polygon_region(
            gdstk,
            gdstk_surface_region(surface.geometry_ref),
            port_region,
            "not",
        )
        refs = geometry_refs_from_gdstk_region(surface.geometry_ref, remainder)
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
