"""Replace the authored auto vacuum region with planner-side complement components.

An auto VACUUM_REGION is an authored envelope plus padding. Planning sweeps it along
z, subtracts every solid obstacle slice by slice, and emits one vacuum component per
connected remainder. Does not resolve which solution touches which conductor face;
see ``solution_adjacency``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import isfinite
from typing import Any, NamedTuple

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.context.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.planning.solutions.occupancy import (
    auto_vacuum_subtractor_xy_region,
    domain_bounds_loop,
    loop_domain_bounds,
    solution_entity_xy_region,
)
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entities import entity_z_range_um
from scgsim.geometry.primitives.entity_validation import is_solution_entity
from scgsim.geometry.primitives.loops import (
    boolean_gdstk_region,
    gdstk_surface_region,
    geometry_refs_from_gdstk_region,
    loop_signature,
    loops_share_edge_overlap,
    normalized_loop,
)
from scgsim.geometry.primitives.spatial import is_same_z


def prepare_auto_vacuum_solution_regions(
    build_input: GeometryBuildInput,
    route: RouteLiteral | None,
    *,
    native_complement: bool = False,
    representation_context: RepresentationContext | None = None,
) -> GeometryBuildInput:
    """Replace auto VACUUM_REGION with planner-side complement components.

    The auto region is only an envelope. Planning (1) pads the envelope, (2) cuts the
    z range at every obstacle top and bottom, (3) subtracts the obstacles' xy
    footprints inside each z slice, and (4) emits one vacuum component per connected
    remainder, so that vacuum never overlaps a solid.
    """
    auto_region = _auto_vacuum_solution_region(build_input)
    if auto_region is None:
        return build_input
    if auto_region.material_kind != "vacuum":
        raise ValueError(
            "auto vacuum region must have vacuum material_kind before planning"
        )

    auto_metadata = dict(auto_region.metadata)
    auto_bounds = _padded_auto_vacuum_bounds(
        build_input,
        route=route,
        auto_region=auto_region,
        representation_context=representation_context,
    )
    auto_region = replace(
        auto_region,
        geometry={
            **dict(auto_region.geometry),
            "domain_bounds_um": {
                "x_min_um": auto_bounds["x_min_um"],
                "y_min_um": auto_bounds["y_min_um"],
                "x_max_um": auto_bounds["x_max_um"],
                "y_max_um": auto_bounds["y_max_um"],
            },
            "z_min_um": auto_bounds["z_min_um"],
            "z_max_um": auto_bounds["z_max_um"],
            "outer_loop": domain_bounds_loop(auto_bounds),
            "hole_loops": (),
        },
    )
    envelope_loop = domain_bounds_loop(auto_bounds)
    if native_complement:
        # The curved compiler retains the same authored envelope and padding,
        # but computes its complement in the shared native planar arrangement.
        return replace(
            build_input,
            entities=tuple(
                auto_region if entity.semantic_id == auto_region.semantic_id else entity
                for entity in build_input.entities
            ),
        )
    auto_z_min_um = float(auto_bounds["z_min_um"])
    auto_z_max_um = float(auto_bounds["z_max_um"])

    all_entities = tuple(build_input.entities)
    obstacle_entities = tuple(
        entity
        for entity in all_entities
        if _is_auto_vacuum_subtractor(
            entity,
            route=route,
            representation_context=representation_context,
        )
        and entity.semantic_id != auto_region.semantic_id
        and not bool(entity.metadata.get("is_auto_vacuum_region"))
    )
    z_slices = _auto_vacuum_z_slices(auto_z_min_um, auto_z_max_um, obstacle_entities)

    import gdstk

    sweep = _VacuumSweep(
        build_input=build_input,
        auto_region=auto_region,
        auto_metadata=auto_metadata,
        auto_bounds=auto_bounds,
        envelope_loop=envelope_loop,
        obstacle_entities=obstacle_entities,
    )
    components: list[SemanticEntitySpec] = []
    component_index = 0
    for z_start, z_end in pairwise(z_slices):
        if z_end - z_start <= TOPOLOGY_EPS_UM:
            continue
        if (
            z_start < auto_z_min_um - TOPOLOGY_EPS_UM
            or z_end > auto_z_max_um + TOPOLOGY_EPS_UM
        ):
            continue
        for piece in _vacuum_pieces_in_slice(sweep, gdstk, z_start, z_end):
            component_index += 1
            components.append(
                _vacuum_component(sweep, piece, component_index, z_start, z_end)
            )

    if not components:
        raise ValueError("auto VACUUM_REGION leaves no finite sweep vacuum geometry")

    retained_entities = tuple(
        entity
        for entity in all_entities
        if entity.semantic_id != auto_region.semantic_id
    )
    return replace(
        build_input,
        entities=tuple(components) + tuple(retained_entities),
        solution_regions=build_input.solution_regions,
    )


def _padded_auto_vacuum_bounds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    auto_region: SemanticEntitySpec,
    representation_context: RepresentationContext | None,
) -> dict[str, float]:
    """Return the auto region's xyz bounds: obstacle envelope plus authored padding."""
    auto_padding = _auto_vacuum_padding(
        dict(auto_region.metadata), auto_region.semantic_id
    )
    enclosure = (
        representation_context.enclosures.get(auto_region.semantic_id)
        if representation_context is not None
        else None
    )
    if route is None and enclosure is not None:
        # The prepared context owns these complete effective bounds. Source
        # padding remains provenance and is not mapped a second time here.
        auto_bounds = dict(enclosure.effective_bounds_um)
    else:
        auto_bounds = auto_vacuum_envelope_bounds(
            build_input,
            route=route,
            representation_context=representation_context,
        )
        auto_bounds = {
            "x_min_um": auto_bounds["x_min_um"] - auto_padding["x_minus_um"],
            "y_min_um": auto_bounds["y_min_um"] - auto_padding["y_minus_um"],
            "x_max_um": auto_bounds["x_max_um"] + auto_padding["x_plus_um"],
            "y_max_um": auto_bounds["y_max_um"] + auto_padding["y_plus_um"],
            "z_min_um": auto_bounds["z_min_um"] - auto_padding["z_minus_um"],
            "z_max_um": auto_bounds["z_max_um"] + auto_padding["z_plus_um"],
        }
    if not auto_bounds["x_min_um"] < auto_bounds["x_max_um"]:
        raise ValueError("auto VACUUM_REGION has non-positive padded x extent")
    if not auto_bounds["y_min_um"] < auto_bounds["y_max_um"]:
        raise ValueError("auto VACUUM_REGION has non-positive padded y extent")
    if not auto_bounds["z_min_um"] < auto_bounds["z_max_um"]:
        raise ValueError("auto VACUUM_REGION has non-positive padded z extent")
    return auto_bounds


def _auto_vacuum_z_slices(
    auto_z_min_um: float,
    auto_z_max_um: float,
    obstacle_entities: Sequence[SemanticEntitySpec],
) -> list[float]:
    """Return the sorted z levels at which the vacuum footprint can change.

    Between two consecutive levels no obstacle starts or ends, so one xy subtraction
    is valid for the whole slice.
    """
    z_events = {auto_z_min_um, auto_z_max_um}
    for entity in obstacle_entities:
        entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
        if not (
            isfinite(entity_z_min_um)
            and isfinite(entity_z_max_um)
            and entity_z_max_um > entity_z_min_um
        ):
            raise ValueError(f"{entity.semantic_id} has non-positive z extent")
        z_events.update((entity_z_min_um, entity_z_max_um))
    z_slices = sorted(z_events)
    if len(z_slices) < 2:
        raise ValueError("auto VACUUM_REGION has no valid z sweep")
    return z_slices


@dataclass(frozen=True)
class _VacuumSweep:
    """The fixed inputs of one auto vacuum sweep along z."""

    build_input: GeometryBuildInput
    auto_region: SemanticEntitySpec
    auto_metadata: Mapping[str, Any]
    auto_bounds: Mapping[str, float]
    envelope_loop: Any
    obstacle_entities: Sequence[SemanticEntitySpec]


class _VacuumPiece(NamedTuple):
    """One connected vacuum remainder inside a z slice, with its touching obstacles."""

    geometry_ref: Mapping[str, Any]
    # Obstacles whose xy footprint touches this piece's boundary.
    subtractor_ids: tuple[str, ...]
    # Obstacles that sit directly below ("bottom") or above ("top") the slice.
    boundary_entity_ids: Mapping[str, list[str]]


def _vacuum_pieces_in_slice(
    sweep: _VacuumSweep,
    gdstk: Any,
    z_start: float,
    z_end: float,
) -> list[_VacuumPiece]:
    """Subtract the obstacles active in one z slice from the envelope footprint."""
    build_input = sweep.build_input
    obstacle_entities = sweep.obstacle_entities
    base_region = solution_entity_xy_region(
        gdstk,
        sweep.auto_region,
        z_min_um=z_start,
        z_max_um=z_end,
    )
    if not base_region:
        return []

    active_subtractor_region: tuple[Any, ...] = ()
    active_subtractors: list[SemanticEntitySpec] = []
    for entity in obstacle_entities:
        obstacle_z_min_um, obstacle_z_max_um = entity_z_range_um(entity)
        if (
            obstacle_z_min_um >= z_end - TOPOLOGY_EPS_UM
            or obstacle_z_max_um <= z_start + TOPOLOGY_EPS_UM
        ):
            continue
        subtractor_regions = auto_vacuum_subtractor_xy_region(
            gdstk, build_input, entity
        )
        if not subtractor_regions:
            continue
        active_subtractors.append(entity)
        if active_subtractor_region:
            active_subtractor_region = boolean_gdstk_region(
                gdstk,
                active_subtractor_region,
                subtractor_regions,
                "or",
            )
        else:
            active_subtractor_region = subtractor_regions

    vacuum_region_refs = geometry_refs_from_gdstk_region(
        {"outer_loop": sweep.envelope_loop},
        boolean_gdstk_region(
            gdstk,
            base_region,
            active_subtractor_region,
            "not",
        ),
    )
    pieces: list[_VacuumPiece] = []
    for geometry_ref in sorted(
        vacuum_region_refs,
        key=lambda region: loop_signature(region["outer_loop"]),
    ):
        component_region = gdstk_surface_region(geometry_ref)
        component_subtractor_ids = tuple(
            sorted(
                entity.semantic_id
                for entity in active_subtractors
                if _auto_vacuum_component_contacts_subtractor(
                    gdstk,
                    build_input=build_input,
                    component_geometry_ref=geometry_ref,
                    subtractor=entity,
                )
            )
        )
        boundary_entity_ids: dict[str, list[str]] = {"bottom": [], "top": []}
        for entity in obstacle_entities:
            entity_z_min_um, entity_z_max_um = entity_z_range_um(entity)
            boundary_key: str | None = None
            if is_same_z(entity_z_max_um, z_start):
                boundary_key = "bottom"
            elif is_same_z(entity_z_min_um, z_end):
                boundary_key = "top"
            if boundary_key is None:
                continue
            entity_region = auto_vacuum_subtractor_xy_region(gdstk, build_input, entity)
            if boolean_gdstk_region(
                gdstk,
                component_region,
                entity_region,
                "and",
            ):
                boundary_entity_ids[boundary_key].append(entity.semantic_id)
        pieces.append(
            _VacuumPiece(geometry_ref, component_subtractor_ids, boundary_entity_ids)
        )
    return pieces


def _vacuum_component(
    sweep: _VacuumSweep,
    piece: _VacuumPiece,
    component_index: int,
    z_start: float,
    z_end: float,
) -> SemanticEntitySpec:
    """Build the vacuum entity for one piece. The first keeps the authored id."""
    auto_region = sweep.auto_region
    geometry_ref = piece.geometry_ref
    component_id = (
        auto_region.semantic_id
        if component_index == 1
        else f"{auto_region.semantic_id}__{component_index:04d}"
    )
    return SemanticEntitySpec(
        semantic_id=component_id,
        role=auto_region.role,
        material_id=auto_region.material_id,
        material_kind=auto_region.material_kind,
        priority=auto_region.priority,
        geometry_kind=auto_region.geometry_kind,
        host_void_semantic_id=auto_region.host_void_semantic_id,
        route_representations=auto_region.route_representations,
        geometry={
            "geometry_kind": auto_region.geometry.get(
                "geometry_kind",
                auto_region.geometry_kind,
            ),
            "outer_loop": geometry_ref["outer_loop"],
            "hole_loops": geometry_ref.get("hole_loops", ()),
            "z_min_um": float(z_start),
            "z_max_um": float(z_end),
            "domain_bounds_um": loop_domain_bounds(
                geometry_ref["outer_loop"],
                geometry_ref.get("hole_loops", ()),
            ),
            "from_soln_region": auto_region.semantic_id,
        },
        metadata={
            **sweep.auto_metadata,
            "is_auto_vacuum_region": True,
            "auto_vacuum_group_id": auto_region.semantic_id,
            "auto_vacuum_envelope_outer_loop": sweep.envelope_loop,
            "auto_vacuum_envelope_bounds_um": dict(sweep.auto_bounds),
            "auto_vacuum_component_index": component_index,
            "auto_vacuum_z_range_um": (float(z_start), float(z_end)),
            "auto_vacuum_subtracting_entity_ids": tuple(piece.subtractor_ids),
            "auto_vacuum_boundary_entity_ids": {
                boundary_key: tuple(sorted(set(entity_ids)))
                for boundary_key, entity_ids in piece.boundary_entity_ids.items()
            },
        },
        polygon_ids=(),
        labels=(),
    )


def _auto_vacuum_solution_region(
    build_input: GeometryBuildInput,
) -> SemanticEntitySpec | None:
    candidates = tuple(
        entity
        for entity in build_input.entities
        if is_solution_entity(entity)
        and bool(entity.metadata.get("is_auto_vacuum_region"))
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError("exactly one auto VACUUM_REGION solution entity is required")
    return candidates[0]


def _auto_vacuum_padding(
    auto_metadata: Mapping[str, Any],
    auto_region_id: str,
) -> dict[str, float]:
    raw = auto_metadata.get("vacuum_region_padding_um")
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{auto_region_id} requires metadata vacuum_region_padding_um for auto envelope."
        )
    required = (
        "x_minus_um",
        "x_plus_um",
        "y_minus_um",
        "y_plus_um",
        "z_minus_um",
        "z_plus_um",
    )
    values = {}
    for key in required:
        value = raw.get(key)
        if (
            value is None
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"{auto_region_id} vacuum padding {key!r} must be a finite non-negative number."
            )
        value = float(value)
        if not isfinite(value) or value < 0.0:
            raise ValueError(
                f"{auto_region_id} vacuum padding {key!r} must be a finite non-negative number."
            )
        values[key] = value
    return values


def auto_vacuum_envelope_bounds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> dict[str, float]:
    """XYZ bounding box in micrometres over subtractor entities and Route-A sheets."""
    bounds: list[dict[str, float]] = []
    for entity in build_input.entities:
        is_subtractor = _is_auto_vacuum_subtractor(
            entity, route=route, representation_context=representation_context
        )
        include_for_xy = _auto_vacuum_surface_sheet_for_xy_envelope(
            entity, route, representation_context=representation_context
        )
        if not is_subtractor and not include_for_xy:
            continue
        if bool(entity.metadata.get("is_auto_vacuum_region")):
            continue
        xy_bounds = _auto_vacuum_entity_xy_bounds(build_input, entity)
        entry = {
            **xy_bounds,
            "z_min_um": float("nan"),
            "z_max_um": float("nan"),
        }
        if is_subtractor:
            z_min_um, z_max_um = entity_z_range_um(entity)
            if not (isfinite(z_min_um) and isfinite(z_max_um) and z_max_um > z_min_um):
                raise ValueError(
                    f"{entity.semantic_id} has non-positive or non-finite z extent."
                )
            entry["z_min_um"] = float(z_min_um)
            entry["z_max_um"] = float(z_max_um)
        elif include_for_xy and not is_subtractor:
            sheet_z_um = auto_vacuum_sheet_z(
                build_input, entity, representation_context=representation_context
            )
            entry["z_min_um"] = sheet_z_um
            entry["z_max_um"] = sheet_z_um

        bounds.append(entry)

    if not bounds:
        raise ValueError(
            "Cannot auto-compute vacuum envelope without route-aware positive occupancy."
        )

    finite_z_bounds = tuple(bound for bound in bounds if isfinite(bound["z_min_um"]))
    if not finite_z_bounds:
        raise ValueError(
            "Cannot auto-compute vacuum envelope without route-aware finite z occupancy."
        )
    return {
        "x_min_um": min(item["x_min_um"] for item in bounds),
        "x_max_um": max(item["x_max_um"] for item in bounds),
        "y_min_um": min(item["y_min_um"] for item in bounds),
        "y_max_um": max(item["y_max_um"] for item in bounds),
        "z_min_um": min(item["z_min_um"] for item in finite_z_bounds),
        "z_max_um": max(item["z_max_um"] for item in finite_z_bounds),
    }


def _auto_vacuum_entity_xy_bounds(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> dict[str, float]:
    """Return the xy bounds of an entity, from the first geometry source it has.

    The sources, in priority order, are ``domain_bounds_um``, ``outer_loop`` (with holes)
    and finally the entity's layout polygons.
    """
    geometry = entity.geometry
    domain_bounds = geometry.get("domain_bounds_um")
    if isinstance(domain_bounds, Mapping):
        return _xy_bounds_from_domain_bounds(entity, domain_bounds)
    if "outer_loop" in geometry:
        return _xy_bounds_from_loops(entity)
    return _xy_bounds_from_polygons(build_input, entity)


def _xy_bounds_from_domain_bounds(
    entity: SemanticEntitySpec, domain_bounds: Mapping[str, Any]
) -> dict[str, float]:
    """Read and validate explicit ``domain_bounds_um``."""
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    missing = [name for name in required if name not in domain_bounds]
    if missing:
        raise TypeError(
            f"{entity.semantic_id} has missing {tuple(missing)} in domain_bounds_um for auto vacuum envelope."
        )
    values = [domain_bounds.get(name) for name in required]
    if any(not isfinite(float(value)) for value in values):
        raise ValueError(
            f"{entity.semantic_id} has non-finite domain_bounds_um for auto vacuum envelope."
        )
    x_min_um, x_max_um, y_min_um, y_max_um = (float(value) for value in values)
    if not (x_min_um < x_max_um and y_min_um < y_max_um):
        raise ValueError(
            f"{entity.semantic_id} has non-positive XY extent for auto vacuum envelope."
        )
    return {
        "x_min_um": x_min_um,
        "x_max_um": x_max_um,
        "y_min_um": y_min_um,
        "y_max_um": y_max_um,
    }


def _xy_bounds_from_loops(entity: SemanticEntitySpec) -> dict[str, float]:
    """Bounds of the entity's ``outer_loop`` and ``hole_loops``."""
    geometry = entity.geometry
    outer_loop = normalized_loop(geometry["outer_loop"])
    hole_loops = tuple(normalized_loop(loop) for loop in geometry.get("hole_loops", ()))
    if not all(
        isfinite(coordinate)
        for loop in (outer_loop, *hole_loops)
        for point in loop
        for coordinate in point
    ):
        raise ValueError(
            f"{entity.semantic_id} has non-finite loop geometry for auto vacuum envelope."
        )
    return loop_domain_bounds(outer_loop, hole_loops)


def _xy_bounds_from_polygons(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> dict[str, float]:
    """Bounds of the layout polygons the entity references, holes included."""
    polygon_ids = tuple(entity.polygon_ids)
    if not polygon_ids:
        raise ValueError(
            f"{entity.semantic_id} must define domain_bounds_um, outer_loop, or polygon_ids for auto vacuum envelope."
        )
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    points: list[tuple[float, float]] = []
    for polygon_id in polygon_ids:
        try:
            polygon = polygons[polygon_id]
        except KeyError as exc:
            raise ValueError(
                f"{entity.semantic_id} references an unknown layout polygon for auto vacuum envelope."
            ) from exc
        loop = normalized_loop(polygon.exterior)
        if len(loop) < 3:
            raise ValueError(
                f"{entity.semantic_id} references degenerate polygon {polygon_id!r} for auto vacuum envelope."
            )
        points.extend(loop)
        for hole in polygon.holes:
            hole_loop = normalized_loop(hole)
            if not all(
                isfinite(coordinate) for point in hole_loop for coordinate in point
            ):
                raise ValueError(
                    f"{entity.semantic_id} references non-finite polygon {polygon_id!r} for auto vacuum envelope."
                )
            points.extend(hole_loop)
    if not points:
        raise ValueError(
            f"{entity.semantic_id} has no geometry points for auto vacuum envelope."
        )
    if not all(isfinite(coordinate) for point in points for coordinate in point):
        raise ValueError(
            f"{entity.semantic_id} has non-finite polygon geometry for auto vacuum envelope."
        )
    return {
        "x_min_um": min(point[0] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "y_max_um": max(point[1] for point in points),
    }


def _auto_vacuum_surface_sheet_for_xy_envelope(
    entity: SemanticEntitySpec,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    return (
        route == "A" and entity.route_representations.get(route) == "surface_sheet"
    ) or (
        route is None
        and planned_conductor_representation(
            entity, route=route, context=representation_context
        )
        == "surface_sheet"
    )


def auto_vacuum_sheet_z(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    representation_context: RepresentationContext | None = None,
) -> float:
    """Z in micrometres of an auto-vacuum Route-A sheet, from context or provenance."""
    context_record = (
        representation_context.entities.get(entity.semantic_id)
        if representation_context is not None
        else None
    )
    if context_record is not None and context_record.representation == "sheet":
        return float(context_record.effective_z_interval_um[0])
    provenance = build_input.metadata.get("route_a_thin_film")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "auto VACUUM_REGION Route-A sheets require thin-film provenance"
        )
    ranges = provenance.get("physical_face_metal_z_ranges_um")
    sheet_positions = provenance.get("effective_sheet_z_um")
    if not isinstance(ranges, Mapping) or not isinstance(sheet_positions, Mapping):
        raise ValueError("auto VACUUM_REGION Route-A sheet provenance is incomplete")
    source_id = entity.metadata.get(
        "source_semantic_id",
        entity.metadata.get("semantic_group_id", entity.semantic_id),
    )
    matches = (
        side
        for side, record in ranges.items()
        if isinstance(record, Mapping) and source_id in record.get("semantic_ids", ())
    )
    sides = tuple(matches)
    if len(sides) != 1 or not isfinite(
        float(sheet_positions.get(sides[0], float("nan")))
    ):
        raise ValueError(
            f"{entity.semantic_id} has no unique effective Route-A sheet position"
        )
    return float(sheet_positions[sides[0]])


def _is_auto_vacuum_subtractor(
    entity: SemanticEntitySpec,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    if bool(entity.metadata.get("is_auto_vacuum_region")):
        return False
    if is_solution_entity(entity):
        return True

    representation = planned_conductor_representation(
        entity, route=route, context=representation_context
    )
    return representation in {"cutout_boundary_shell", "material_volume"}


def _auto_vacuum_component_contacts_subtractor(
    gdstk: Any,
    *,
    build_input: GeometryBuildInput,
    component_geometry_ref: Mapping[str, Any],
    subtractor: SemanticEntitySpec,
) -> bool:
    """Whether a subtractor owns an exact boundary on one vacuum component.

    A z-slab can have multiple disconnected complement components.  The ledger
    must therefore record only a subtractor whose canonical planar boundary
    shares a nonzero segment with this component, rather than every obstacle
    active in the slab.  This remains geometry/topology bookkeeping; semantic
    identity comes from the pre-existing structured entity record.
    """
    subtractor_region = auto_vacuum_subtractor_xy_region(gdstk, build_input, subtractor)
    if not subtractor_region:
        return False
    subtractor_refs = geometry_refs_from_gdstk_region(
        {},
        subtractor_region,
    )
    component_loops = (
        normalized_loop(component_geometry_ref["outer_loop"]),
        *(
            normalized_loop(loop)
            for loop in component_geometry_ref.get("hole_loops", ())
        ),
    )
    return any(
        loops_share_edge_overlap(component_loop, subtractor_loop)
        for component_loop in component_loops
        for subtractor_ref in subtractor_refs
        for subtractor_loop in (
            normalized_loop(subtractor_ref["outer_loop"]),
            *(normalized_loop(loop) for loop in subtractor_ref.get("hole_loops", ())),
        )
    )
