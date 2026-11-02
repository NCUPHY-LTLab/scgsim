"""Read source occupancy and exact polygon coverage for solution and conductor entities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, Any

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.primitives.entities import entity_by_id, entity_z_range_um
from scgsim.geometry.primitives.entity_validation import is_solution_entity
from scgsim.geometry.primitives.grid import (
    boolean_polygon_region,
    polygon_boolean,
    split_gdstk_cutline_loop,
)
from scgsim.geometry.primitives.loops import (
    geometry_refs_from_gdstk_region,
    loops_share_edge_overlap,
)
from scgsim.geometry.primitives.planar_loops import clean_loop, filter_gdstk_polygons
from scgsim.geometry.primitives.spatial import range_within_bounds, same_coordinate_z
from scgsim.semantics.route_a import geometry_z_range, record_geometry, same_z

if TYPE_CHECKING:
    from gdstk import Polygon


def verified_route_a_substrate_support(
    build_input: GeometryBuildInput, stack: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Bind each face metal to one positive-area dielectric contact plane.

    The normalized face polygons and modeled substrate region (rectangle or
    explicit outer/hole loops) are intersected as regions. A metal overhang
    remains exposed; neither bounds nor a point/edge touch establishes contact.
    """
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    layers = stack.get("layers")
    if isinstance(layers, str | bytes) or not isinstance(layers, Sequence):
        raise TypeError("Route A requires a sequence of stack layers.")
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    substrates = tuple(
        (
            entity
            for entity in build_input.entities
            if is_solution_entity(entity) and entity.material_kind == "dielectric"
        )
    )
    result: dict[str, dict[str, str]] = {}
    for record in layers:
        if not isinstance(record, Mapping):
            raise TypeError("Route A stack layers must contain mappings.")
        if record.get("part_role") != "face_metal":
            continue
        source_id = record.get("semantic_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("face_metal requires semantic_id for substrate support.")
        face_z_min, face_z_max = geometry_z_range(
            record_geometry(record, source_id), source_id
        )
        if face_z_max <= face_z_min:
            raise ValueError(f"{source_id} physical face-metal thickness must be > 0.")
        face_entities = tuple(
            (
                entity
                for entity in build_input.entities
                if entity.semantic_id == source_id
                or entity.metadata.get("semantic_group_id") == source_id
                or entity.metadata.get("source_semantic_id") == source_id
            )
        )
        face_regions = _face_metal_regions(gdstk, polygons, face_entities, source_id)
        if not face_regions:
            raise ValueError(
                f"{source_id} has no normalized face-metal polygon for substrate attachment."
            )
        matches: list[tuple[str, str]] = []
        for substrate in substrates:
            sub_z_min, sub_z_max = entity_z_range_um(substrate)
            sides = (
                *(("lower",) if same_z(sub_z_max, face_z_min) else ()),
                *(("upper",) if same_z(sub_z_min, face_z_max) else ()),
            )
            if not sides:
                continue
            substrate_region = solution_entity_xy_region(gdstk, substrate)
            if boolean_polygon_region(gdstk, face_regions, substrate_region, "and"):
                matches.extend(((side, substrate.semantic_id) for side in sides))
        if len(matches) != 1:
            raise ValueError(
                f"{source_id} requires exactly one positive-area dielectric substrate contact at its physical face; matched {matches!r} among {tuple((entity.semantic_id for entity in substrates))!r}."
            )
        side, substrate_id = matches[0]
        result[source_id] = {"side": side, "substrate_id": substrate_id}
    return result


def solution_entity_xy_region(
    gdstk: Any,
    entity: SemanticEntitySpec,
    *,
    z_min_um: float | None = None,
    z_max_um: float | None = None,
) -> tuple[Any, ...]:
    """Build the XY polygon region represented by a solution entity."""
    del z_min_um, z_max_um
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return _xy_region_from_loops(
            gdstk, geometry["outer_loop"], tuple(geometry.get("hole_loops", ()))
        )
    bounds = solution_bounds(entity)
    return (gdstk.Polygon(domain_bounds_loop(bounds)),)


def auto_vacuum_subtractor_xy_region(
    gdstk: Any, build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> tuple[Any, ...]:
    """Use the normalized polygons for a source entity lacking a planned loop."""
    if is_solution_entity(entity) or "outer_loop" in entity.geometry:
        return solution_entity_xy_region(gdstk, entity)
    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    region: tuple[Any, ...] = ()
    for polygon_id in entity.polygon_ids:
        polygon = polygons_by_id[polygon_id]
        polygon_region = _xy_region_from_loops(gdstk, polygon.exterior, polygon.holes)
        region = (
            polygon_region
            if not region
            else boolean_polygon_region(gdstk, region, polygon_region, "or")
        )
    if not entity.polygon_ids:
        return solution_entity_xy_region(gdstk, entity)
    return region


def _xy_region_from_loops(
    gdstk: Any, outer_loop: Any, hole_loops: Sequence[Any]
) -> tuple[Any, ...]:
    outer = clean_loop(outer_loop)
    split_outer, *split_holes = split_gdstk_cutline_loop(outer)
    holes = tuple(
        (
            clean_loop(loop)
            for loop in tuple(hole_loops) + tuple(split_holes)
            if len(loop) >= 3
        )
    )
    if not holes:
        return (gdstk.Polygon(split_outer),)
    return filter_gdstk_polygons(
        polygon_boolean(
            (gdstk.Polygon(split_outer),),
            tuple((gdstk.Polygon(hole_loop) for hole_loop in holes)),
            "not",
            precision=1e-09,
        )
    )


def loop_domain_bounds(
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> dict[str, float]:
    """Compute XY bounds across the outer and hole rings."""
    points = tuple(
        (point for loop in (outer_loop, *hole_loops) for point in clean_loop(loop))
    )
    return {
        "x_min_um": min((point[0] for point in points)),
        "y_min_um": min((point[1] for point in points)),
        "x_max_um": max((point[0] for point in points)),
        "y_max_um": max((point[1] for point in points)),
    }


def active_route_conductor_entities(
    build_input: GeometryBuildInput,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> tuple[SemanticEntitySpec, ...]:
    """Select conductor entities active in the chosen route representation."""
    if route is None and representation_context is not None:
        return route_conductor_contact_footprints(
            build_input, route, representation_context=representation_context
        )
    return tuple(
        (
            entity
            for entity in build_input.entities
            if not is_solution_entity(entity)
            and planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            is not None
            and ("outer_loop" in entity.geometry)
        )
    )


def route_conductor_contact_footprints(
    build_input: GeometryBuildInput,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> tuple[SemanticEntitySpec, ...]:
    """Recognize effective contacts on each declared source polygon.

    Temporary footprints retain parent Entity identity and original holes;
    they neither replace source bodies nor invent a bounding exterior.
    """
    if route is not None:
        return active_route_conductor_entities(
            build_input, route, representation_context=representation_context
        )
    if representation_context is None:
        return ()
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    footprints: list[SemanticEntitySpec] = []
    for entity in build_input.entities:
        if (
            is_solution_entity(entity)
            or planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            is None
        ):
            continue
        if "outer_loop" in entity.geometry:
            footprints.append(entity)
            continue
        for polygon_id in entity.polygon_ids:
            polygon = polygons[polygon_id]
            footprints.append(
                replace(
                    entity,
                    polygon_ids=(polygon_id,),
                    geometry={
                        **entity.geometry,
                        "outer_loop": polygon.exterior,
                        "hole_loops": polygon.holes,
                    },
                )
            )
    return tuple(footprints)


def entity_loop_bounds(entity: SemanticEntitySpec) -> Mapping[str, float]:
    """Compute XY bounds across an entity's outer and hole loops."""
    points = tuple(
        (
            point
            for loop in (
                entity.geometry["outer_loop"],
                *entity.geometry.get("hole_loops", ()),
            )
            for point in clean_loop(loop)
        )
    )
    return {
        "x_min_um": min((point[0] for point in points)),
        "y_min_um": min((point[1] for point in points)),
        "x_max_um": max((point[0] for point in points)),
        "y_max_um": max((point[1] for point in points)),
    }


def auto_vacuum_component_contacts_subtractor(
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
    subtractor_refs = geometry_refs_from_gdstk_region({}, subtractor_region)
    component_loops = (
        clean_loop(component_geometry_ref["outer_loop"]),
        *(clean_loop(loop) for loop in component_geometry_ref.get("hole_loops", ())),
    )
    return any(
        (
            loops_share_edge_overlap(component_loop, subtractor_loop)
            for component_loop in component_loops
            for subtractor_ref in subtractor_refs
            for subtractor_loop in (
                clean_loop(subtractor_ref["outer_loop"]),
                *(clean_loop(loop) for loop in subtractor_ref.get("hole_loops", ())),
            )
        )
    )


def entity_occupied_region(
    gdstk: Any,
    entity: SemanticEntitySpec,
    *,
    build_input: GeometryBuildInput | None = None,
) -> tuple[Any, ...]:
    """Use planned loops or the complete explicitly bound source polygon set."""
    if "outer_loop" not in entity.geometry:
        if build_input is None or not entity.polygon_ids:
            return ()
        return auto_vacuum_subtractor_xy_region(gdstk, build_input, entity)
    outer = gdstk.Polygon(clean_loop(entity.geometry["outer_loop"]))
    holes = tuple(
        (
            gdstk.Polygon(clean_loop(hole_loop))
            for hole_loop in entity.geometry.get("hole_loops", ())
        )
    )
    if not holes:
        return filter_gdstk_polygons((outer,))
    return boolean_polygon_region(gdstk, (outer,), holes, "not")


def domain_bounds_loop(bounds: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    """Read the ordered rectangular loop of declared XY domain bounds."""
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )


def solution_id_for_entity_coverage(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    z_um: float,
    mode: str,
    include_auto_vacuum: bool = True,
) -> str | None:
    """Resolve one exact solution cover for a structured conductor footprint.

    This is topology coverage, not a centroid or bounds proxy: every occupied
    polygon point must be inside one candidate solution region at the declared
    z relation. Disjoint covers therefore fail instead of assigning one part
    of a conductor to an arbitrary solution volume.
    """
    import gdstk  # noqa: PLC0415 -- defer the optional polygon engine until this operation reaches native geometry.

    occupied_region = entity_occupied_region(gdstk, entity, build_input=build_input)
    if not occupied_region:
        raise ValueError(f"{entity.semantic_id} has no occupied geometry region.")
    candidates: list[str] = []
    for solution in solution_region_entities(build_input):
        if not include_auto_vacuum and bool(
            solution.metadata.get("is_auto_vacuum_region")
        ):
            continue
        if mode == "bottom":
            matches = same_coordinate_z(float(solution.geometry["z_max_um"]), z_um)
        elif mode == "top":
            matches = same_coordinate_z(float(solution.geometry["z_min_um"]), z_um)
        elif mode == "containing":
            matches = (
                float(solution.geometry["z_min_um"])
                < z_um
                < float(solution.geometry["z_max_um"])
            )
        else:
            raise ValueError(f"unsupported solution adjacency mode {mode!r}")
        if not matches:
            continue
        uncovered_region = boolean_polygon_region(
            gdstk, occupied_region, solution_entity_xy_region(gdstk, solution), "not"
        )
        if not uncovered_region:
            candidates.append(solution.semantic_id)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"{entity.semantic_id} {mode} footprint has ambiguous exact solution coverage: {candidates!r}"
        )
    return candidates[0]


def solution_region_entities(
    build_input: GeometryBuildInput,
) -> tuple[SemanticEntitySpec, ...]:
    """Select the input entities that represent solution regions."""
    return tuple(
        (entity for entity in build_input.entities if is_solution_entity(entity))
    )


def solution_bounds(entity: SemanticEntitySpec) -> Mapping[str, float]:
    """Read a solution entity's declared domain bounds."""
    bounds = entity.geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise TypeError(f"{entity.semantic_id} requires domain_bounds_um")
    return bounds


def sidewall_on_solution_outer_boundary(
    geometry_ref: Mapping[str, Any], solution: SemanticEntitySpec
) -> bool:
    """Test whether a sidewall lies on the solution's outer XY boundary."""
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        return False
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    bounds = solution_bounds(solution)
    x_min = float(bounds["x_min_um"])
    x_max = float(bounds["x_max_um"])
    y_min = float(bounds["y_min_um"])
    y_max = float(bounds["y_max_um"])
    if same_coordinate_z(start[0], x_min) and same_coordinate_z(end[0], x_min):
        return range_within_bounds(start[1], end[1], y_min, y_max)
    if same_coordinate_z(start[0], x_max) and same_coordinate_z(end[0], x_max):
        return range_within_bounds(start[1], end[1], y_min, y_max)
    if same_coordinate_z(start[1], y_min) and same_coordinate_z(end[1], y_min):
        return range_within_bounds(start[0], end[0], x_min, x_max)
    if same_coordinate_z(start[1], y_max) and same_coordinate_z(end[1], y_max):
        return range_within_bounds(start[0], end[0], x_min, x_max)
    return False


def required_host_solution_id(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> str:
    """Return one authored conductor host, never an inferred global vacuum."""
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    host = entity_by_id(build_input, host_id)
    if not is_solution_entity(host):
        raise ValueError(
            f"{entity.semantic_id} host_void_semantic_id {host_id!r} is not a solution_region"
        )
    return host_id


def _face_metal_regions(
    gdstk: ModuleType,
    polygons: Mapping[str, LayoutPolygonSpec],
    face_entities: Sequence[SemanticEntitySpec],
    source_id: str,
) -> list[Polygon]:
    """Collect normalized face-metal regions while retaining source failures."""
    face_regions: list[Any] = []
    for entity in face_entities:
        for polygon_id in entity.polygon_ids:
            polygon = polygons.get(polygon_id)
            if polygon is None:
                raise ValueError(
                    f"{source_id} references missing normalized polygon {polygon_id!r}."
                )
            exterior = (gdstk.Polygon(clean_loop(polygon.exterior)),)
            holes = tuple((gdstk.Polygon(clean_loop(hole)) for hole in polygon.holes))
            face_regions.extend(boolean_polygon_region(gdstk, exterior, holes, "not"))
    return face_regions
