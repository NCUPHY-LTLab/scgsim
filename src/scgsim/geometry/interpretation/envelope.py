"""Resolve route-aware automatic vacuum envelopes and their authored padding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, cast

from scgsim.geometry.grid.planar_loops import clean_loop
from scgsim.geometry.interpretation.representation import (
    planned_conductor_representation,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.entities import entity_z_range_um, is_solution_entity
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.refusals import Declared, internal
from scgsim.geometry.models.representation import RepresentationContext


def _body(entity: object) -> Declared:
    """A body the vacuum envelope encloses, as its stack declares it."""
    return Declared(
        f"Entity {getattr(entity, 'semantic_id', None)!r}",
        "the stack: its PDK level, or the stack file's layers and solution_regions",
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


def is_auto_vacuum_subtractor(
    entity: SemanticEntitySpec,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    """Test whether the route treats an entity as positive vacuum occupancy."""
    if bool(entity.metadata.get("is_auto_vacuum_region")):
        return False
    if is_solution_entity(entity):
        return True
    representation = planned_conductor_representation(
        entity, route=route, context=representation_context
    )
    return representation in {"cutout_boundary_shell", "material_volume"}


def _auto_vacuum_surface_sheet_for_xy_envelope(
    entity: SemanticEntitySpec,
    route: RouteLiteral | None,
    *,
    representation_context: RepresentationContext | None = None,
) -> bool:
    return (
        route == "A"
        and entity.route_representations.get(route) == "surface_sheet"
        or (
            route is None
            and planned_conductor_representation(
                entity, route=route, context=representation_context
            )
            == "surface_sheet"
        )
    )


def _declared_domain_bounds(
    entity: SemanticEntitySpec, domain_bounds: Mapping[str, Any]
) -> dict[str, float]:
    """A solution region's declared XY bounds, which must be complete, finite and of positive extent."""
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    missing = [name for name in required if name not in domain_bounds]
    if missing:
        raise TypeError(
            _body(entity).expected(
                "domain_bounds_um",
                "x_min_um, x_max_um, y_min_um and y_max_um",
                sorted(domain_bounds),
            )
        )
    values = [cast(float, domain_bounds.get(name)) for name in required]
    if any((not isfinite(float(value)) for value in values)):
        raise ValueError(_body(entity).expected("domain_bounds_um", "finite", values))
    x_min_um, x_max_um, y_min_um, y_max_um = (float(value) for value in values)
    if not (x_min_um < x_max_um and y_min_um < y_max_um):
        raise ValueError(
            _body(entity).expected("domain_bounds_um", "a positive XY extent", values)
        )
    return {
        "x_min_um": x_min_um,
        "x_max_um": x_max_um,
        "y_min_um": y_min_um,
        "y_max_um": y_max_um,
    }


def _auto_vacuum_entity_xy_bounds(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> dict[str, float]:
    geometry = entity.geometry
    domain_bounds = geometry.get("domain_bounds_um")
    if isinstance(domain_bounds, Mapping):
        return _declared_domain_bounds(entity, domain_bounds)
    if "outer_loop" in geometry:
        outer_loop = clean_loop(geometry["outer_loop"])
        hole_loops = tuple(
            (clean_loop(loop) for loop in geometry.get("hole_loops", ()))
        )
        if not all(
            (
                isfinite(coordinate)
                for loop in (outer_loop, *hole_loops)
                for point in loop
                for coordinate in point
            )
        ):
            raise ValueError(
                internal(
                    f"Entity {entity.semantic_id!r} reached the vacuum envelope with a non-finite outline; the source stage refuses such a polygon where it is drawn"
                )
            )
        return loop_domain_bounds(outer_loop, hole_loops)
    polygon_ids = tuple(entity.polygon_ids)
    if not polygon_ids:
        raise ValueError(
            internal(
                f"Entity {entity.semantic_id!r} reached the vacuum envelope with no domain bounds, outline or polygons"
            )
        )
    return _polygon_entity_bounds(build_input, entity, polygon_ids)


def auto_vacuum_padding(
    auto_metadata: Mapping[str, Any], auto_region_id: str
) -> dict[str, float]:
    """Read the authored six-sided padding with its existing numeric contract."""
    raw = auto_metadata.get("vacuum_region_padding_um")
    if not isinstance(raw, Mapping):
        raise TypeError(
            internal(
                f"the automatic vacuum region {auto_region_id!r} carries no vacuum_region_padding_um"
            )
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
                internal(
                    f"the automatic vacuum region {auto_region_id!r} carries padding {key!r} = {value!r}, which set_vacuum_region refuses"
                )
            )
        value = float(value)
        if not isfinite(value) or value < 0.0:
            raise ValueError(
                internal(
                    f"the automatic vacuum region {auto_region_id!r} carries padding {key!r} = {value!r}, which set_vacuum_region refuses"
                )
            )
        values[key] = value
    return values


def auto_vacuum_envelope_bounds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    representation_context: RepresentationContext | None = None,
) -> dict[str, float]:
    """Collect the route's participating geometry into vacuum envelope bounds."""
    bounds: list[dict[str, float]] = []
    for entity in build_input.entities:
        is_subtractor = is_auto_vacuum_subtractor(
            entity, route=route, representation_context=representation_context
        )
        include_for_xy = _auto_vacuum_surface_sheet_for_xy_envelope(
            entity, route, representation_context=representation_context
        )
        if not is_subtractor and (not include_for_xy):
            continue
        if bool(entity.metadata.get("is_auto_vacuum_region")):
            continue
        xy_bounds = _auto_vacuum_entity_xy_bounds(build_input, entity)
        z_min_um, z_max_um = _envelope_z_interval(
            build_input, entity, is_subtractor, representation_context
        )
        bounds.append({**xy_bounds, "z_min_um": z_min_um, "z_max_um": z_max_um})
    if not bounds:
        raise ValueError(
            Declared(
                "the input",
                "the stack and the layout: give the vacuum region a body to enclose",
            ).refusal(
                "no body occupies a positive area on this route, so the automatic vacuum envelope has nothing to enclose"
            )
        )
    finite_z_bounds = tuple((bound for bound in bounds if isfinite(bound["z_min_um"])))
    if not finite_z_bounds:
        raise ValueError(
            Declared("the input", "the stack: give a body a finite Z interval").refusal(
                "no body has a finite Z interval on this route, so the automatic vacuum envelope has no height"
            )
        )
    return {
        "x_min_um": min((item["x_min_um"] for item in bounds)),
        "x_max_um": max((item["x_max_um"] for item in bounds)),
        "y_min_um": min((item["y_min_um"] for item in bounds)),
        "y_max_um": max((item["y_max_um"] for item in bounds)),
        "z_min_um": min((item["z_min_um"] for item in finite_z_bounds)),
        "z_max_um": max((item["z_max_um"] for item in finite_z_bounds)),
    }


def _envelope_z_interval(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    is_subtractor: bool,
    representation_context: RepresentationContext | None,
) -> tuple[float, float]:
    """The Z a body adds to the envelope: a solid's interval, or a sheet's plane as low and high."""
    if is_subtractor:
        z_min_um, z_max_um = entity_z_range_um(entity)
        if not (isfinite(z_min_um) and isfinite(z_max_um) and (z_max_um > z_min_um)):
            raise ValueError(
                _body(entity).expected(
                    "Z extent", "positive and finite", (z_min_um, z_max_um)
                )
            )
        return float(z_min_um), float(z_max_um)
    sheet_z_um = auto_vacuum_sheet_z(
        build_input, entity, representation_context=representation_context
    )
    return sheet_z_um, sheet_z_um


def auto_vacuum_sheet_z(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    representation_context: RepresentationContext | None = None,
) -> float:
    """Resolve the effective sheet height used by automatic envelope planning."""
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
            internal(
                "an automatic vacuum region on Route A met a sheet without its thin-film provenance"
            )
        )
    ranges = provenance.get("physical_face_metal_z_ranges_um")
    sheet_positions = provenance.get("effective_sheet_z_um")
    if not isinstance(ranges, Mapping) or not isinstance(sheet_positions, Mapping):
        raise ValueError(
            internal(
                "an automatic vacuum region on Route A met incomplete sheet provenance"
            )
        )
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
            internal(
                f"Entity {entity.semantic_id!r} has no unique effective Route-A sheet position"
            )
        )
    return float(sheet_positions[sides[0]])


def auto_vacuum_solution_region(
    build_input: GeometryBuildInput,
) -> SemanticEntitySpec | None:
    """Resolve the single authored automatic vacuum region, when present."""
    candidates = tuple(
        (
            entity
            for entity in build_input.entities
            if is_solution_entity(entity)
            and bool(entity.metadata.get("is_auto_vacuum_region"))
        )
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            internal(
                f"the input holds {len(candidates)} automatic vacuum regions; set_vacuum_region makes one"
            )
        )
    return cast(Sequence[SemanticEntitySpec], candidates)[0]


def _polygon_entity_bounds(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    polygon_ids: tuple[str, ...],
) -> dict[str, float]:
    """Read complete source polygon bounds with the existing geometry checks."""
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    points: list[tuple[float, float]] = []
    for polygon_id in polygon_ids:
        try:
            polygon = polygons[polygon_id]
        except KeyError as exc:
            raise ValueError(
                internal(
                    f"Entity {entity.semantic_id!r} names a layout polygon the input does not hold"
                )
            ) from exc
        loop = clean_loop(polygon.exterior)
        if len(loop) < 3:
            raise ValueError(
                internal(
                    f"Entity {entity.semantic_id!r} polygon {polygon_id!r} is degenerate; the source stage refuses such a polygon where it is drawn"
                )
            )
        points.extend(loop)
        for hole in polygon.holes:
            hole_loop = clean_loop(hole)
            if not all(
                (isfinite(coordinate) for point in hole_loop for coordinate in point)
            ):
                raise ValueError(
                    internal(
                        f"Entity {entity.semantic_id!r} polygon {polygon_id!r} is not finite; the source stage refuses such a polygon where it is drawn"
                    )
                )
            points.extend(hole_loop)
    if not points:
        raise ValueError(
            internal(
                f"Entity {entity.semantic_id!r} reached the vacuum envelope with no geometry points"
            )
        )
    if not all((isfinite(coordinate) for point in points for coordinate in point)):
        raise ValueError(
            internal(
                f"Entity {entity.semantic_id!r} reached the vacuum envelope with non-finite polygons; the source stage refuses such a polygon where it is drawn"
            )
        )
    return {
        "x_min_um": min((point[0] for point in points)),
        "x_max_um": max((point[0] for point in points)),
        "y_min_um": min((point[1] for point in points)),
        "y_max_um": max((point[1] for point in points)),
    }
