"""Where each source entity's planar footprint comes from (stage S2).

Owns resolving the footprint rings of an effective entity and putting them on the
model grid.  The rules mirror what the compiler's planning read:

* a **solution region** is its ``outer_loop`` / ``hole_loops`` when authored, else the
  rectangle of ``domain_bounds_um`` (``domain_planning.occupancy.solution_entity_xy_region``);
* the **automatic vacuum envelope** is the rectangle of the prepared enclosure
  (``RepresentationContext.enclosures``), which already holds the mapped effective
  bounds; padding is provenance and is never applied a second time
  (``domain_planning.vacuum``, the route-less branch of ``prepare_envelope``);
* a **conductor** is one footprint per owned source polygon -- the entity-owned
  logical polygons the curved path bound boundary records to
  (``curved_geometry.boundaries.source_rings``).  The entity's ``outer_loop`` is the grid
  union of those same polygons (``source.grid._project_entities``), so the covered
  region is identical; it is the fallback for an entity without polygons
  (``domain_planning.occupancy.route_conductor_contact_footprints``).

Rings are kept in source order here: boundary-curve bindings address them by index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.junctions import is_junction_end_piece
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.exact.grid import Grid
from scgsim.geometry.sgb.exact.plane import Point, Ring


@dataclass(frozen=True)
class SourceFootprint:
    """One connected source footprint of an entity: lattice rings in source order.

    ``authored`` is false for a junction end piece the partition derived: a boundary
    selector point binds authored footprints only, so one the end piece overlaps still names
    the terminal's own polygon.
    """

    entity_id: str
    polygon: int
    outer: Ring
    holes: tuple[Ring, ...] = ()
    authored: bool = True

    def rings(self) -> tuple[Ring, ...]:
        """The outer ring followed by the holes."""
        return (self.outer, *self.holes)


def is_auto_vacuum_region(entity: SemanticEntitySpec) -> bool:
    """Whether the entity is the automatic vacuum envelope."""
    return bool(entity.metadata.get("is_auto_vacuum_region"))


def is_solution_region(entity: SemanticEntitySpec) -> bool:
    """Whether the entity is a vacuum or dielectric solution region."""
    return entity.role == "solution_region" and entity.material_kind in {
        "vacuum",
        "dielectric",
    }


def source_footprints(
    entity: SemanticEntitySpec,
    build_input: GeometryBuildInput,
    grid: Grid,
    representation: RepresentationContext,
) -> list[SourceFootprint]:
    """The footprints of an entity on the model grid, one per connected source footprint."""
    authored: list[bool] = []
    if is_auto_vacuum_region(entity):
        loops = [(_envelope_rectangle(entity, representation), ())]
    elif is_solution_region(entity):
        loops = [_solution_loops(entity)]
    else:
        loops, authored = conductor_loops(entity, build_input)
    footprints = []
    for polygon, (outer, holes) in enumerate(loops):
        what = f"{entity.semantic_id} polygon {polygon}"
        footprints.append(
            SourceFootprint(
                entity.semantic_id,
                polygon,
                lattice_ring(outer, grid, f"{what} outer ring"),
                tuple(
                    lattice_ring(hole, grid, f"{what} hole {k}")
                    for k, hole in enumerate(holes)
                ),
                authored[polygon] if authored else True,
            )
        )
    return footprints


def _envelope_rectangle(
    entity: SemanticEntitySpec, representation: RepresentationContext
) -> Sequence[Sequence[float]]:
    enclosure = representation.enclosures.get(entity.semantic_id)
    if enclosure is not None:
        return _rectangle(enclosure.effective_bounds_um, entity.semantic_id)
    bounds = entity.geometry.get("domain_bounds_um")
    if isinstance(bounds, Mapping):
        return _rectangle(bounds, entity.semantic_id)
    raise InputError(
        f"{entity.semantic_id} is the automatic vacuum envelope but has neither a "
        "prepared enclosure nor domain_bounds_um"
    )


def _solution_loops(
    entity: SemanticEntitySpec,
) -> tuple[Sequence[Sequence[float]], Sequence[Sequence[Sequence[float]]]]:
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return geometry["outer_loop"], tuple(geometry.get("hole_loops", ()))
    bounds = geometry.get("domain_bounds_um", geometry.get("bounds_um"))
    if not isinstance(bounds, Mapping):
        raise InputError(
            f"{entity.semantic_id} has no source planar footprint: a solution region "
            "needs outer_loop or domain_bounds_um"
        )
    return _rectangle(bounds, entity.semantic_id), ()


def conductor_loops(
    entity: SemanticEntitySpec, build_input: GeometryBuildInput
) -> tuple[
    list[tuple[Sequence[Sequence[float]], Sequence[Sequence[Sequence[float]]]]],
    list[bool],
]:
    """The rings of each owned polygon in micrometres, and whether each is authored (not a junction end piece).

    The list is in footprint order: ``source_footprints`` numbers a conductor's footprints by
    it, and boundary bindings name a ring by that number.
    """
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    loops, authored = [], []
    for polygon_id in entity.polygon_ids:
        polygon = polygons.get(polygon_id)
        if polygon is None:
            raise InputError(
                f"{entity.semantic_id} references unknown polygon_id {polygon_id!r}"
            )
        loops.append((polygon.exterior, polygon.holes))
        authored.append(not is_junction_end_piece(polygon))
    if loops:
        return loops, authored
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return [(geometry["outer_loop"], tuple(geometry.get("hole_loops", ())))], [True]
    raise InputError(
        f"{entity.semantic_id} has no source planar footprint: it needs polygon_ids or outer_loop"
    )


def _rectangle(
    bounds: Mapping[str, Any], entity_id: str
) -> tuple[tuple[float, float], ...]:
    try:
        x0, x1 = float(bounds["x_min_um"]), float(bounds["x_max_um"])
        y0, y1 = float(bounds["y_min_um"]), float(bounds["y_max_um"])
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(
            f"{entity_id} bounds must define x_min_um, x_max_um, y_min_um and y_max_um"
        ) from error
    if not (x0 < x1 and y0 < y1):
        raise InputError(f"{entity_id} has a non-positive XY extent")
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def lattice_ring(ring_um: Sequence[Sequence[float]], grid: Grid, what: str) -> Ring:
    """Exact lattice coordinates of a ring given in micrometres (source order, nothing dropped).

    A coordinate that is not on the grid fails closed: the source stage projects every
    vertex onto the declared grid, so an off-lattice vertex means the input was not
    prepared with this grid.
    """
    points: list[Point] = []
    for x_um, y_um in ((p[0], p[1]) for p in ring_um):
        x, y = grid.lattice_units(x_um), grid.lattice_units(y_um)
        if x is None or y is None:
            raise InputError(
                f"{what}: vertex ({x_um}, {y_um}) um is off the {grid.size_um} um model "
                "grid; the source stage must project the geometry onto the declared grid"
            )
        points.append((x, y))
    return tuple(points)
