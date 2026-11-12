"""Bind authored boundaries and explicit reconstruction to original source rings.

The compiler's construction path binds records here; the SGB route, the junction partition
included, binds them with ``sgb.prisms.curves``.  A refusal names the record, its Entity and
the ring's layer, says where in micrometres the ring, the indices or the arc points at fault
lie, and who must change them: the declaration, or the layout for an authored arc.  Where SGB
refuses the same fault, the refusal is SGB's, word for word.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from scgsim.geometry.models.common import Coordinate, PolygonRing
from scgsim.geometry.models.grid import (
    format_extent,
    format_point,
    geometry_grid_context,
)
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
    arc_without_circle,
    collinear,
    drawn_chain_on_ring,
    drawn_ring_of,
    drawn_vertices_off_ring,
    exact_point_um,
    reconstruction_anchors,
    reconstruction_chain_refusal,
    reconstruction_without_circle,
    ring_lost_drawn_order,
    ring_runs,
    ring_without_drawn_order,
    twice_area,
)
from scgsim.geometry.models.junctions import is_junction_end_piece
from scgsim.geometry.primitives.grid_regions import region_key
from scgsim.geometry.source.curves import (
    BoundaryKey,
    SourceRings,
    boundary_binding,
    select_boundary,
)

BoundaryChains = dict[BoundaryKey, tuple[SourceCurveSpec, ...]]
ReconstructionSelection = tuple[tuple[int, ...], tuple[SourceCurveSpec, ...], str]


def source_z_range(entity: SemanticEntitySpec) -> tuple[float, float]:
    """Read the source-authored lower and upper Z coordinates."""
    geometry = entity.geometry
    lower = float(geometry.get("z_min_um", geometry.get("z_um", 0.0)))
    return lower, float(
        geometry.get("z_max_um", lower + float(geometry.get("thickness_um", 0.0)))
    )


def source_rings(build_input: GeometryBuildInput) -> SourceRings:
    """Use adapter-normalized Entity-owned logical polygons, including holes."""
    polygons = {p.polygon_id: p for p in build_input.polygons}
    result: SourceRings = {}
    for entity in build_input.entities:
        if entity.polygon_ids:
            result[entity.semantic_id] = [
                (pid, polygons[pid].exterior, polygons[pid].holes)
                for pid in entity.polygon_ids
            ]
        else:
            outer = entity.geometry.get("outer_loop")
            if entity.metadata.get("is_auto_vacuum_region"):
                outer = entity.metadata.get("auto_vacuum_envelope_outer_loop", outer)
            if outer is None:
                bounds = entity.geometry.get(
                    "domain_bounds_um", entity.geometry.get("bounds_um")
                )
                if bounds is None:
                    raise ValueError(
                        f"{entity.semantic_id} has no source planar footprint"
                    )
                outer = (
                    (bounds["x_min_um"], bounds["y_min_um"]),
                    (bounds["x_max_um"], bounds["y_min_um"]),
                    (bounds["x_max_um"], bounds["y_max_um"]),
                    (bounds["x_min_um"], bounds["y_max_um"]),
                )
            result[entity.semantic_id] = [
                (
                    entity.semantic_id,
                    outer,
                    ()
                    if entity.metadata.get("is_auto_vacuum_region")
                    else entity.geometry.get("hole_loops", ()),
                )
            ]
    return result


def circle_through_points(points: Sequence[Coordinate]) -> tuple[Coordinate, float]:
    """Solve the circle through three supplied arc points, exactly on their decimal values.

    The binding (:func:`source_boundaries`) refuses an arc whose three points lie on one line,
    read the same way, so the determinant is never zero here for an arc that passed it,
    however the floats round.
    """
    a, b, c = (exact_point_um(point) for point in points)
    determinant = 2 * twice_area(a, b, c)
    a_sq, b_sq, c_sq = (p[0] ** 2 + p[1] ** 2 for p in (a, b, c))
    ux = (
        a_sq * (b[1] - c[1]) + b_sq * (c[1] - a[1]) + c_sq * (a[1] - b[1])
    ) / determinant
    uy = (
        a_sq * (c[0] - b[0]) + b_sq * (a[0] - c[0]) + c_sq * (b[0] - a[0])
    ) / determinant
    return (float(ux), float(uy)), math.hypot(float(a[0] - ux), float(a[1] - uy))


def _on_one_line(points: Sequence[Coordinate]) -> bool:
    # Whether an arc's three points, in micrometres, define no circle, by the exact rule
    # collinear states, on the values circle_through_points solves on.
    a, b, c = (exact_point_um(point) for point in points)
    return collinear(a, b, c)


@dataclass(frozen=True)
class _BindingPlaces:
    """What the binding's refusals call a ring, and the grid they place it on."""

    grid_size_um: float | None
    # A ring's polygon is one its Entity owns; an outline ring, keyed by the Entity's own
    # semantic ID, has none, whatever polygon happens to carry that ID.
    owned: Mapping[tuple[str, str], LayoutPolygonSpec]

    @classmethod
    def read(cls, build_input: GeometryBuildInput) -> _BindingPlaces:
        """The grid and the owned polygons of an input."""
        by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
        context = geometry_grid_context(build_input)
        return cls(
            None if context is None else context.grid_size_um,
            {
                (entity.semantic_id, polygon_id): by_id[polygon_id]
                for entity in build_input.entities
                for polygon_id in entity.polygon_ids
            },
        )

    def layers(self) -> dict[tuple[str, str], str]:
        """The layer of each owned polygon, by (Entity, polygon ID)."""
        return {key: polygon.layer for key, polygon in self.owned.items()}

    def polygon(self, key: BoundaryKey) -> LayoutPolygonSpec | None:
        """The polygon whose ring a key names, or None for an Entity's outline ring."""
        return self.owned.get((key[0], key[1]))

    def claimants(self, key: BoundaryKey, names: Sequence[str]) -> str:
        """The records that claim one ring, with its Entity and layer."""
        polygon = self.polygon(key)
        on = "" if polygon is None else f" on layer {polygon.layer}"
        return f"{' and '.join(names)} (Entity {key[0]!r}{on})"

    def claim(self, key: BoundaryKey, ring: PolygonRing, names: Sequence[str]) -> str:
        """The records that claim one ring, and where the ring lies."""
        return (
            f"{self.claimants(key, names)} select the ring at "
            f"{format_extent(ring, self.grid_size_um)}"
        )


def _named(
    record: BoundaryCurveChainSpec | GdsBoundaryReconstructionSpec,
    polygon: LayoutPolygonSpec | None,
) -> str:
    """The record, its Entity and the selected polygon's layer, as refusals name them."""
    on = "" if polygon is None else f", layer {polygon.layer}"
    return f"{record.boundary_id} (Entity {record.entity_id!r}{on})"


def _drawn_ring(
    record: GdsBoundaryReconstructionSpec,
    key: BoundaryKey,
    ring: PolygonRing,
    polygon: LayoutPolygonSpec | None,
    grid_size_um: float | None,
) -> PolygonRing:
    # The ring the record's indices name, by the one rule drawn_ring_of states; a ring no
    # polygon holds (an Entity's outline loop) is given as drawn.
    if polygon is None:
        return ring
    _, _, role, hole_index = key
    drawn = drawn_ring_of(polygon, 0 if role == "outer" else 1 + cast(int, hole_index))
    if drawn is None:
        raise ValueError(
            ring_without_drawn_order(
                _named(record, polygon), format_extent(ring, grid_size_um)
            )
        )
    return drawn


def _reconstruct(  # noqa: PLR0913 -- the record, its two rings, how refusals name it, and the grid
    record: GdsBoundaryReconstructionSpec,
    ring: PolygonRing,
    drawn: PolygonRing,
    what: str,
    grid_size_um: float | None,
):
    # The record's indices name the drawn ring; they are checked there, then carried onto
    # the ring this stage holds, which may start elsewhere, run the other way and hold
    # crossings the grid noded in.
    def drawn_place(i: int) -> str:
        return format_point(*drawn[i], grid_size_um)

    refusal = reconstruction_chain_refusal(
        record, what, len(drawn), drawn_place, format_extent(drawn, grid_size_um)
    )
    if refusal is not None:
        raise ValueError(refusal)
    segments = tuple(record.segment_indices)
    drawn_anchors = reconstruction_anchors(record, len(drawn))
    drawn_keys = [region_key(point, grid_size_um) for point in drawn]
    ring_keys = [region_key(point, grid_size_um) for point in ring]
    chain = drawn_chain_on_ring(
        segments, drawn_anchors, record.closed, drawn_keys, ring_keys
    )
    if chain is None:
        raise ValueError(
            ring_lost_drawn_order(
                what,
                [
                    f"drawn vertex {i} at {drawn_place(i)} {fault}"
                    for i, fault in drawn_vertices_off_ring(drawn_keys, ring_keys)
                ],
            )
        )
    segments, anchors = chain.segments, chain.anchors
    count = len(ring)
    points = tuple(ring[i] for i in anchors)
    if record.kind == "circular_arc" and _on_one_line(points):
        raise ValueError(
            reconstruction_without_circle(what, drawn_anchors, drawn_place)
        )
    if record.kind == "circular_arc" and record.closed:
        center, radius = circle_through_points(points)
        start = points[0]
        angle = math.atan2(start[1] - center[1], start[0] - center[0])
        area = sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
        )
        direction = 1 if area > 0 else -1

        def position(t):
            return (
                center[0] + radius * math.cos(t),
                center[1] + radius * math.sin(t),
            )

        opposite = position(angle + direction * math.pi)
        curves = (
            SourceCurveSpec(
                "circular_arc",
                (start, position(angle + direction * math.pi / 2), opposite),
            ),
            SourceCurveSpec(
                "circular_arc",
                (opposite, position(angle + direction * 3 * math.pi / 2), start),
            ),
        )
    else:
        curves = (
            SourceCurveSpec(
                record.kind,
                points,
                degree=record.degree,
                parameter_interval=record.parameter_interval,
            ),
        )
    return (
        segments,
        curves,
        {
            "requested": asdict(record),
            "drawn_ring": drawn,
            "original_ring_um": ring,
            "selected_segments": [
                {"index": i, "start_um": ring[i], "end_um": ring[(i + 1) % count]}
                for i in segments
            ],
            "selected_anchor_indices": anchors,
            "selected_anchor_points_um": points,
        },
    )


def _refuse_arcs_without_circle(
    record: BoundaryCurveChainSpec, what: str, grid_size_um: float | None
) -> None:
    # SGB refuses an authored arc whose three points define no circle, but the compiler's
    # curve kernel builds arcs without SGB and solves the circle with circle_through_points.
    # The arc is refused here, in SGB's words, by the exact rule that circle solves on.  A
    # component's arc was decided in its own frame by the source stage; this decides an arc
    # declared in assembly coordinates, and guards the circle the kernel solves.
    for index, curve in enumerate(record.curves):
        if curve.kind == "circular_arc" and _on_one_line(curve.points_um):
            raise ValueError(
                arc_without_circle(
                    f"{what} curve {index}",
                    [format_point(x, y, grid_size_um) for x, y in curve.points_um],
                )
            )


def source_boundaries(
    build_input: GeometryBuildInput, rings: SourceRings
) -> tuple[BoundaryChains, list[dict[str, object]]]:
    """Bind authored and reconstructed curves in original source-ring order."""
    places = _BindingPlaces.read(build_input)
    end_pieces = frozenset(
        polygon.polygon_id
        for polygon in build_input.polygons
        if is_junction_end_piece(polygon)
    )
    layers = places.layers()

    def select(
        record: BoundaryCurveChainSpec | GdsBoundaryReconstructionSpec,
    ) -> tuple[BoundaryKey, PolygonRing]:
        return select_boundary(
            record,
            rings,
            end_pieces,
            layers=layers,
            grid_size_um=places.grid_size_um,
        )

    authored: dict[BoundaryKey, BoundaryCurveChainSpec] = {}
    provenance: list[dict[str, object]] = []
    for raw in build_input.boundary_curves:
        record = boundary_binding(raw)
        key, ring = select(record)
        if key in authored:
            names = (authored[key].boundary_id, record.boundary_id)
            raise ValueError(
                f"multiple complete authored boundaries claim {key}: "
                f"{places.claim(key, ring, names)}. Fix the declaration: give each ring "
                "one complete boundary record."
            )
        _refuse_arcs_without_circle(
            record, _named(record, places.polygon(key)), places.grid_size_um
        )
        authored[key] = record
        provenance.append(
            {
                "source": asdict(record),
                "original_ring_um": ring,
                "operation": "authored",
            }
        )
    reconstructed: defaultdict[BoundaryKey, list[ReconstructionSelection]] = (
        defaultdict(list)
    )
    for raw in build_input.boundary_reconstruction:
        record = boundary_binding(raw, reconstruction=True)
        key, ring = select(record)
        polygon = places.polygon(key)
        drawn = _drawn_ring(record, key, ring, polygon, places.grid_size_um)
        segments, curves, operation = _reconstruct(
            record, ring, drawn, _named(record, polygon), places.grid_size_um
        )
        reconstructed[key].append((segments, curves, record.boundary_id))
        provenance.append({"operation": "explicit_gds_reconstruction", **operation})
    return _ring_chains(rings, authored, reconstructed, places), provenance


def _ring_chains(
    rings: SourceRings,
    authored: Mapping[BoundaryKey, BoundaryCurveChainSpec],
    reconstructed: Mapping[BoundaryKey, Sequence[ReconstructionSelection]],
    places: _BindingPlaces,
) -> BoundaryChains:
    """Each source ring as its authored curves, or its segments with reconstructions put in."""
    result: BoundaryChains = {}
    for entity_id, records in rings.items():
        for pid, outer, holes in records:
            for role, index, ring in [
                ("outer", None, outer),
                *[("hole", i, h) for i, h in enumerate(holes)],
            ]:
                key = (entity_id, pid, role, index)
                if key in authored:
                    if key in reconstructed:
                        names = (
                            authored[key].boundary_id,
                            *(bid for _, _, bid in reconstructed[key]),
                        )
                        raise ValueError(
                            f"authored and reconstructed boundaries both claim {key}: "
                            f"{places.claim(key, ring, names)}. Fix the declaration: "
                            "describe the ring by one authored boundary or by "
                            "reconstruction records, not both."
                        )
                    result[key] = authored[key].curves
                    continue
                result[key] = _replace_segments(
                    key, ring, reconstructed.get(key, ()), places
                )
    return result


def _replace_segments(
    key: BoundaryKey,
    ring: PolygonRing,
    operations: Sequence[ReconstructionSelection],
    places: _BindingPlaces,
) -> tuple[SourceCurveSpec, ...]:
    """Replace selected source chains while retaining unselected straight segments."""
    claimed = {}
    starts = {}
    for segments, curves, bid in operations:
        shared = [i for i in segments if i in claimed]
        if shared:
            names = tuple(dict.fromkeys((*(claimed[i] for i in shared), bid)))
            runs = ring_runs(
                shared,
                len(ring),
                lambda i: format_point(*ring[i], places.grid_size_um),
            )
            raise ValueError(
                f"overlapping reconstruction segment claims on {key}: "
                f"{places.claimants(key, names)} both replace the ring {runs}. Fix the "
                "declaration: give each drawn segment of the ring to one reconstruction "
                "record."
            )
        for i in segments:
            claimed[i] = bid
        starts[segments[0]] = curves
    # Traverse from an operation start when it crosses original ring index zero.
    first = next((segments[0] for segments, _, _ in operations if 0 in segments), 0)
    indices = [(first + i) % len(ring) for i in range(len(ring))]
    curves = []
    for i in indices:
        if i in starts:
            curves.extend(starts[i])
        elif i not in claimed:
            curves.append(
                SourceCurveSpec(
                    "line_segment",
                    (
                        cast(Coordinate, tuple(ring[i])),
                        cast(Coordinate, tuple(ring[(i + 1) % len(ring)])),
                    ),
                )
            )
    return tuple(curves)
