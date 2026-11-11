"""Authored boundary curves and explicit GDS reconstruction (stage S2).

Owns binding ``boundary_curves`` and ``boundary_reconstruction`` records to the
source rings of an entity and turning them into ring primitives (straight edges
and circular arcs), as the compiler's ``construction.curved_geometry.boundaries``
did:

* an **authored** record replaces one complete outer or hole ring with its chain of
  line and circular-arc curves;
* a **reconstruction** record replaces selected ordered segments of one ring with a
  circular arc (open chain: three anchors; closed chain: a full circle through three
  anchors), keeping the unselected straight segments.

Lines and circular arcs are the curve classes of this version.  Interpolation
splines and B-splines fail closed, naming the boundary and the entity: B-spline
carriers are a later stage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    SourceCurveSpec,
)
from scgsim.geometry.sgb.errors import InputError, UnsupportedCurveError
from scgsim.geometry.sgb.plane import (
    Point,
    Ring,
    fraction_to_rep,
    point_in_ring,
    ring_twice_area,
)
from scgsim.geometry.sgb.prisms.arcs import arc_through_points, full_circle
from scgsim.geometry.sgb.prisms.footprints import SourceFootprint
from scgsim.geometry.sgb.records.carriers import Lineage
from scgsim.geometry.sgb.records.grid import Grid
from scgsim.geometry.sgb.records.prisms import LineEdge, RingPrimitive

RingKey = tuple[
    str, int, int
]  # (entity id, polygon index, ring index: 0 outer, 1 + k for hole k)
BoundaryRecord = BoundaryCurveChainSpec | GdsBoundaryReconstructionSpec
SUPPORTED_CURVES = "lines and circular arcs"


def bind_boundaries(
    build_input: GeometryBuildInput,
    grid: Grid,
    footprints: Mapping[str, Sequence[SourceFootprint]],
) -> dict[RingKey, tuple[RingPrimitive, ...]]:
    """Ring primitives for every ring an authored or reconstruction record binds."""
    binder = _BoundaryBinder(grid, footprints, _entity_layers(build_input))
    bound: dict[RingKey, tuple[RingPrimitive, ...]] = {}
    for record in build_input.boundary_curves:
        key, _ = binder.select_ring(record)
        if key in bound:
            raise InputError(f"multiple complete authored boundaries claim ring {key}")
        bound[key] = binder.authored_primitives(record, key)
    selections: dict[RingKey, list[_Selection]] = defaultdict(list)
    for reconstruction in build_input.boundary_reconstruction:
        key, ring = binder.select_ring(reconstruction)
        selections[key].append(binder.reconstruction(reconstruction, key, ring))
    for key, chosen in selections.items():
        if key in bound:
            raise InputError(
                f"authored and reconstructed boundaries both claim ring {key}"
            )
        bound[key] = _replace_segments(key, binder.ring_of(key), chosen)
    return bound


@dataclass(frozen=True)
class _Selection:
    """The source segments one reconstruction record replaces, and what replaces them."""

    segments: tuple[int, ...]
    primitives: tuple[RingPrimitive, ...]
    boundary_id: str


class _BoundaryBinder:
    """Selects the ring a record addresses and builds its primitives on the grid."""

    def __init__(
        self,
        grid: Grid,
        footprints: Mapping[str, Sequence[SourceFootprint]],
        layers: Mapping[str, str],
    ) -> None:
        self.grid = grid
        self.footprints = footprints
        self.layers = layers

    def _named(self, record: BoundaryRecord) -> str:
        """The record, its Entity and the Entity's layer, as refusals name them."""
        layer = self.layers.get(record.entity_id)
        on = f", layer {layer}" if layer else ""
        return f"{record.boundary_id} (Entity {record.entity_id!r}{on})"

    # -- selecting the ring ------------------------------------------------------------

    def select_ring(self, record: BoundaryRecord) -> tuple[RingKey, Ring]:
        """The one authored entity-owned footprint the selector point lies in, and the ring the role names.

        A junction end piece is never the selector's polygon: it can overlap the terminal's
        authored polygon, which the selector names as it did before the partition.
        """
        selector = fraction_to_rep(
            self.grid.exact_units(record.selector_point_um[0]),
            self.grid.exact_units(record.selector_point_um[1]),
        )
        matches = [
            footprint
            for footprint in self.footprints.get(record.entity_id, ())
            if footprint.authored
            and sum(point_in_ring(ring, selector) for ring in footprint.rings()) % 2
            == 1
        ]
        if len(matches) != 1:
            raise InputError(
                f"{record.boundary_id} selector must bind one Entity-owned logical polygon "
                f"of {record.entity_id!r}; it binds {len(matches)}"
            )
        footprint = matches[0]
        if record.role == "outer":
            ring_index = 0
        else:
            ring_index = 1 + (record.hole_index or 0)
            if ring_index > len(footprint.holes):
                raise InputError(
                    f"{record.boundary_id}: {record.entity_id!r} polygon {footprint.polygon} "
                    f"has no hole {record.hole_index}"
                )
        key = (record.entity_id, footprint.polygon, ring_index)
        return key, footprint.rings()[ring_index]

    def ring_of(self, key: RingKey) -> Ring:
        """The source ring of a key."""
        entity_id, polygon, ring_index = key
        footprint = next(f for f in self.footprints[entity_id] if f.polygon == polygon)
        return footprint.rings()[ring_index]

    # -- authored chains ---------------------------------------------------------------

    def authored_primitives(
        self, record: BoundaryCurveChainSpec, key: RingKey
    ) -> tuple[RingPrimitive, ...]:
        """The primitives of an authored chain, checked to be one closed chain."""
        primitives: list[RingPrimitive] = []
        first_start: Point | None = None
        previous_end: Point | None = None
        for index, curve in enumerate(record.curves):
            what = f"{self._named(record)} curve {index}"
            start, end, new = self._curve_primitives(curve, key, index, what)
            if previous_end is not None and start != previous_end:
                raise InputError(f"{what} does not start where the previous curve ends")
            first_start = start if first_start is None else first_start
            previous_end = end
            primitives.extend(new)
        if first_start is None or previous_end != first_start:
            raise InputError(
                f"{record.boundary_id} (Entity {record.entity_id!r}) curves do not form one closed chain"
            )
        return tuple(primitives)

    def _curve_primitives(
        self, curve: SourceCurveSpec, key: RingKey, index: int, what: str
    ) -> tuple[Point, Point, list[RingPrimitive]]:
        if curve.kind == "line_segment":
            start, end = (self._lattice_point(p, what) for p in curve.points_um)
            return start, end, [LineEdge(start, end, index)]
        if curve.kind == "circular_arc":
            start = self._lattice_point(curve.points_um[0], what)
            end = self._lattice_point(curve.points_um[2], what)
            through = self._exact_point(curve.points_um[1])
            arc_id = _arc_id(key, index)
            lineage = Lineage("polygon_edge", key[0], key[1], key[2], index)
            return (
                start,
                end,
                list(
                    arc_through_points(
                        arc_id, start, through, end, lineage, what, grid=self.grid
                    )
                ),
            )
        raise UnsupportedCurveError(
            f"{what}: {curve.kind!r} curves are not supported yet "
            f"(this version carries {SUPPORTED_CURVES})"
        )

    # -- reconstruction ----------------------------------------------------------------

    def reconstruction(
        self, record: GdsBoundaryReconstructionSpec, key: RingKey, ring: Ring
    ) -> _Selection:
        """The segments a reconstruction record replaces and the circular arc that replaces them."""
        what = self._named(record)
        if record.kind != "circular_arc":
            raise UnsupportedCurveError(
                f"{what}: reconstruction kind {record.kind!r} is not supported yet "
                f"(this version carries {SUPPORTED_CURVES}; use 'circular_arc')"
            )
        segments = _selected_segments(record, ring, what)
        vertices = (*segments, (segments[-1] + 1) % len(ring))
        anchors = tuple(record.anchor_indices) or vertices
        if any(i not in vertices for i in anchors):
            raise InputError(f"{what} anchors must belong to the selected source chain")
        if len(anchors) != 3:
            raise InputError(
                f"{what} circular reconstruction requires exactly three anchors"
            )
        if not record.closed and (
            anchors[0] != vertices[0] or anchors[-1] != vertices[-1]
        ):
            raise InputError(
                f"{what} reconstruction must retain the original chain endpoints"
            )
        points = tuple(ring[i] for i in anchors)
        arc_id = _arc_id(key, segments[0])
        lineage = Lineage("polygon_edge", key[0], key[1], key[2], segments[0])
        if record.closed:
            ccw = ring_twice_area(ring) > 0
            arcs = full_circle(
                arc_id,
                (points[0], points[1], points[2]),
                ccw,
                lineage,
                what,
                grid=self.grid,
            )
        else:
            through = (Fraction(points[1][0]), Fraction(points[1][1]))
            arcs = arc_through_points(
                arc_id, points[0], through, points[2], lineage, what, grid=self.grid
            )
        return _Selection(segments, tuple(arcs), record.boundary_id)

    # -- coordinates -------------------------------------------------------------------

    def _lattice_point(self, point_um: Sequence[float], what: str) -> Point:
        x, y = (
            self.grid.lattice_units(point_um[0]),
            self.grid.lattice_units(point_um[1]),
        )
        if x is None or y is None:
            raise InputError(
                f"{what}: end point ({point_um[0]}, {point_um[1]}) um is off the "
                f"{self.grid.size_um} um model grid"
            )
        return (x, y)

    def _exact_point(self, point_um: Sequence[float]) -> tuple[Fraction, Fraction]:
        return (self.grid.exact_units(point_um[0]), self.grid.exact_units(point_um[1]))


def _entity_layers(build_input: GeometryBuildInput) -> dict[str, str]:
    """The source layer(s) of each Entity's polygons, joined for a refusal."""
    layer_of = {p.polygon_id: p.layer for p in build_input.polygons}
    return {
        e.semantic_id: ", ".join(
            sorted({layer_of[i] for i in e.polygon_ids if i in layer_of})
        )
        for e in build_input.entities
    }


def _arc_id(key: RingKey, edge: int) -> str:
    return f"{key[0]}/polygon{key[1]}/ring{key[2]}/arc{edge}"


def _selected_segments(
    record: GdsBoundaryReconstructionSpec, ring: Ring, what: str
) -> tuple[int, ...]:
    count = len(ring)
    segments = tuple(record.segment_indices)
    if not segments or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count
        for i in segments
    ):
        raise InputError(f"{what} requires explicit original segment indices")
    if len(set(segments)) != len(segments) or any(
        b != (a + 1) % count for a, b in zip(segments, segments[1:], strict=False)
    ):
        raise InputError(f"{what} segments must form one ordered source chain")
    closes = len(segments) == count and (segments[-1] + 1) % count == segments[0]
    if record.closed != closes:
        raise InputError(f"{what} closure differs from the selected source chain")
    return segments


def _replace_segments(
    key: RingKey, ring: Ring, selections: Sequence[_Selection]
) -> tuple[RingPrimitive, ...]:
    """Replace the selected source chains by their arcs and keep the unselected straight segments."""
    claimed: set[int] = set()
    starts: dict[int, tuple[RingPrimitive, ...]] = {}
    for selection in selections:
        if claimed & set(selection.segments):
            raise InputError(f"overlapping reconstruction segment claims on ring {key}")
        claimed.update(selection.segments)
        starts[selection.segments[0]] = selection.primitives
    count = len(ring)
    # Walk from a selection start that crosses original ring index zero, so the chain stays contiguous.
    first = next((s.segments[0] for s in selections if 0 in s.segments), 0)
    primitives: list[RingPrimitive] = []
    for step in range(count):
        i = (first + step) % count
        if i in starts:
            primitives.extend(starts[i])
        elif i not in claimed and ring[i] != ring[(i + 1) % count]:
            primitives.append(LineEdge(ring[i], ring[(i + 1) % count], i))
    return tuple(primitives)
