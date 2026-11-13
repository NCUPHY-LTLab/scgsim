"""Authored boundary curves and explicit GDS reconstruction (stage S2).

Owns binding ``boundary_curves`` and ``boundary_reconstruction`` records to the
source rings of an entity and turning them into ring primitives (straight edges
and circular arcs), as the compiler's ``construction.curved_geometry.boundaries``
did:

* an **authored** record replaces one complete outer or hole ring with its chain of
  line and circular-arc curves;
* a **reconstruction** record replaces selected ordered segments of one ring with a
  circular arc (open chain: three anchors; closed chain: a full circle through three
  anchors), keeping the unselected straight segments.  Its indices name the ring as
  drawn in the GDS, which the source stage records in the polygon's ``drawn_rings``;
  they are carried onto the footprint's ring by ``drawn_chain_on_ring``.

Lines and circular arcs are the curve classes of this version.  Interpolation
splines and B-splines fail closed, naming the boundary and the entity: B-spline
carriers are a later stage.

A refusal names the record, its Entity and layer, where the fault lies in micrometres and
who must act.  The wording of the index and arc refusals lives in ``models.input``, shared
with the compiler's construction binder.  The junction partition reads a lumped port's
terminals through this binding (``bound_rings``), so a terminal's record is refused here too.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.models.grid import format_box, format_point
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SourceCurveSpec,
    arc_without_circle,
    collinear,
    drawn_chain_on_ring,
    drawn_ring_of,
    drawn_vertices_off_ring,
    reconstruction_anchors,
    reconstruction_chain_refusal,
    reconstruction_without_circle,
    ring_lost_drawn_order,
    ring_runs,
    ring_without_drawn_order,
)
from scgsim.geometry.sgb.errors import InputError, UnsupportedCurveError
from scgsim.geometry.sgb.exact.plane import (
    Point,
    Ring,
    fraction_to_rep,
    point_in_ring,
    ring_twice_area,
)
from scgsim.geometry.sgb.prisms.arcs import arc_through_points, full_circle
from scgsim.geometry.sgb.prisms.footprints import SourceFootprint
from scgsim.geometry.sgb.exact.carriers import Lineage
from scgsim.geometry.sgb.exact.grid import Grid
from scgsim.geometry.sgb.exact.prisms import LineEdge, RingPrimitive

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
    binder = _BoundaryBinder(
        grid, footprints, _entity_layers(build_input), _footprint_polygons(build_input)
    )
    bound: dict[RingKey, tuple[RingPrimitive, ...]] = {}
    authored: dict[RingKey, BoundaryRecord] = {}
    for record in build_input.boundary_curves:
        key, _ = binder.select_ring(record)
        if key in bound:
            raise InputError(
                f"multiple complete authored boundaries claim ring {key}: "
                f"{binder.claim(key, (authored[key], record))}. Fix the declaration: give "
                "each ring one complete boundary record."
            )
        authored[key] = record
        bound[key] = binder.authored_primitives(record, key)
    selections: dict[RingKey, list[_Selection]] = defaultdict(list)
    reconstructions: dict[RingKey, list[BoundaryRecord]] = defaultdict(list)
    for reconstruction in build_input.boundary_reconstruction:
        key, ring = binder.select_ring(reconstruction)
        selections[key].append(binder.reconstruction(reconstruction, key, ring))
        reconstructions[key].append(reconstruction)
    for key, chosen in selections.items():
        if key in bound:
            raise InputError(
                f"authored and reconstructed boundaries both claim ring {key}: "
                f"{binder.claim(key, (authored[key], *reconstructions[key]))}. Fix the "
                "declaration: describe the ring by one authored boundary or by "
                "reconstruction records, not both."
            )
        bound[key] = binder.replace_segments(key, chosen)
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
        polygons: Mapping[tuple[str, int], LayoutPolygonSpec],
    ) -> None:
        self.grid = grid
        self.footprints = footprints
        self.layers = layers
        self.polygons = polygons

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
                f"of {record.entity_id!r}; it binds {len(matches)}: "
                f"{self._selector_finding(record, matches)}. Fix the declaration: put the "
                "boundary's selector point inside the one polygon of the Entity whose ring "
                "it describes, and name the Entity that owns that polygon."
            )
        footprint = matches[0]
        if record.role == "outer":
            ring_index = 0
        else:
            ring_index = 1 + (record.hole_index or 0)
            if ring_index > len(footprint.holes):
                raise InputError(
                    f"{record.boundary_id}: {record.entity_id!r} polygon {footprint.polygon} "
                    f"has no hole {record.hole_index}: the polygon{self._on(record)} at "
                    f"{self._place(footprint.outer)} has {len(footprint.holes)}. Fix the "
                    "declaration: give the hole_index of one of its holes, counted from 0."
                )
        key = (record.entity_id, footprint.polygon, ring_index)
        return key, footprint.rings()[ring_index]

    def ring_of(self, key: RingKey) -> Ring:
        """The source ring of a key."""
        entity_id, polygon, ring_index = key
        footprint = next(f for f in self.footprints[entity_id] if f.polygon == polygon)
        return footprint.rings()[ring_index]

    def claim(self, key: RingKey, records: Sequence[BoundaryRecord]) -> str:
        """The records that claim one ring, and where the ring lies."""
        return (
            f"{self._claimants(key, [record.boundary_id for record in records])} select "
            f"the ring at {self._place(self.ring_of(key))}"
        )

    def _claimants(self, key: RingKey, names: Sequence[str]) -> str:
        """The records that claim one ring, with its Entity and the Entity's layer."""
        layer = self.layers.get(key[0])
        on = f" on layer {layer}" if layer else ""
        return f"{' and '.join(names)} (Entity {key[0]!r}{on})"

    def _selector_finding(
        self, record: BoundaryRecord, matches: Sequence[SourceFootprint]
    ) -> str:
        """Where a boundary's selector point lies, and the authored polygons of its Entity."""
        x, y = record.selector_point_um
        point = format_point(x, y, self.grid.size_um)
        authored = [f for f in self.footprints.get(record.entity_id, ()) if f.authored]
        if not authored:
            return f"the Entity owns no polygon; the selector point lies at {point}"
        if not matches:
            extent = self._place(tuple(p for f in authored for p in f.outer))
            return (
                f"the selector point lies at {point}, outside every polygon of the Entity"
                f"{self._on(record)}, which lie within {extent}"
            )
        where = ", ".join(self._place(f.outer) for f in matches)
        return f"the selector point lies at {point}, inside each of its polygons at {where}"

    def _on(self, record: BoundaryRecord) -> str:
        layer = self.layers.get(record.entity_id)
        return f" on layer {layer}" if layer else ""

    def _place(self, ring: Ring) -> str:
        """Where a lattice ring lies, as the box around it in micrometres."""
        xs, ys = (
            [self.grid.to_um(x) for x, _ in ring],
            [self.grid.to_um(y) for _, y in ring],
        )
        return format_box((min(xs), min(ys), max(xs), max(ys)), self.grid.size_um)

    def _point(self, point: Point) -> str:
        return format_point(
            self.grid.to_um(point[0]), self.grid.to_um(point[1]), self.grid.size_um
        )

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
                raise InputError(
                    f"{what} does not start where the previous curve ends: it starts at "
                    f"{self._point(start)}, the previous curve ends at "
                    f"{self._point(previous_end)}. Fix the layout: start each curve where "
                    "the one before it ends."
                )
            first_start = start if first_start is None else first_start
            previous_end = end
            primitives.extend(new)
        if first_start is None or previous_end != first_start:
            ends = (
                "it has no curve"
                if first_start is None or previous_end is None
                else f"the chain{self._on(record)} starts at {self._point(first_start)} "
                f"and ends at {self._point(previous_end)}"
            )
            raise InputError(
                f"{record.boundary_id} (Entity {record.entity_id!r}) curves do not form one "
                f"closed chain: {ends}. Fix the layout: end the last curve where the first "
                "starts."
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
            # A component's arc was decided in its own frame by the source stage.  This
            # decides an arc declared in assembly coordinates, and it is the last guard:
            # SGB solves the circle on these three points.
            if collinear(start, through, end):
                size = self.grid.size_um
                raise InputError(
                    arc_without_circle(
                        what, [format_point(x, y, size) for x, y in curve.points_um]
                    )
                )
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
            f"(this version carries {SUPPORTED_CURVES}){self._runs(curve)}. "
            "Fix the layout: draw the curve as lines and circular arcs."
        )

    def _runs(self, curve: SourceCurveSpec) -> str:
        """Where an authored curve starts and ends, as the layout author wrote it."""
        if not curve.points_um:
            return ""
        (x0, y0), (x1, y1) = curve.points_um[0], curve.points_um[-1]
        size = self.grid.size_um
        return (
            f"; the curve runs from {format_point(x0, y0, size)} "
            f"to {format_point(x1, y1, size)}"
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
                f"(this version carries {SUPPORTED_CURVES}); the ring it selects lies at "
                f"{self._place(ring)}. Fix the declaration: give the record kind "
                "'circular_arc', or remove it to keep the ring as drawn."
            )
        # The indices name the drawn ring: they are checked there, by the rule the
        # construction binder also states, then carried onto the ring this footprint holds,
        # which may start elsewhere, run the other way and hold crossings the grid noded in.
        drawn = self._drawn_ring(record, key, ring, what)

        def drawn_place(i: int) -> str:
            return self._point(drawn[i])

        refusal = reconstruction_chain_refusal(
            record, what, len(drawn), drawn_place, self._place(drawn)
        )
        if refusal is not None:
            raise InputError(refusal)
        segments = tuple(record.segment_indices)
        drawn_anchors = reconstruction_anchors(record, len(drawn))
        chain = drawn_chain_on_ring(segments, drawn_anchors, record.closed, drawn, ring)
        if chain is None:
            raise InputError(
                ring_lost_drawn_order(
                    what,
                    [
                        f"drawn vertex {i} at {drawn_place(i)} {fault}"
                        for i, fault in drawn_vertices_off_ring(drawn, ring)
                    ],
                )
            )
        first, middle, last = (ring[i] for i in chain.anchors)
        # The source stage decided anchors whose cell it read, on the vertices as drawn.
        # This decides the rest, an input built by hand, and it is the last guard: SGB
        # solves the circle on these three lattice points.
        if collinear(first, middle, last):
            raise InputError(
                reconstruction_without_circle(what, drawn_anchors, drawn_place)
            )
        segments = chain.segments
        arc_id = _arc_id(key, segments[0])
        lineage = Lineage("polygon_edge", key[0], key[1], key[2], segments[0])
        if record.closed:
            ccw = ring_twice_area(ring) > 0
            arcs = full_circle(
                arc_id, (first, middle, last), ccw, lineage, what, grid=self.grid
            )
        else:
            through = (Fraction(middle[0]), Fraction(middle[1]))
            arcs = arc_through_points(
                arc_id, first, through, last, lineage, what, grid=self.grid
            )
        return _Selection(segments, tuple(arcs), record.boundary_id)

    def _drawn_ring(
        self, record: GdsBoundaryReconstructionSpec, key: RingKey, ring: Ring, what: str
    ) -> Ring:
        """The ring the record's indices name, on the lattice, by the rule ``drawn_ring_of`` states.

        A footprint no polygon backs, as an Entity's outline loop, is given as drawn.
        """
        polygon = self.polygons.get((key[0], key[1]))
        if polygon is None:
            return ring
        drawn = drawn_ring_of(polygon, key[2])
        if drawn is None:
            raise InputError(ring_without_drawn_order(what, self._place(ring)))
        return tuple(
            self._lattice_point(point, f"{what} drawn ring") for point in drawn
        )

    def replace_segments(
        self, key: RingKey, selections: Sequence[_Selection]
    ) -> tuple[RingPrimitive, ...]:
        """Replace the selected source chains by their arcs and keep the unselected straight segments."""
        ring = self.ring_of(key)
        claimed: dict[int, str] = {}
        starts: dict[int, tuple[RingPrimitive, ...]] = {}
        for selection in selections:
            shared = [i for i in selection.segments if i in claimed]
            if shared:
                names = tuple(
                    dict.fromkeys(
                        (*(claimed[i] for i in shared), selection.boundary_id)
                    )
                )
                runs = ring_runs(shared, len(ring), lambda i: self._point(ring[i]))
                raise InputError(
                    f"overlapping reconstruction segment claims on ring {key}: "
                    f"{self._claimants(key, names)} both replace the ring {runs}. "
                    "Fix the declaration: give each drawn segment of the ring to one "
                    "reconstruction record."
                )
            claimed.update(dict.fromkeys(selection.segments, selection.boundary_id))
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

    # -- coordinates -------------------------------------------------------------------

    def _lattice_point(self, point_um: Sequence[float], what: str) -> Point:
        x, y = (
            self.grid.lattice_units(point_um[0]),
            self.grid.lattice_units(point_um[1]),
        )
        if x is None or y is None:
            raise InputError(
                f"{what}: end point ({point_um[0]}, {point_um[1]}) um is off the "
                f"{self.grid.size_um} um model grid. Fix the layout: draw the ends of each "
                "curve on the grid."
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


def _footprint_polygons(
    build_input: GeometryBuildInput,
) -> dict[tuple[str, int], LayoutPolygonSpec]:
    """Each Entity's source polygons by footprint index, as ``source_footprints`` numbers them."""
    by_id = {p.polygon_id: p for p in build_input.polygons}
    return {
        (entity.semantic_id, index): by_id[polygon_id]
        for entity in build_input.entities
        for index, polygon_id in enumerate(entity.polygon_ids)
        if polygon_id in by_id
    }


def _arc_id(key: RingKey, edge: int) -> str:
    return f"{key[0]}/polygon{key[1]}/ring{key[2]}/arc{edge}"
