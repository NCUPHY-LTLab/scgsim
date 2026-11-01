"""Partition each junction sheet into end A, a central RLC rectangle and end B.

A junction is an authored sheet bridging two terminal conductors. AEDT needs the
central rectangle (the lumped RLC) alone on the sheet, with whatever remains of the
sheet on each side attached to its terminal as a detached "end" polygon. This module
proves in a local integer frame that such a split exists, then asks `junction_model`
to reproduce it on the model polygons.

Owns: the per-junction local proof and the partition records it produces. Does not
own: exact predicates and Boolean wrappers (`junction_exact.py`), the model-coordinate
lowering (`junction_model.py`) or native and EPR analysis, which never read this
module's intermediate data.

Local frame: `s` runs along the junction direction (terminal A at small `s`, B at
large `s`) and `t` across it, both as integers in units of `PRECISION_UM` with the
sheet's lower-left corner at the origin. Inside the sheet, the "corridor" is the
stretch of `s` between the metal of the two terminals; each end cut must lie in it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction

from scgsim.aedt._data import canonical_sha256
from scgsim.aedt.preparation.junction_exact import (
    PRECISION_UM,
    boolean,
    closed_contact,
    covered,
    inside_ring,
    plain_with_dataclasses,
    positive_boundary_contact,
    rectangle,
    region_from_polygons,
    rings_from_region,
    section,
    twice_signed_area,
)
from scgsim.aedt.preparation.junction_model import model_geometry
from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.geometry import GeometryBuildInput, LayoutPolygonSpec

METHOD = "scgsim.aedt.junction-partition.v1"


def partition_junctions(
    build_input: GeometryBuildInput, junctions: Sequence[PlanarJunction], step: float
):
    """Detach AEDT ends under their source Entities, retaining authored polygons.

    `step` is the source grid pitch in micrometres; end cuts lie on that grid.
    """
    if not junctions:
        return build_input
    if "aedt_junction_partitions" in build_input.metadata:
        raise ValueError(
            "junction input is already partitioned; use immutable authored input"
        )
    originals = {p.polygon_id: p for p in build_input.polygons}
    originals.update(
        {
            r.source_polygon_id: LayoutPolygonSpec(
                r.source_polygon_id, r.source_layer, r.exterior, r.holes
            )
            for r in build_input.port_sheet_regions
        }
    )
    authored_ids = {j.source_polygon_id for j in junctions}
    entities = {e.semantic_id: e for e in build_input.entities}
    records = {}
    for junction in junctions:
        records[junction.junction_id] = _prove_junction(
            junction, originals, authored_ids, entities, step
        )
    model, derived_entities = model_geometry(build_input, originals, records)
    return _partitioned_input(
        build_input, records, model, derived_entities, authored_ids
    )


def _partitioned_input(build_input, records, model, derived_entities, authored_ids):
    """Rebuild the input with partitioned polygons, Entities and sheet regions.

    The authored input's hash is recorded so the partition can be traced back to it.
    """
    metadata = dict(build_input.metadata)
    metadata["aedt_authored_input_sha256"] = canonical_sha256(
        plain_with_dataclasses(build_input)
    )
    metadata["aedt_junction_partitions"] = records
    original_ids = {p.polygon_id for p in build_input.polygons}
    derived_polygons = tuple(model[p.polygon_id] for p in build_input.polygons) + tuple(
        p
        for pid, p in model.items()
        if pid not in original_ids and pid not in authored_ids
    )
    regions = tuple(
        replace(
            r,
            exterior=model[r.source_polygon_id].exterior,
            holes=model[r.source_polygon_id].holes,
        )
        if r.source_polygon_id in model
        else r
        for r in build_input.port_sheet_regions
    )
    return replace(
        build_input,
        polygons=derived_polygons,
        entities=derived_entities,
        port_sheet_regions=regions,
        metadata=metadata,
    )


@dataclass
class _LocalFrame:
    """The junction sheet's own axes and its size in grid units."""

    dx: float
    dy: float
    px: float
    py: float
    origin: tuple[float, float]
    span: int
    width: int
    deviations: list  # rounding error of every coordinate converted so far

    @classmethod
    def of_sheet(cls, sheet, junction):
        """Axes along the junction direction, anchored at the sheet's lower corner."""
        dx, dy = junction.direction_xy
        px, py = -dy, dx
        along = [x * dx + y * dy for x, y in sheet.exterior]
        across = [x * px + y * py for x, y in sheet.exterior]
        origin = (min(along), min(across))
        span = round((max(along) - origin[0]) / PRECISION_UM)
        width = round((max(across) - origin[1]) / PRECISION_UM)
        if span <= 0 or width <= 0:
            raise ValueError(
                "junction authored geometry collapses at Boolean precision"
            )
        return cls(dx, dy, px, py, origin, span, width, [])

    def to_local(self, polygon):
        """Convert a polygon to integer `(s, t)` rings, checking it stays valid."""
        rings = []
        for ring in (polygon.exterior, *polygon.holes):
            converted = []
            for x, y in ring:
                u = x * self.dx + y * self.dy - self.origin[0]
                v = x * self.px + y * self.py - self.origin[1]
                point = (round(u / PRECISION_UM), round(v / PRECISION_UM))
                self.deviations.append(
                    max(
                        abs(point[0] * PRECISION_UM - u),
                        abs(point[1] * PRECISION_UM - v),
                    )
                )
                converted.append(point)
            rings.append(tuple(converted))
        for index, ring in enumerate(rings):
            area = twice_signed_area(ring)
            if len(set(ring)) < 3 or area == 0:
                raise ValueError(
                    "numerical junction geometry failure: source polygon/hole collapses at Boolean precision"
                )
            if index and any(inside_ring(point, rings[0]) != 1 for point in ring):
                raise ValueError(
                    "numerical junction geometry failure: source hole loses interior topology"
                )
        return tuple(rings)


@dataclass
class _Terminals:
    """Local-frame conductor polygons of terminal A, terminal B and all other nets."""

    own_a: list
    own_b: list
    other: list


@dataclass
class _Corridor:
    """The sheet's free area and the clear stretch of `s` between the terminals."""

    free: list  # sheet minus both terminals' metal, as gdstk polygons
    lower: int  # largest `s` reached by terminal A's metal
    upper: int  # smallest `s` reached by terminal B's metal
    span: int
    width: int


def _prove_junction(junction, originals, authored_ids, entities, step):
    """Prove the A / central / B split of one junction and return its record."""
    frame = _LocalFrame.of_sheet(originals[junction.source_polygon_id], junction)
    by_entity = {
        e.semantic_id: [
            frame.to_local(originals[p]) for p in e.polygon_ids if p not in authored_ids
        ]
        for e in entities.values()
        if e.material_kind == "conductor"
    }
    terminals = _terminals(junction, by_entity, entities)
    corridor = _corridor(terminals, frame)
    choices, checks = _choose_end_cuts(corridor, terminals, step)
    (a, end_a, ka), (b, end_b, kb) = choices
    central = _prove_partition(corridor, terminals, choices)
    end_records = _end_records(
        junction,
        (
            ("A", end_a, a, junction.terminal_a_net),
            ("B", end_b, b, junction.terminal_b_net),
        ),
        by_entity,
        entities,
        corridor.width,
    )
    cid = f"aedt_junction_{canonical_sha256([junction.junction_id, 'central'])[:24]}"
    return {
        "method": METHOD,
        "authored_polygon_id": junction.source_polygon_id,
        "central_polygon_id": cid,
        "direction_xy": [frame.dx, frame.dy],
        "local_origin_projection_um": list(frame.origin),
        "source_dbu_um": step,
        "boolean_precision_um": PRECISION_UM,
        "span_um": frame.span * PRECISION_UM,
        "width_um": frame.width * PRECISION_UM,
        "corridor_um": [corridor.lower * PRECISION_UM, corridor.upper * PRECISION_UM],
        "cut_indices": [ka, kb],
        "actual_cuts_um": [a * PRECISION_UM, b * PRECISION_UM],
        "candidate_checks": checks,
        "candidate_count": len(checks),
        "proof_conditions": {
            "full_width_corridor_containment": True,
            "closed_corridor_side_clearance": True,
            "terminal_contact_ordering": "A_left_B_right",
            "partition_coverage_and_disjoint_interiors": True,
        },
        "maximum_input_coordinate_deviation_um": max(frame.deviations, default=0.0),
        "ends": end_records,
        "free_local_regions": plain_with_dataclasses(rings_from_region(corridor.free)),
        "central_local_regions": plain_with_dataclasses(central),
    }


def _terminals(junction, by_entity, entities):
    """Group the conductor polygons into terminal A, terminal B and everything else."""
    by_net = defaultdict(list)
    for eid, polygons in by_entity.items():
        by_net[entities[eid].net_id].extend(polygons)
    own_a, own_b = by_net[junction.terminal_a_net], by_net[junction.terminal_b_net]
    if not own_a or not own_b:
        raise ValueError(
            "junction requires both explicit terminal conductor geometries"
        )
    other = [
        p
        for net, polygons in by_net.items()
        if net not in {junction.terminal_a_net, junction.terminal_b_net}
        for p in polygons
    ]
    return _Terminals(own_a, own_b, other)


def _corridor(terminals, frame):
    """Find the corridor between the terminals and check the sheet around it.

    The corridor must be clear of metal on its two long sides and must lie wholly
    inside the free sheet, because no general search for other layouts is done.
    """
    span, width = frame.span, frame.width
    own_a, own_b, other = terminals.own_a, terminals.own_b, terminals.other
    if _contacts(other, span, width) is not None:
        raise ValueError("third-net closed contact with authored junction sheet")
    ca, cb = _contacts(own_a, span, width), _contacts(own_b, span, width)
    if ca is None or cb is None or ca[1] >= cb[0]:
        raise ValueError("unsupported junction contact ordering or clear corridor")
    lower, upper = ca[1], cb[0]
    for polygons in (own_a, own_b, other):
        for rings in polygons:
            for y in (0, width):
                intervals, points = section(rings, 1, y, lower, upper)
                if any(a < b for a, b in intervals) or any(
                    lower < v < upper for v in points
                ):
                    raise ValueError("unsupported corridor side-boundary metal contact")
    sheet_region = region_from_polygons(rectangle(0, 0, span, width))
    free = boolean(sheet_region, region_from_polygons([*own_a, *own_b]), "not")
    corridor = region_from_polygons(rectangle(lower, 0, upper, width))
    if boolean(corridor, free, "not"):
        raise ValueError(
            "unsupported corridor containment; no general search performed"
        )
    return _Corridor(free, lower, upper, span, width)


def _contacts(polygons, span, width):
    """Find extrema of area, segment and point contact with closed authored S.

    Returns the smallest and largest `s` at which the polygons touch the sheet
    rectangle (area, edge or point contact), or None if they never touch it.
    """
    region = boolean(
        region_from_polygons(polygons),
        region_from_polygons(rectangle(0, 0, span, width)),
        "and",
    )
    positions = [round(x / PRECISION_UM) for p in region for x, _ in p.points]
    for rings in polygons:
        for axis, c, low, high in (
            (0, 0, 0, width),
            (0, span, 0, width),
            (1, 0, 0, span),
            (1, width, 0, span),
        ):
            intervals, points = section(rings, axis, c, low, high)
            if axis == 0 and (intervals or points):
                positions.append(c)
            elif axis == 1:
                positions.extend(points)
                positions.extend(v for pair in intervals for v in pair)
    return (min(positions), max(positions)) if positions else None


def _choose_end_cuts(corridor, terminals, step):
    """Pick one valid grid cut per end, recording every candidate tried.

    Candidates are tried nearest the terminal metal first; the first one whose
    residual end passes `_end_candidate` is taken. Returns the two choices as
    `(cut, end polygons, grid index)` and the candidate evidence.
    """
    lower, upper = corridor.lower, corridor.upper
    choices = []
    checks = []
    for left, bound, own, foreign in (
        (True, lower, terminals.own_a, terminals.own_b + terminals.other),
        (False, upper, terminals.own_b, terminals.own_a + terminals.other),
    ):
        chosen = None
        for index in _candidate_indices(bound, step, left=left):
            requested, cut = _cut(index, step)
            interior = (
                (Decimal(index) * Decimal(str(step)) > _bound_decimal(bound))
                if left
                else (Decimal(index) * Decimal(str(step)) < _bound_decimal(bound))
            )
            if interior and cut == bound:
                raise ValueError(
                    "numerical junction geometry failure: interior grid cut collapses to boundary"
                )
            if (
                cut < lower
                or cut > upper
                or (interior and (cut <= lower or cut >= upper))
            ):
                continue
            end, evidence = _end_candidate(corridor, own, foreign, cut, left=left)
            checks.append(
                {
                    "side": "A" if left else "B",
                    "index": index,
                    "requested_s_um": requested,
                    "actual_s_um": cut * PRECISION_UM,
                    "deviation_um": cut * PRECISION_UM - requested,
                    **evidence,
                }
            )
            if end is not None:
                chosen = (cut, end, index)
                break
        if chosen is None:
            raise ValueError(
                "unsupported junction end topology on verified corridor/grid"
            )
        choices.append(chosen)
    return choices, checks


def _bound_decimal(bound):
    """A bound in grid units as an exact micrometre `Decimal`."""
    rational = Fraction(bound)
    return (
        Decimal(rational.numerator)
        / Decimal(rational.denominator)
        * Decimal(str(PRECISION_UM))
    )


def _candidate_indices(bound, step, *, left):
    """Grid indices to try for an end cut, nearest the terminal metal first.

    The grid line exactly on the metal bound comes first when the bound is on the
    grid; then the next grid line into the corridor.
    """
    q = _bound_decimal(bound) / Decimal(str(step))
    floor = int(q.to_integral_value(rounding=ROUND_FLOOR))
    aligned = q == floor
    if left:
        return ([floor] if aligned else []) + [floor + 1]
    return ([floor] if aligned else []) + [floor if not aligned else floor - 1]


def _cut(index, step):
    """Return the requested cut in micrometres and its rounded grid coordinate."""
    requested = float(Decimal(index) * Decimal(str(step)))
    return requested, round(requested / PRECISION_UM)


def _end_candidate(corridor, own, foreign, cut, *, left):
    """Test cutting the sheet at `cut` and keeping the free part on the terminal side.

    Returns `(end polygons or None, evidence)`. An empty end is valid when the cut
    sits on the metal bound. Each rejection says which condition failed.
    """
    span, width = corridor.span, corridor.width
    clip = rectangle(0, 0, cut, width) if left else rectangle(cut, 0, span, width)
    region = (
        boolean(corridor.free, region_from_polygons(clip), "and")
        if (cut > 0 if left else cut < span)
        else []
    )
    try:
        end = rings_from_region(region)
    except ValueError as exc:
        return None, {"valid": False, "reason": str(exc)}
    if len(end) > 1:
        return None, {
            "valid": False,
            "reason": "nonempty end has disconnected components",
            "components": len(end),
        }
    if end and not positive_boundary_contact(end, own):
        return None, {
            "valid": False,
            "reason": "end lacks positive-length own-arm contact",
        }
    if closed_contact(end, foreign):
        return None, {
            "valid": False,
            "reason": "end contacts wrong terminal or third net",
        }
    if not covered([*own, *end], 0, cut, 0, width):
        return None, {
            "valid": False,
            "reason": "designated end edge is not wholly covered",
        }
    return end, {"valid": True, "empty_residual": not end, "components": len(end)}


def _prove_partition(corridor, terminals, choices):
    """Check the A / central / B split at the chosen cuts; return the central polygon."""
    (a, end_a, _), (b, end_b, _) = choices
    width = corridor.width
    own_a, own_b, other = terminals.own_a, terminals.own_b, terminals.other
    if a >= b:
        raise ValueError("junction DBU grid has no positive central span")
    central = rectangle(a, 0, b, width)
    for y in (0, width):
        for rings in [*own_a, *own_b, *other, *end_a, *end_b]:
            intervals, points = section(rings, 1, y, a, b)
            if any(x < z for x, z in intervals) or any(a < v < b for v in points):
                raise ValueError("RLC open side has forbidden line or point contact")
    for cut, wrong in ((a, own_b + other + end_b), (b, own_a + other + end_a)):
        if any(any(section(rings, 0, cut, 0, width)) for rings in wrong):
            raise ValueError("wrong-net RLC end/corner contact")
    parts = [
        region_from_polygons(end_a),
        region_from_polygons(central),
        region_from_polygons(end_b),
    ]
    union = boolean([p for part in parts for p in part], [], "or")
    if boolean(union, corridor.free, "xor") or any(
        boolean(parts[i], parts[j], "and") for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError(
            "numerical junction geometry failure: partition coverage/interiors differ"
        )
    return central


def _end_records(junction, ends, by_entity, entities, width):
    """Describe each end and the single source Entity that owns it.

    An end belongs to the one Entity on its net that it touches along a boundary;
    an empty end belongs to the one Entity covering the whole cut edge.
    """
    end_records = []
    for label, end, cut, net in ends:
        owners = [
            eid
            for eid, polygons in by_entity.items()
            if entities[eid].net_id == net
            and (
                positive_boundary_contact(end, polygons)
                if end
                else covered(polygons, 0, cut, 0, width)
            )
        ]
        if len(owners) != 1:
            raise ValueError(
                "junction end lacks an unambiguous source Entity/material/Z lineage"
            )
        owner = entities[owners[0]]
        polygon_id = (
            f"aedt_junction_{canonical_sha256([junction.junction_id, label, owner.semantic_id])[:24]}"
            if end
            else None
        )
        end_records.append(
            {
                "side": label,
                "source_entity_id": owner.semantic_id,
                "net_id": net,
                "material_id": owner.material_id,
                "geometry": plain_with_dataclasses(owner.geometry),
                "representation": dict(owner.route_representations),
                "polygon_id": polygon_id,
                "empty_residual": not end,
                "local_regions": plain_with_dataclasses(end),
            }
        )
    return end_records
