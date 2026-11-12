"""The cut rules: where a cut line may lie, and the located refusal of one that may not.

A 2D model stands for an infinitely long, uniform line with this cross-section, so the section must be the
section of such a line (the decisions on issue 188):

* the cut crosses every curve it meets -- the source edges and Surface-EPR band seams of every emitted face --
  between the curve's ends and at a right angle (an oblique section widens the gaps and gives a plausible but
  wrong L and C);
* it runs along no edge, passes through no corner (a vertex of the layout) and crosses no arc; a cut that comes
  within :data:`ARC_CLEARANCE` of an arc counts as crossing it, because a realised arc may lie that far from its
  source circle and its polyline witness;
* it lies within the model's planar extent, vacuum region included, and both its ends lie on the extent's
  edge, so the section is the whole cross-section and each end edge is the trace of the model's boundary;
* in the section, nothing excluded from the model (a conductor, or the space outside) touches another such
  region only at a point, which would pinch the 2D domain.

Each refusal is an :class:`~scgsim.geometry.sgb.errors.InputError` that names what the cut meets (the Entity and
its source layer, or the band family and margin), the level, the place in micrometres and the fix, and counts
the further places along the cut where a rule is broken.  The place reported is the one nearest the cut's start.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import NoReturn

from scgsim.documentation import see_also
from scgsim.geometry.models.grid import format_box, format_point
from scgsim.geometry.sgb.assembly.records import TopoEdge, Topology
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.labelling.prism_set import named_entities
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import Lineage, base_carrier_id
from scgsim.geometry.sgb.records.labels import Domain
from scgsim.geometry.sgb.section.frame import Along, CutFrame

ARC_CLEARANCE = 0.5
"""How near an arc a cut may pass without crossing it, in grid units: the snapping of the arrangement's
vertices moves a curve by less than this, and an arc's polyline witness lies within a quarter of it."""

_EXPLAINED = ("docs/geometry-sgb", "cut-rules")
_KIND_ORDER = ("along", "corner", "oblique", "arc")


def curves_of_emitted_faces(model: ConformalModel) -> list[str]:
    """The horizontal edges in the loops of every emitted face, sorted: the curves a section can meet."""
    topology, routed = model.topology, model.routed
    found: set[str] = set()
    for face in topology.faces.values():
        if not routed.is_emitted_face(face.neg_volume, face.pos_volume):
            continue
        for loop in face.loops:
            found.update(
                edge_id
                for edge_id, _ in loop
                if topology.edges[edge_id].kind == "horizontal"
            )
    return sorted(found)


# ---------------------------------------------------------------------------- the cut within the extent


def check_within_extent(
    frame: CutFrame, topology: Topology, curves: Sequence[str]
) -> None:
    """Refuse a cut with an end beyond the model's planar extent, the box around every curve of the model."""
    box = planar_extent(topology, curves)
    for u in (0, frame.length):
        x, y = frame.plan_point(u)
        if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
            raise InputError(
                f"{frame.text()} runs outside the model: its end {frame.point_text(u)} lies beyond the "
                f"model's planar extent {_box_text(frame, box)}, vacuum region included. Fix the cut "
                "line: put both ends on the edge of that extent."
                + see_also(*_EXPLAINED)
            )


def planar_extent(
    topology: Topology, curves: Sequence[str]
) -> tuple[int, int, int, int]:
    """The box ``(x0, y0, x1, y1)`` around the ends of the given edges, in grid units."""
    points = [
        topology.vertices[vertex_id]
        for edge_id in curves
        for vertex_id in (topology.edges[edge_id].v0, topology.edges[edge_id].v1)
    ]
    xs, ys = [int(p.x) for p in points], [int(p.y) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def refuse_open_end(  # noqa: PLR0913 - the cut, the model, the end and what continues past it
    frame: CutFrame,
    model: ConformalModel,
    curves: Sequence[str],
    u: int,
    z_range: tuple[Fraction, Fraction],
    beyond: Domain | None,
) -> NoReturn:
    """Refuse a cut whose end at ``u`` lies inside the model: ``beyond`` continues past it over ``z_range``."""
    box = _box_text(frame, planar_extent(model.topology, curves))
    what = (
        "the model" if beyond is None else CurveNames(model).domains([beyond], z_range)
    )
    raise InputError(
        f"{frame.text()} ends at {frame.point_text(u)}, inside the model: {what} continues past it at "
        f"z {_z(z_range[0])}..{_z(z_range[1])} um. A 2D model is the whole cross-section of the line, so "
        "the cut runs from one edge of the model's planar extent to the other. Fix the cut line: extend "
        f"it to the edge of the extent {box}, vacuum region included."
        + see_also(*_EXPLAINED)
    )


def refuse_outside_stretch(
    frame: CutFrame,
    model: ConformalModel,
    curves: Sequence[str],
    u0: int,
    u1: int,
) -> NoReturn:
    """Refuse a cut that leaves the model between ``u0`` and ``u1``: nothing of the model lies there at any level."""
    box = _box_text(frame, planar_extent(model.topology, curves))
    raise InputError(
        f"{frame.text()} runs outside the model over {frame.span_text(u0, u1)}: no part of the model lies "
        f"there at any level, although it lies within the box {box} around the model. Fix the cut line: "
        "cut where the model's planar extent, vacuum region included, covers the whole cut."
        + see_also(*_EXPLAINED)
    )


def _box_text(frame: CutFrame, box: tuple[int, int, int, int]) -> str:
    to_um = frame.grid.to_um
    return format_box(tuple(to_um(v) for v in box), frame.grid.size_um)


# ---------------------------------------------------------------------------- crossings of the curves


@dataclass(frozen=True)
class _Finding:
    """One place where the cut breaks a rule at an edge of the model."""

    u: Along  # where along the cut, in grid units
    z: Fraction
    kind: str  # one of _KIND_ORDER
    edge: str
    key: str  # what makes two findings one place: the edge, or the vertex of a corner
    until: Along = 0  # "along": the end of the stretch the cut runs along the edge
    angle: float = 0.0  # "oblique": the angle between the edge and the cut, in degrees


def cross_curves(
    frame: CutFrame, model: ConformalModel, curves: Sequence[str]
) -> dict[str, int]:
    """Edge id -> ``u`` of every one of ``curves`` that the cut crosses, all of them at a right angle.

    A cut that breaks a crossing rule of the module docstring at any of them is refused, at the place nearest
    its start.
    """
    topology = model.topology
    crossings: dict[str, int] = {}
    findings: list[_Finding] = []
    for edge_id in curves:
        found = _cross(frame, topology, topology.edges[edge_id])
        if isinstance(found, _Finding):
            findings.append(found)
        elif found is not None:
            crossings[edge_id] = found
    if findings:
        _refuse_crossing(frame, model, curves, findings)
    return crossings


def _cross(
    frame: CutFrame, topology: Topology, edge: TopoEdge
) -> int | _Finding | None:
    """Where the cut crosses one horizontal edge, a finding where that breaks a rule, or ``None``."""
    v0, v1 = topology.vertices[edge.v0], topology.vertices[edge.v1]
    a, b = (int(v0.x), int(v0.y)), (int(v1.x), int(v1.y))
    oa, ob = frame.offset(a), frame.offset(b)
    if edge.curve is None and oa * ob > 0:
        return None  # both ends on one side of the cut's line: the usual case, decided first
    ua, ub = frame.u_of(a), frame.u_of(b)
    if edge.curve is None and oa == ob == 0:
        return _collinear(edge, v0.z, ua, ub, frame.length)
    for offset, u, vertex in ((oa, ua, edge.v0), (ob, ub, edge.v1)):
        if offset == 0 and 0 <= u <= frame.length:
            return _Finding(u, v0.z, "corner", edge.id, vertex)
    if edge.curve is not None:
        return _cross_arc(frame, edge, (a, b), v0.z)
    if oa == 0 or ob == 0:
        return None  # it ends on the cut's line beyond the cut
    if ua == ub:
        return int(ua) if 0 <= ua <= frame.length else None
    u = ua + (ub - ua) * Fraction(int(oa), int(oa - ob))
    if not 0 <= u <= frame.length:
        return None
    return _Finding(u, v0.z, "oblique", edge.id, edge.id, angle=_angle(frame, a, b))


def _collinear(
    edge: TopoEdge, z: Fraction, ua: Along, ub: Along, length: int
) -> _Finding | None:
    """An edge on the cut's line: the stretch of the cut along it, a corner where it only touches the cut, or nothing."""
    low, high = max(min(ua, ub), 0), min(max(ua, ub), length)
    if low < high:
        return _Finding(low, z, "along", edge.id, edge.id, until=high)
    if low == high:
        return _Finding(low, z, "corner", edge.id, edge.v0 if ua == low else edge.v1)
    return None


def _angle(frame: CutFrame, a: Point, b: Point) -> float:
    """The angle between the edge ``a -> b`` and the cut, in degrees (90 at a right angle)."""
    along = abs(b[frame.along] - a[frame.along])
    across = abs(b[frame.across] - a[frame.across])
    return math.degrees(math.atan2(across, along))


def _cross_arc(
    frame: CutFrame, edge: TopoEdge, ends: tuple[Point, Point], z: Fraction
) -> _Finding | None:
    """The finding of an arc edge that the cut crosses or passes within :data:`ARC_CLEARANCE` of."""
    arc = edge.curve
    assert arc is not None
    centre_offset = float(frame.offset(arc.centre))
    if abs(centre_offset) > arc.radius + ARC_CLEARANCE:
        return None
    # Where the circle meets the cut's line; a circle that only comes near it is nearest at its foot point.
    half = math.sqrt(max(0.0, arc.radius**2 - centre_offset**2))
    centre_u = float(frame.u_of(arc.centre))
    for u in sorted({centre_u - half, centre_u + half}):
        on_cut = -ARC_CLEARANCE <= u <= frame.length + ARC_CLEARANCE
        if on_cut and _on_arc(edge, ends, frame.plan_point(u)):
            return _Finding(u, z, "arc", edge.id, edge.id)
    return None


def _on_arc(
    edge: TopoEdge, ends: tuple[Point, Point], point: tuple[Along, Along]
) -> bool:
    """Whether a point near an arc's circle lies within the arc's sweep, give or take the clearance."""
    arc = edge.curve
    assert arc is not None
    cx, cy = arc.centre
    turn = 2 * math.pi

    def angle(p: tuple[Along, Along]) -> float:
        return math.atan2(float(p[1]) - cy, float(p[0]) - cx)

    start, end, at = angle(ends[0]), angle(ends[1]), angle(point)
    sweep = (end - start) % turn if arc.ccw else (start - end) % turn
    reach = (at - start) % turn if arc.ccw else (start - at) % turn
    slack = ARC_CLEARANCE / arc.radius
    return reach <= sweep + slack or reach >= turn - slack


def _refuse_crossing(
    frame: CutFrame,
    model: ConformalModel,
    curves: Sequence[str],
    findings: list[_Finding],
) -> NoReturn:
    """Raise the refusal of the finding nearest the cut's start, counting the other places."""
    first = min(findings, key=lambda f: (f.u, f.z, _KIND_ORDER.index(f.kind), f.edge))
    others = len({(f.kind, f.key) for f in findings}) - 1
    topology = model.topology
    names = CurveNames(model)
    edge = topology.edges[first.edge]
    at = f"{frame.point_text(first.u)} on level z = {_z(first.z)} um"
    if first.kind == "along":
        message = (
            f"{frame.text()} runs along an edge of {names.curves([edge], first.z)} on level z = "
            f"{_z(first.z)} um, over {frame.span_text(first.u, first.until)}: the section there lies on the "
            "edge, so it has no 2D model. Fix the cut line: move it off the edge, so that it crosses the "
            "line's edges at a right angle."
        )
    elif first.kind == "corner":
        meeting = [
            topology.edges[e]
            for e in curves
            if first.key in (topology.edges[e].v0, topology.edges[e].v1)
        ]
        message = (
            f"{frame.text()} passes through a corner at {at}, where edges of "
            f"{names.curves(meeting, first.z)} meet: the section through a corner is not the section of a "
            "uniform line. Fix the cut line: move it along the line to a straight, uniform section, where it "
            "crosses each edge between its ends."
        )
    elif first.kind == "oblique":
        message = (
            f"{frame.text()} crosses an edge of {names.curves([edge], first.z)} at {at} at "
            f"{first.angle:.1f} degrees instead of a right angle: an oblique section widens the line's gaps "
            "and gives a plausible but wrong L and C. Fix the cut line: cut the line where its edges run "
            "straight across the cut, which runs along x or y."
        )
    else:
        message = _arc_message(frame, names, edge, at, first.z)
    if others:
        message += f" The cut line breaks these rules at {others} more place{'s' if others > 1 else ''}."
    raise InputError(message + see_also(*_EXPLAINED))


def _arc_message(
    frame: CutFrame, names: CurveNames, edge: TopoEdge, at: str, z: Fraction
) -> str:
    arc = edge.curve
    assert arc is not None
    what = names.curves([edge], z)
    if names.is_band_seam(edge):
        return (
            f"{frame.text()} crosses the arc of {what} at {at}, where the seam turns round a corner or the "
            "end of the face it bands: the band seams of a uniform line are straight. Fix the cut line: move "
            "it onto a straight section of the line, farther than the largest margin from the corner."
        )
    to_um = frame.grid.to_um
    centre = format_point(
        to_um(arc.centre[0]), to_um(arc.centre[1]), frame.grid.size_um
    )
    return (
        f"{frame.text()} crosses an arc of {what} at {at} (centre {centre}, radius "
        f"{to_um(arc.radius):.4g} um): a uniform line has straight edges, so the section of a bend is not "
        "the section of a line. Fix the cut line: move it onto a straight section of the line."
    )


# ---------------------------------------------------------------------------- contacts at a point


def refuse_point_contact(  # noqa: PLR0913 - the cut, the model, the point and what touches there
    frame: CutFrame,
    model: ConformalModel,
    u: int,
    z: Fraction,
    excluded: Sequence[Domain | None],
    itself: Domain | None = None,
) -> NoReturn:
    """Refuse a section pinched at the point ``(u, z)``.

    ``excluded`` are the regions excluded from the model (conductor domains, ``None`` for the outside) that touch
    there only at the point; ``itself`` is instead the domain of a face that touches itself there.
    """
    names = CurveNames(model)
    if itself is not None:
        what = f"{names.domains([itself], (z, z))} where it touches itself"
    else:
        conductors = [d for d in excluded if d is not None]
        what = names.domains(conductors, (z, z)) if conductors else "the model's edge"
        if None in excluded and conductors:
            what += " and the edge of the model"
        what += " where they touch"
    raise InputError(
        f"{frame.text()} cuts {what} only at a point, {frame.point_text(u)} at z = {_z(z)} um: there the "
        "2D model's domain narrows to a point, which a mesh cannot join. Fix the layout or the cut line: "
        "draw the conductors so that they overlap along an edge or stay apart, or move the cut line to "
        "where they do." + see_also(*_EXPLAINED)
    )


# ---------------------------------------------------------------------------- names in messages


class CurveNames:
    """Who a curve or a domain of the model belongs to, as the refusals name it."""

    def __init__(self, model: ConformalModel) -> None:
        catalogue = model.labelled.catalogue
        self._grid = catalogue.grid
        self._prisms = catalogue.prisms
        self._lineage: dict[str, Lineage] = {}
        for prism in catalogue.prisms:
            for carrier in prism.carriers():
                self._lineage[base_carrier_id(carrier.id)] = carrier.lineage
            for arc in prism.arcs:
                self._lineage[arc.id] = arc.lineage
        for family in catalogue.families:
            for carrier in family.carriers:
                self._lineage[carrier.id] = carrier.lineage
            for arc in family.arcs:
                self._lineage[arc.id] = arc.lineage

    def curves(self, edges: Iterable[TopoEdge], z: Fraction) -> str:
        """The Entities (with their source layers) and the band seams that drew ``edges`` at level ``z``."""
        lineages = [
            self._lineage[base]
            for edge in edges
            for carrier in edge.sources
            if (base := base_carrier_id(carrier)) in self._lineage
        ]
        entities = sorted(
            {ln.entity_id for ln in lineages if ln.kind == "polygon_edge"}
        )
        parts = [self._entities(entities, (z, z))] if entities else []
        for family, margin in sorted(
            {
                (ln.family_id, ln.band_member.margin if ln.band_member else None)
                for ln in lineages
                if ln.kind == "family_member"
            },
            key=lambda item: (item[0], item[1] or 0),
        ):
            parts.append(self._family(family, margin))
        return " and ".join(parts) or "the layout"

    def is_band_seam(self, edge: TopoEdge) -> bool:
        """Whether an edge comes from a Surface-EPR band seam."""
        return any(
            (ln := self._lineage.get(base_carrier_id(c))) is not None
            and ln.band_member is not None
            for c in edge.sources
        )

    def domains(
        self, domains: Iterable[Domain], z_range: tuple[Fraction, Fraction]
    ) -> str:
        """The owning Entities of ``domains``, each with its source layer where it reaches ``z_range``."""
        return self._entities(sorted({d.name for d in domains}), z_range)

    def _entities(
        self, entities: Sequence[str], z_range: tuple[Fraction, Fraction]
    ) -> str:
        wanted = set(entities)
        prisms = [
            p
            for p in self._prisms
            if p.entity_id in wanted and p.z0 <= z_range[1] and z_range[0] <= p.z1
        ]
        named = {p.entity_id for p in prisms}
        listed = [named_entities(prisms)] if prisms else []
        listed += [e for e in entities if e not in named]
        return ", ".join(listed)

    def _family(self, family: str, margin: int | None) -> str:
        if margin is None:
            return f"the curve family {family!r}"
        return f"the Surface-EPR band seam of family {family!r} at margin {self._grid.to_um(margin):g} um"


def _z(z: Fraction) -> str:
    return f"{float(z):g}"
