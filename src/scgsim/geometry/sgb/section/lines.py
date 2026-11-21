"""The line cross-section a plane section shows: its conductor pieces with their roles, its gaps and dielectrics.

A 2D model stands for a uniform line.  Beyond the model, a 2D solver needs to know which conductor is the signal
and which the reference, and where the gaps between them lie: Palace selects its mode and draws its voltage
paths across the gaps, and Q2D groups its conductors by role.  :func:`line_cross_section` reads them off a
:class:`~scgsim.geometry.sgb.section.records.SectionModel` and returns the detached
:class:`~scgsim.geometry.models.section.LineCrossSection`.

* **Pieces.**  Each region a conductor fills in the section (``SectionModel.conductors``) is one piece, with its
  Entity, Net and material; a Route B conductor, which is no face, is found as a Route C conductor face is.
* **Roles from the structure.**  The Net found on both outer sides of the line is the reference, and the piece
  between them is the signal.  Nothing is declared and nothing is guessed: a section whose structure does not
  decide the roles is refused.
* **What this version classifies.**  One signal piece between two reference pieces, side by side in one film, on
  a cut that crosses nothing else: a coplanar waveguide (CPW).  Any other section is an
  :class:`~scgsim.geometry.exact.errors.InputError` that lists the pieces found, each with its Entity, source
  layer, Net and place in micrometres, and says who must change what.
* **Gaps and dielectrics.**  A gap is the stretch of the cut between two neighbouring pieces, across their film;
  the dielectrics are the section's dielectric and vacuum faces.

Every range comes from the section's exact coordinates, in micrometres, and nothing read here depends on the
route: the Route B and Route C sections of one plan at one cut give equal records.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import NamedTuple

from scgsim.geometry.exact.errors import InputError
from scgsim.geometry.models.section import (
    DielectricKind,
    LineConductor,
    LineCrossSection,
    LineDielectric,
    LineGap,
    Span,
)
from scgsim.geometry.sgb.section.records import SectionConductor, SectionModel
from scgsim.geometry.sgb.section.rules import CurveNames

LINE_PIECES = 3
"""A CPW's conductor pieces on the cut: a reference piece, the signal, a reference piece."""

_SCOPE = (
    "A line cross-section in this version is one signal piece between two pieces of one reference Net, side by "
    "side in one film, on a cut that crosses nothing else (a coplanar waveguide)."
)
_MOVE_THE_CUT = "Fix the cut line: move it to where it crosses only the line, or change the layout there."
_SET_NETS = "GeometryPlan.set_nets"


def line_cross_section(section: SectionModel) -> LineCrossSection:
    """The classified line cross-section ``section`` shows; any other section is refused (see the module docstring)."""
    _check_no_sheets(section)
    problem = _scope_problem(section.conductors)
    if problem is not None:
        raise InputError(
            f"{section.frame.text()} crosses {_found(section)}. {problem.reason} {_SCOPE} {problem.fix}"
        )
    # The scope holds: every piece has one domain, and the first piece's Net is the one on both outer sides.
    reference = section.conductors[0].domains[0].net
    conductors = tuple(
        _conductor(section, conductor, reference) for conductor in section.conductors
    )
    gaps = tuple(
        _gap(section, before, after) for before, after in pairwise(section.conductors)
    )
    return LineCrossSection(section.cut, conductors, gaps, _dielectrics(section))


def _check_no_sheets(section: SectionModel) -> None:
    """Refuse a section that crosses a zero-thickness sheet: it has no cross-section this version classifies."""
    sheets = sorted(
        (
            section.vertices[edge.v0].u,
            edge.owner.kind,
            edge.owner.name,
            section.vertices[edge.v0].z,
        )
        for edge in section.edges.values()
        if edge.owner is not None
    )
    if not sheets:
        return
    u, kind, name, z = sheets[0]
    frame = section.frame
    raise InputError(
        f"{frame.text()} crosses the {kind} sheet {name} at {frame.point_text(u)} on level z = {float(z):g} um"
        + (f" and {len(sheets) - 1} more sheet traces" if len(sheets) > 1 else "")
        + ". A sheet has no thickness, so it is no conductor piece and no gap; a conductor sheet comes from "
        f"Route A and a port sheet from the plan's ports. {_SCOPE} Fix the model or the cut line: build the "
        "model on Route B or C, and cut the line away from its port sheets."
    )


class _Problem(NamedTuple):
    """Why a section is no line cross-section, and who must change what."""

    reason: str
    fix: str


def _scope_problem(conductors: tuple[SectionConductor, ...]) -> _Problem | None:
    """The first reason the conductors on the cut are not one CPW, in the order a reader would fix them.

    Every conductor has a Net (SGB builds no conductor without one), and conductors of different Nets never
    touch (SGB refuses the contact), so two Entities in one region are of one Net.
    """
    for conductor in conductors:
        if len(conductor.domains) > 1:
            names = " and ".join(d.name for d in conductor.domains)
            return _Problem(f"{names} touch, with no gap between them.", _MOVE_THE_CUT)
    count = len(conductors)
    if count != LINE_PIECES:
        return _Problem(_count_reason(count), _MOVE_THE_CUT)
    if len({conductor.z for conductor in conductors}) > 1:
        return _Problem(
            "The pieces do not lie side by side in one film: they span different heights.",
            _MOVE_THE_CUT,
        )
    for before, after in pairwise(conductors):
        if before.u[1] >= after.u[0]:
            return _Problem(
                f"{before.domains[0].name} and {after.domains[0].name} touch, with no gap between them.",
                _MOVE_THE_CUT,
            )
    first, middle, last = (conductor.domains[0].net for conductor in conductors)
    if first != last:
        return _Problem(
            f"The outer pieces are of different Nets ({first}, {last}), so the structure does not say which "
            "Net is the reference.",
            f"Fix the plan's Nets with {_SET_NETS}, or move the cut to where one Net lies on both sides of the "
            "line.",
        )
    if middle == first:
        return _Problem(
            f"All three pieces are of Net {first}, so none is a signal.",
            f"Fix the plan's Nets with {_SET_NETS}, or move the cut to the line.",
        )
    return None


_TOO_FEW = {
    0: "A line needs conductors on the cut.",
    1: "One conductor shows no gap.",
    2: "Two conductors show a single gap: a line has a gap on each side of its signal.",
}


def _count_reason(count: int) -> str:
    """Why ``count`` conductor pieces are not a CPW's three."""
    return _TOO_FEW.get(
        count,
        f"That is {count - LINE_PIECES} more than a line's three: something besides the line lies on the cut.",
    )


def _found(section: SectionModel) -> str:
    """The conductor pieces on the cut, each with its Entity, source layer, Net and place, for a refusal."""
    conductors = section.conductors
    if not conductors:
        return "no conductor piece"
    names = CurveNames(section.source)
    frame = section.frame
    listed = []
    for conductor in conductors:
        nets = ", ".join(sorted({d.net for d in conductor.domains}))
        listed.append(
            f"{names.domains(conductor.domains, conductor.z)} (Net {nets}) at "
            f"{frame.span_text(*conductor.u)}, z {float(conductor.z[0]):g}..{float(conductor.z[1]):g} um"
        )
    count = len(conductors)
    return f"{count} conductor piece{'s' if count > 1 else ''}: " + "; ".join(listed)


# --------------------------------------------------------------------------- the records


def _conductor(
    section: SectionModel, conductor: SectionConductor, reference: str
) -> LineConductor:
    """One piece as the record holds it: the Net of both outer sides is the reference, the others signals."""
    domain = conductor.domains[0]
    return LineConductor(
        entity=domain.name,
        net=domain.net,
        role="reference" if domain.net == reference else "signal",
        material=domain.material,
        u_um=_u_span(section, conductor.u),
        z_um=_z_span(conductor.z),
    )


def _gap(
    section: SectionModel, before: SectionConductor, after: SectionConductor
) -> LineGap:
    """The gap from the wall of ``before`` to the wall of ``after``, across the film both lie in."""
    return LineGap(
        u_um=_u_span(section, (before.u[1], after.u[0])),
        z_um=_z_span(before.z),
        nets=(before.domains[0].net, after.domains[0].net),
    )


def _dielectrics(section: SectionModel) -> tuple[LineDielectric, ...]:
    """The dielectric and vacuum faces of the section, ordered by height, then along the cut."""
    found = []
    for face in section.faces.values():
        domain = face.domain
        if domain.kind == "conductor":
            continue
        kind: DielectricKind = "vacuum" if domain.kind == "vacuum" else "dielectric"
        corners = [
            section.vertices[vertex]
            for edge_id, _ in face.loops[0]
            for vertex in (section.edges[edge_id].v0, section.edges[edge_id].v1)
        ]
        us = [corner.u for corner in corners]
        zs = [corner.z for corner in corners]
        found.append(
            LineDielectric(
                entity=domain.name,
                kind=kind,
                material=domain.material,
                u_um=_u_span(section, (min(us), max(us))),
                z_um=_z_span((min(zs), max(zs))),
            )
        )
    return tuple(sorted(found, key=lambda d: (d.z_um, d.u_um, d.entity)))


def _u_span(section: SectionModel, u: tuple[int, int]) -> Span:
    """A range along the cut in micrometres, from its exact grid units (so 86000 units of 1 nm are 86.0 um)."""
    size = Fraction(repr(section.frame.grid.size_um))
    return (float(u[0] * size), float(u[1] * size))


def _z_span(z: tuple[Fraction, Fraction]) -> Span:
    return (float(z[0]), float(z[1]))
