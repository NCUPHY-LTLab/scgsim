"""Unsupported geometry: sample layouts SCGSim refuses, drawn on Supported Geometry and pinned at their refusal.

Each sample is drawn directly as SGB records on the 1 nm grid, like the supported samples of
``sgb_supported_kit``, and banded the same way (MA, MS and SA at 0.5 and 1.0 um).  Each is refused with a
message that names the conductor and the place:

* :func:`arc_cut_by_the_die_edge` -- a curved strip whose two long edges are exact arcs runs into the die edge.
  SGB continues a conductor cut by the die edge along a straight edge only, so the build is refused
  (Issue #138).
* :func:`strip_into_notch` -- a strip cut on the floor of a notch in the die's top edge; continued past the die
  edge, its right side ends 0.1 um from the notch wall, inside the band margin, so the build is refused
  (Issue #138).
* :func:`straight_lead` -- an isolated lead along x.  At 2.004 um, banded at 1.0 um, it keeps a core 4 grid
  steps wide, which the meshing stage refuses; at exactly 2 um it has no core and is modeled (Issue #128).
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from sgb_supported_kit import (
    METAL_TOP,
    Conductor,
    Point,
    Scene,
    catalogue,
    die_footprint,
    lattice,
    to_lattice,
)

from scgsim.geometry.sgb.records.carriers import ArcCarrier, Lineage, LineEdge
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismCatalogue, PrismRecord

# --------------------------------------------------------------------------- an arc cut by the die edge

# The strip lies between two circles about one centre, from the die's left edge (x = 0) round to y = 0.  Both
# circles cross x = 0 at lattice points (6-2.5-6.5 and 6-8-10 triangles), so the arcs end exactly on the die edge.
ARC_DIE = (0.0, -4.0, 20.0, 14.0)
ARC_CENTRE_UM = (6.0, 0.0)
ARC_INNER_R_UM = 6.5
ARC_OUTER_R_UM = 10.0
# Where the inner arc meets the die edge, 157.4 degrees round from +x, and where the outer arc does, 126.9 degrees.
ARC_INNER_CUT_UM = (0.0, 2.5)
ARC_OUTER_CUT_UM = (0.0, 8.0)


def arc_cut_by_the_die_edge() -> PrismCatalogue:
    """The curved strip on its die: substrate below z = 0, vacuum above, the strip 0.2 um of metal on z = 0."""
    die = catalogue(Scene("arc_cut_by_the_die_edge", (), ARC_DIE, []))
    footprint, arcs = Footprint.from_primitives(curved_strip_ring())
    strip = PrismRecord(
        "strip",
        footprint,
        exact_z(0),
        exact_z(METAL_TOP),
        "Al",
        "conductor",
        net_id="strip",
        arcs=arcs,
    )
    return replace(die, prisms=(*die.prisms, strip))


def curved_strip_ring() -> list[LineEdge | ArcCarrier]:
    """The strip's outline, counter-clockwise: its straight end on y = 0, the outer arc, the cut, the inner arc.

    Each arc is given in two pieces of at most a quarter turn, split at the top of its circle.
    """
    cx, cy = (lattice(v) for v in ARC_CENTRE_UM)
    inner, outer = lattice(ARC_INNER_R_UM), lattice(ARC_OUTER_R_UM)
    inner_cut, outer_cut = to_lattice([ARC_INNER_CUT_UM, ARC_OUTER_CUT_UM])
    return [
        LineEdge((cx + inner, cy), (cx + outer, cy), 0),
        _strip_arc(1, outer, (cx + outer, cy), (cx, cy + outer), ccw=True),
        _strip_arc(2, outer, (cx, cy + outer), outer_cut, ccw=True),
        LineEdge(outer_cut, inner_cut, 3),
        _strip_arc(4, inner, inner_cut, (cx, cy + inner), ccw=False),
        _strip_arc(5, inner, (cx, cy + inner), (cx + inner, cy), ccw=False),
    ]


def _strip_arc(
    edge: int, radius: int, start: Point, end: Point, *, ccw: bool
) -> ArcCarrier:
    """Source edge ``edge`` of the strip: an exact arc about the strip's centre, travelled as the outline runs."""
    cx, cy = (lattice(v) for v in ARC_CENTRE_UM)
    return ArcCarrier(
        f"strip/edge{edge}",
        (Fraction(cx), Fraction(cy)),
        Fraction(radius),
        ccw,
        (Fraction(start[0]), Fraction(start[1])),
        (Fraction(end[0]), Fraction(end[1])),
        Lineage("polygon_edge", "strip", 0, 0, edge),
    )


# --------------------------------------------------------------------------- the die outline beside a cut

NOTCH_DIE = (0.0, -30.0, 40.0, 30.0)
# The die's outline with an 8 x 4 um notch in its top edge; the strip is cut on the notch's floor, y = 26 um.
NOTCH_OUTLINE = [(0, -30), (40, -30), (40, 30), (22, 30), (22, 26), (14, 26), (14, 30), (0, 30)]  # fmt: skip
NOTCH_STRIP = [(8.4, -14.0), (11.4, -14.0), (21.4, 26.0), (18.4, 26.0)]


def strip_into_notch() -> Scene:
    """A strip 3 um wide in x cut on the floor of the notch; its right side, continued, ends 0.1 um from the wall."""
    notched = Scene(
        "strip_into_notch",
        (),
        NOTCH_DIE,
        [Conductor("strip", "strip", to_lattice(NOTCH_STRIP))],
    )
    notched.prisms = [
        PrismRecord(
            "substrate",
            Footprint.from_rings(to_lattice(NOTCH_OUTLINE)),
            exact_z(-100),
            exact_z(0),
            "substrate",
            "dielectric",
        ),
        PrismRecord(
            "vacuum",
            die_footprint(notched),
            exact_z(0),
            exact_z(100),
            "vacuum",
            "vacuum",
        ),
    ]
    return notched


# --------------------------------------------------------------------------- a lead about twice a margin wide

# An isolated lead along x on a 40 x 60 um die, banded at the kit's margins (0.5, 1.0) um on a 1 nm grid.
LEAD_DIE = (0.0, -30.0, 40.0, 30.0)
LEAD_X_UM = (10.0, 30.0)


def straight_lead(width_um: float) -> Scene:
    """A lead along x from x = 10 to x = 30 um, ``width_um`` wide, on its own on the die."""
    half = width_um / 2
    x0, x1 = LEAD_X_UM
    ring = to_lattice([(x0, -half), (x1, -half), (x1, half), (x0, half)])
    return Scene("straight_lead", (), LEAD_DIE, [Conductor("lead", "lead", ring)])
