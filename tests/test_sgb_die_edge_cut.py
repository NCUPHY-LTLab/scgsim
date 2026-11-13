"""A conductor cut by the die edge continues past it: its bands run straight to the die edge (Issue #138).

The die edge is a dicing line.  A strip that runs onto it obliquely is measured as the strip it was before the
dicing cut, so each band offset runs straight to the die edge and no half circle is drawn about the strip's
corner there.  The scene is a strip 10 um tall in y that crosses a 20 x 60 um die from x = 0 to x = 20 with
slope 2: its sides are 44.72 um long inside the die and meet the die edge at 26.57 degrees.  Every band of a
strip that continues is a parallelogram along a side, ``margin`` wide and as long as the side, on the strip
(MA, MS) and on the exposed substrate beside it (SA), so the areas are calculated by hand.  A half circle about
each corner, the strip ending there, would leave 0.94 um^2 more in the last band of each family, fifteen times
that band's snap bound.

The same holds with vacuum beyond the substrate's edge, as the PDK's simulations pad the model.

Counterexamples: an end that the die edge does not cut keeps its half circle.
* Where the strip runs into a pad of its own Net inside the die, the contact is no distance origin and no dicing
  line, and the bands of the strip round its corner there.
* Where the strip runs onto the rim of a vacuum bore that the die encloses, the cell across holds no substrate
  and no metal, like the space beyond the die, but it does not reach the outside of the model: the rim is no
  dicing line, and the strip's bands round its corner there too.  Read as a die edge, the rim had the strip
  continued through the bore, and the SA curve beside it no longer closed.

A strip that runs into a corner of the die continues past the corner like any other cut.  Where a continuation
would put bands on the die from a conductor that is not drawn, the build is refused with the conductor and the
cut point: a die outline that comes back within the band margin beside it (a notch, or the outline of a conductor
that overhangs the die from a vertex on its edge), and two continuations that cross past the die edge.

Only a straight edge is continued.  A conductor whose gap edge runs onto the die edge as an exact arc is refused,
with the conductor and the cut point; the same arc ending on a pad of its own Net keeps its half circle.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from sgb_supported_kit import (
    MARGINS,
    Conductor,
    Expectation,
    Scene,
    assert_exact_bands,
    catalogue,
    conductor_shape,
    die_footprint,
    die_shape,
    planar_expectations,
    rect,
    reference,
    shape,
    to_lattice,
    type_bands,
)
from sgb_unsupported_cells import arc_cut_by_the_die_edge, strip_into_notch

from scgsim.geometry.sgb.bands.requests import InsetRequest, InterfaceSurfaces
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import (
    Footprint,
    PrismCatalogue,
    PrismRecord,
)

DIE = (0.0, -30.0, 20.0, 30.0)
STRIP = [(0.0, -25.0), (20.0, 15.0), (20.0, 25.0), (0.0, -15.0)]
SIDE_UM = math.hypot(20.0, 40.0)  # each side of the strip inside the die
STRIP_UM2 = 10.0 * 20.0
EXPOSED_UM2 = 20.0 * 60.0 - STRIP_UM2
STRIP_FAMILIES = ("ma::strip::top", "ms::strip::bottom")
SA_FAMILY = "sa::substrate|vacuum::z=0.0"


def strip_scene(vacuum_beyond_the_die: bool) -> Scene:
    """The strip on the die; with ``vacuum_beyond_the_die`` vacuum surrounds the substrate, 10 um past its edge."""
    scene = Scene(
        "strip_cut", (), DIE, [Conductor("strip", "strip", to_lattice(STRIP))]
    )
    if vacuum_beyond_the_die:
        x0, y0, x1, y1 = DIE
        padded = to_lattice(
            [
                (x0 - 10, y0 - 10),
                (x1 + 10, y0 - 10),
                (x1 + 10, y1 + 10),
                (x0 - 10, y1 + 10),
            ]
        )
        die = die_footprint(scene)
        beside = Footprint.from_rings(
            padded, [to_lattice([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])]
        )
        scene.prisms = [
            PrismRecord(
                "substrate", die, exact_z(-100), exact_z(0), "substrate", "dielectric"
            ),
            PrismRecord(
                "vacuum",
                Footprint.from_rings(padded),
                exact_z(0),
                exact_z(100),
                "vacuum",
                "vacuum",
            ),
            PrismRecord(
                "vacuum_beside", beside, exact_z(-100), exact_z(0), "vacuum", "vacuum"
            ),
        ]
    return scene


def continued_strip_bands(total_um2: float) -> tuple[float, ...]:
    """The bands of a support beside both sides of the strip, the strip continuing past the die edge."""
    near = [2 * m * SIDE_UM for m in MARGINS]
    return (near[0], near[1] - near[0], total_um2 - near[1])


def cap_arcs(model: ConformalModel, family_id: str) -> list:
    """The band arcs of a family that are pieces of a half circle about the end of a run of origins."""
    (family,) = [
        f for f in model.labelled.catalogue.families if f.family_id == family_id
    ]
    return [
        arc
        for arc in family.arcs
        if arc.lineage.band_member is not None
        and arc.lineage.band_member.role in ("capstart", "capend")
    ]


@pytest.mark.parametrize(
    "vacuum_beyond_the_die", [False, True], ids=["model_edge", "vacuum_beyond"]
)
def test_a_strip_cut_obliquely_by_the_die_edge_has_the_bands_of_a_strip_that_continues(
    vacuum_beyond_the_die,
):
    """The reference is the hand calculation, and the build meets it within the snap bound, with no cap arc."""
    scene = strip_scene(vacuum_beyond_the_die)
    expected = reference(scene)
    for family in STRIP_FAMILIES:
        assert expected[family].areas == pytest.approx(
            continued_strip_bands(STRIP_UM2), abs=1e-6
        )
    assert expected[SA_FAMILY].areas == pytest.approx(
        continued_strip_bands(EXPOSED_UM2), abs=1e-6
    )
    model = assert_exact_bands(scene)
    for family in (*STRIP_FAMILIES, SA_FAMILY):
        assert cap_arcs(model, family) == [], family


def contact_expectations(scene: Scene) -> dict[str, Expectation]:
    """The planar reference, with each conductor measured from its own outline less the contact and the die edge.

    Across the contact one conductor is no origin of the other, so neither conductor's gap edges are the
    boundary of one region here.
    """
    out = planar_expectations(scene)
    strip, pad = (conductor_shape(c) for c in scene.conductors)
    for name, own, other in (("strip", strip, pad), ("pad", pad, strip)):
        gap = own.boundary.difference(other.boundary)
        out[f"ma::{name}::top"] = (own, gap)
        out[f"ms::{name}::bottom"] = (own, gap)
    return out


def test_an_end_the_die_edge_does_not_cut_keeps_its_half_circle():
    """The strip runs from the die edge into a pad of its own Net: only its corners at the pad have cap arcs."""
    die = (0.0, -30.0, 40.0, 30.0)
    pad = [(20.0, -20.0), (30.0, -20.0), (30.0, 28.0), (20.0, 28.0)]
    scene = Scene(
        "strip_into_pad",
        (),
        die,
        [
            Conductor("strip", "N", to_lattice(STRIP)),
            Conductor("pad", "N", to_lattice(pad)),
        ],
        expectations=contact_expectations,
    )
    model = assert_exact_bands(scene)
    for family in STRIP_FAMILIES:
        # The strip's obtuse corner on the pad, (20, 15) um: the half circle about it bounds both bands.  At the
        # acute corner, (20, 25) um, each offset meets the contact before its end, so no piece of that half
        # circle lies in the strip.
        centres = {
            (float(arc.centre[0]), float(arc.centre[1]))
            for arc in cap_arcs(model, family)
        }
        assert centres == {(20_000.0, 15_000.0)}, family


# A rectangular bore through the substrate, filled with vacuum, whose left side the strip's end lies on.
BORE = [(20.0, -20.0), (30.0, -20.0), (30.0, 28.0), (20.0, 28.0)]


def bore_scene() -> Scene:
    """The strip from the die edge onto the rim of a vacuum bore inside a 40 x 60 um die."""
    scene = Scene(
        "strip_onto_bore",
        (),
        (0.0, -30.0, 40.0, 30.0),
        [Conductor("strip", "strip", to_lattice(STRIP))],
        expectations=bore_expectations,
    )
    die = die_footprint(scene)
    bore = to_lattice(BORE)
    scene.prisms = [
        PrismRecord(
            "substrate",
            Footprint.from_rings(die.rings()[0], [bore[::-1]]),
            exact_z(-100),
            exact_z(0),
            "substrate",
            "dielectric",
        ),
        PrismRecord(
            "bore",
            Footprint.from_rings(bore),
            exact_z(-100),
            exact_z(0),
            "vacuum",
            "vacuum",
        ),
        PrismRecord("vacuum", die, exact_z(0), exact_z(100), "vacuum", "vacuum"),
    ]
    return scene


def bore_expectations(scene: Scene) -> dict[str, Expectation]:
    """The strip measured from its two sides only: its ends lie on the die edge and on the bore's rim.

    The exposed substrate is the die less the bore and the strip; the bore's rim is no origin of it either.
    """
    die, bore = die_shape(scene), shape(to_lattice(BORE))
    strip = conductor_shape(scene.conductors[0])
    sides = strip.boundary.difference(die.boundary.union(bore.boundary))
    return {
        "ma::strip::top": (strip, sides),
        "ms::strip::bottom": (strip, sides),
        SA_FAMILY: (die.difference(bore).difference(strip), sides),
    }


def test_an_end_on_the_rim_of_an_enclosed_vacuum_bore_keeps_its_half_circle():
    """The bore holds no substrate and no metal, but the die encloses it: only the die-edge end continues."""
    model = assert_exact_bands(bore_scene())
    for family in STRIP_FAMILIES:
        centres = {
            (float(arc.centre[0]), float(arc.centre[1]))
            for arc in cap_arcs(model, family)
        }
        assert centres == {(20_000.0, 15_000.0)}, family
    sa_centres = {float(arc.centre[0]) for arc in cap_arcs(model, SA_FAMILY)}
    assert sa_centres == {20_000.0}  # on the bore's rim; none on the die edge, x = 0


def test_a_strip_cut_at_a_corner_of_the_die_continues_past_it():
    """One side of the strip ends in the die's top-right corner, the other on its right edge, both at 45 degrees."""
    corner = Scene(
        "strip_into_corner",
        (),
        (0.0, -30.0, 40.0, 30.0),
        [
            Conductor(
                "strip",
                "strip",
                to_lattice([(10.0, 0.0), (16.0, 0.0), (40.0, 24.0), (40.0, 30.0)]),
            )
        ],
    )
    model = assert_exact_bands(corner)
    for family in (*STRIP_FAMILIES, SA_FAMILY):
        # the only caps are those of the strip's end inside the die, at y = 0
        assert {float(arc.centre[1]) for arc in cap_arcs(model, family)} <= {0.0}


def refusal(scene: Scene) -> str:
    """The message the build of ``scene`` is refused with."""
    return refused_build(catalogue(scene), type_bands())


def refused_build(cat: PrismCatalogue, insets: tuple[InsetRequest, ...]) -> str:
    """The message the Route B build of ``cat`` with ``insets`` is refused with."""
    with pytest.raises(InputError) as raised:
        build_model_from_catalogue(cat, "B", insets=insets)
    return str(raised.value)


def test_a_die_outline_that_comes_back_beside_a_continuation_is_refused():
    """A strip cut on the floor of a notch in the die's top edge: its right side's continuation ends 0.1 um from the notch wall."""
    message = refusal(strip_into_notch())
    assert "the die outline comes back within the band margin (1 um)" in message
    assert "of 'strip', cut by the die edge at (21.4000, 26.0000) um" in message
    assert "SCGSim does not model it" in message


def test_a_conductor_overhanging_the_die_from_a_vertex_on_its_edge_is_refused():
    """A pad leaves the substrate at a vertex of its outline, (0, 4) um, and hangs 3 um past the die edge.

    Its top edge is a gap edge up to that vertex, where the die edge cuts it; past the die edge the pad's own
    outline turns down beside the continuation, so the outline comes back at the cut.
    """
    overhang = Scene(
        "pad_overhang",
        (),
        (0.0, -30.0, 40.0, 30.0),
        [
            Conductor(
                "pad",
                "pad",
                to_lattice([(10, -4), (10, 4), (0, 4), (-3, 1), (-3, -4)]),
            )
        ],
    )
    padded = to_lattice([(-10, -30), (40, -30), (40, 30), (-10, 30)])
    overhang.prisms = [
        PrismRecord(
            "substrate",
            die_footprint(overhang),
            exact_z(-100),
            exact_z(0),
            "substrate",
            "dielectric",
        ),
        PrismRecord(
            "vacuum",
            Footprint.from_rings(padded),
            exact_z(0),
            exact_z(100),
            "vacuum",
            "vacuum",
        ),
    ]
    message = refusal(overhang)
    assert "the die outline comes back within the band margin (1 um)" in message
    assert "of 'pad', cut by the die edge at (0.0000, 4.0000) um" in message


def test_two_continuations_that_cross_past_the_die_edge_are_refused():
    """Two strips converge on the top edge with 0.2 um of substrate between them: their facing sides cross 0.3 um past it."""
    converging = Scene(
        "strips_converging",
        (),
        (0.0, -30.0, 40.0, 30.0),
        [
            Conductor(
                "strip_left",
                "left",
                to_lattice([(7.0, 0.0), (10.0, 0.0), (19.9, 30.0), (16.9, 30.0)]),
            ),
            Conductor(
                "strip_right",
                "right",
                to_lattice([(30.0, 0.0), (33.0, 0.0), (23.1, 30.0), (20.1, 30.0)]),
            ),
        ],
    )
    message = refusal(converging)
    # The two cuts are named in the order of the SA support's ring.
    assert "the continuations of 'strip_right' and 'strip_left'" in message
    assert (
        "cut by the die edge at (20.1000, 30.0000) and (19.9000, 30.0000) um" in message
    )
    assert "cross past the die edge" in message


@pytest.mark.parametrize(
    ("band_type", "cut_um"),
    [
        ("MA", "(0.0000, 2.5000)"),
        ("MS", "(0.0000, 2.5000)"),
        ("SA", "(0.0000, 8.0000)"),
    ],
)
def test_a_conductor_cut_by_the_die_edge_along_an_arc_is_refused(band_type, cut_um):
    """A curved strip whose long edges are exact arcs runs into the die's left edge: no band type builds it.

    A straight gap edge cut by the die edge is continued along its line; an arc cannot be, and SCGSim does not
    model it.  The strip's own faces name the strip at the first cut on their ring, the inner arc's; the exposed
    substrate, an interface, names the conductor beside its first cut, the outer arc's.
    """
    request = InsetRequest(band_type.lower(), InterfaceSurfaces(band_type), MARGINS)
    message = refused_build(arc_cut_by_the_die_edge(), (request,))
    assert (
        f"the die edge cuts 'strip' at {cut_um} um, where its gap edge is an exact arc"
        in message
    )
    assert "SCGSim does not model it" in message
    assert "Let a straight edge of the conductor meet the die edge" in message


def test_an_arc_that_ends_on_a_pad_of_its_own_net_keeps_its_half_circle():
    """The same curved strip ends on a pad of its own Net inside the die: the contact is no die edge, and it builds.

    The outer arc's end on the pad keeps the half circle of the strip's bands about it, as on a straight edge.
    """
    pad = Scene(
        "arc_onto_pad",
        (),
        (-6.0, -4.0, 20.0, 14.0),
        [Conductor("pad", "strip", to_lattice(rect(-4.0, 0.0, 0.0, 12.0)))],
    )
    strip = arc_cut_by_the_die_edge().prisms[-1]
    assert strip.entity_id == "strip" and strip.arcs
    with_pad = catalogue(pad)
    model = build_model_from_catalogue(
        replace(with_pad, prisms=(*with_pad.prisms, strip)), "B", insets=type_bands()
    )
    for family in STRIP_FAMILIES:
        centres = {
            (float(arc.centre[0]), float(arc.centre[1]))
            for arc in cap_arcs(model, family)
        }
        assert (0.0, 8_000.0) in centres, family
