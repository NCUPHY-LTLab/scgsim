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

Counterexample: an end that the die edge does not cut keeps its half circle.  Where the strip runs into a pad of
its own Net inside the die, the contact is no distance origin and no dicing line, and the bands of the strip
round its corner there.
"""

from __future__ import annotations

import math

import pytest
from sgb_supported_kit import (
    MARGINS,
    Conductor,
    Expectation,
    Scene,
    assert_exact_bands,
    conductor_shape,
    die_footprint,
    planar_expectations,
    reference,
    to_lattice,
)

from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismRecord

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
    return [arc for arc in family.arcs if "/cap" in arc.id]


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
