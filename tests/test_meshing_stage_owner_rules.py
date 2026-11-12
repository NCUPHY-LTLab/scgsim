"""The two owner rules the sidecar v3 made decidable: a support ring's size (#147) and a short source edge (#138 (c)).

The meshing stage attributes a fine band edge or face by lineage, read from the sidecar alone (#170, slices D and
E).  Both scenes are real SGB builds of Route B with MA, MS and SA bands at (0.5, 1.0) um, checked as ``mesh()``
checks them:

* a bar 2.004 um wide cut by the die edge, so its 1 um bands leave a core 4 nm wide.  Its support ring has four
  primitives; where the die-edge primitive is the ring's last, the two long sides are the ring's first and third,
  and only the ring's size shows they are not neighbours (#147);
* ``junction_fan``, whose SA band offsets a 0.1954 um chord of the ground's round cap by 1 um and leaves an edge
  15.6 grid steps long (#138 (c)).
"""

from __future__ import annotations

import copy

import pytest
import sgb_supported_cells as disk_qubit
from sgb_supported_kit import Conductor, Scene, catalogue, to_lattice, type_bands

from scgsim.contracts.sgb_sidecar import Sidecar, read_sidecar
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.meshing import check_fine_structure

DIE = (-20.0, -20.0, 40.0, 40.0)
BAR_WIDTH_UM = 2.004  # twice the 1 um margin and 4 grid steps
# The bar, 40 um long from y = 10 um, runs into the die edge at one end and ends inside the die at the other.
BARS = {
    "left": [
        (-20.0, 10.0),
        (20.0, 10.0),
        (20.0, 10.0 + BAR_WIDTH_UM),
        (-20.0, 10.0 + BAR_WIDTH_UM),
    ],
    "right": [
        (0.0, 10.0),
        (40.0, 10.0),
        (40.0, 10.0 + BAR_WIDTH_UM),
        (0.0, 10.0 + BAR_WIDTH_UM),
    ],
}
# Where the die edge sits on the ring of the bar's top face, as SGB orders it: last on the left, second on the right.
DIE_EDGE_ON_RING = {
    "left": [False, False, False, True],
    "right": [False, True, False, False],
}
# Where the core ends inside the die: 1 um (the margin) before the bar's free end.
CORE_END_X_UM = {"left": 19.0, "right": 1.0}


def banded_sidecar(scene: Scene, tmp_path) -> Sidecar:
    """The sidecar of a scene built as the sample set builds it: Route B, MA, MS and SA bands by type."""
    model = build_model_from_catalogue(catalogue(scene), "B", insets=type_bands())
    return read_sidecar(emit_model(model, tmp_path).sidecar)


def bar_cut_by_the_die_edge(side: str) -> Scene:
    """A bar 2.004 um wide from the die edge on ``side`` 40 um into the die, in a Net of its own."""
    return Scene(
        f"bar_cut_{side}", (), DIE, [Conductor("bar", "N", to_lattice(BARS[side]))]
    )


def top_ring(sidecar: Sidecar, entity: str) -> list[str]:
    """The primitives, in ring order, of the one support ring of an Entity's top-face (MA) band family."""
    (family,) = [
        fid
        for fid, record in sidecar["band_family"].items()
        if record["support"] == {"entity": entity, "side": "top"}
    ]
    (ring,) = [r for r in sidecar["support_ring"].values() if r["family"] == family]
    return ring["primitives"]


@pytest.mark.parametrize("side", sorted(BARS))
def test_a_core_where_the_support_ring_wraps_round_is_the_erosion_on_either_side(
    side, tmp_path
):
    """The bar's 4 nm core is refused as the erosion whichever end the die cuts: #147's ring size, from the sidecar.

    The core lies between the offsets of the bar's two long sides.  On the ring of four primitives, ``N = 4`` as
    ``support_ring`` lists them, those sides are places 0 and 2: ``(0 - 2) mod 4 = 2``, so they share no vertex and
    are not neighbours.  Cut on the left, the die edge closes the ring as its last primitive and no band curve
    follows it, so place 2 is the last one a band member names.

    Before (3cf8596): the left bar's two class B cores and two class C core ends were owned by ``sgb``, with the fix
    "an SGB defect ... the layout needs no change", because without the ring's size places 0 and 2 might have
    been neighbours across the ring's start; its mirror image, cut on the right, was owned by ``erosion``.  After:
    both are the erosion, with the fix to change the margins or the width.
    """
    sidecar = banded_sidecar(bar_cut_by_the_die_edge(side), tmp_path)
    ring = top_ring(sidecar, "bar")
    assert [sidecar["support_primitive"][p]["die_edge"] for p in ring] == (
        DIE_EDGE_ON_RING[side]
    )
    report = check_fine_structure(sidecar)
    cores = [f for f in report.refusals if f.measure == "face width 2A/P"]
    assert sorted(g for f in cores for g in f.groups) == [
        "MA::bar__top::band=2",
        "MS::bar__bottom::band=2",
    ]
    ends = [f for f in report.warnings if f.measure == "edge length"]
    assert len(ends) == 2  # noqa: PLR2004 -- the core's end in the MA and in the MS band
    for end in ends:
        assert end.location_um[0] == pytest.approx(CORE_END_X_UM[side], abs=1e-6)
    assert {(f.rule, f.owner) for f in report.findings} == {
        ("B", "erosion"),
        ("C", "erosion"),
    }
    for finding in report.findings:
        assert "change the margins in surface_epr_specs" in finding.fix


# --------------------------------------------------------------------------- a source edge shorter than its margin

SHORT_EDGE = "e:9509e27bdf7e2d28"
CHORD = "ground/polygon0/ring1/edge29"


@pytest.fixture(scope="module")
def junction_fan_sidecar(tmp_path_factory) -> Sidecar:
    """The sidecar of ``junction_fan``, built once."""
    return banded_sidecar(
        disk_qubit.SCENES["junction_fan"](), tmp_path_factory.mktemp("junction_fan")
    )


def test_a_short_band_edge_on_a_source_edge_shorter_than_its_margin_is_the_erosion(
    junction_fan_sidecar,
):
    """The 15.6-step SA edge is the erosion of a 0.1954 um chord, and its fix names that chord: #138 (c).

    Its carrier offsets support primitive 13 of its ring by the 1 um margin, and the primitive lies on the ground's
    source edge ``ground/polygon0/ring1/edge29``: 0.1954 um, shorter than the margin.  The place and length in the
    message are the ``source_curve`` record's.  The class and action do not change: a class C warning.

    Before (3cf8596): owner ``sgb``, fix "an SGB defect ... the layout needs no change".  After: owner ``erosion``,
    and the fix names the chord, where it runs and how long it is, and says to declare the arc exactly or to draw
    longer chords.
    """
    sidecar = junction_fan_sidecar
    (finding,) = [
        f for f in check_fine_structure(sidecar).findings if f.entity == SHORT_EDGE
    ]
    assert (finding.rule, finding.action, finding.owner) == ("C", "warn", "erosion")
    assert finding.value_steps == pytest.approx(15.62, abs=0.005)
    chord = sidecar["source_curve"][CHORD]
    assert chord["length_um"] < 1.0
    (x0, y0), (x1, y1) = chord["curve"]["start_um"], chord["curve"]["end_um"]
    assert finding.fix == (
        f"the source edge {CHORD}, {chord['length_um']:.6g} um long from ({x0:.4f}, {y0:.4f}) to "
        f"({x1:.4f}, {y1:.4f}) um, is shorter than the 1 um band margin it is offset by, so the exact "
        "erosion leaves an edge this short. Declare the arc that edge is a chord of exactly, or draw it "
        "with longer chords."
    )
    assert (
        "0.195389 um long from (164.2780, -171.3220) to (164.4020, -171.4730) um"
        in (finding.describe())
    )


def test_where_the_offsets_of_far_edges_meet_the_fix_stays_the_margin_or_the_width(
    junction_fan_sidecar,
):
    """At the lead's neck, exactly twice the margin wide, the four 1 nm band edges keep the erosion's width fix.

    Their band curves offset chords of the lead's round tip, 0.196 and 0.390 um long, so a source edge shorter than
    the margin lies under them too.  But they are where the offsets of chords from the two sides of the tip meet
    (places 13 and 20 of a 43-primitive ring, not neighbours), which the margin and the neck width explain whatever
    the chords' lengths (#138 (b)): their fix is unchanged from 3cf8596.
    """
    neck = [
        f
        for f in check_fine_structure(junction_fan_sidecar).findings
        if f.entity != SHORT_EDGE
    ]
    assert len(neck) == 4  # noqa: PLR2004 -- the MA and MS band edges on each side of the neck
    for finding in neck:
        assert (finding.rule, finding.owner, finding.value_steps) == (
            "C",
            "erosion",
            1.0,
        )
        assert finding.fix.startswith(
            "the band margin meets the conductor's local width"
        )


def test_the_owner_follows_the_source_curve_length_the_sidecar_records(
    junction_fan_sidecar,
):
    """The rule reads the chord's length from ``source_curve``: recorded as long as the margin, the edge is SGB's."""
    sidecar = copy.deepcopy(junction_fan_sidecar)
    sidecar["source_curve"][CHORD]["length_um"] = 1.0
    (finding,) = [
        f for f in check_fine_structure(sidecar).findings if f.entity == SHORT_EDGE
    ]
    assert finding.owner == "sgb"
    assert CHORD not in finding.fix
