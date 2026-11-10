"""The Supported Geometry sample set: every unit cell SCGSim supports builds with the bands of its exact erosion.

Each scene of ``sgb_supported_cells`` (cells 1-12, the Disk Qubit) and ``sgb_supported_pdk_cells`` (cells 13-24,
the other PDK cells) is built with ``build_model_from_catalogue`` under Route B, with MA, MS and SA bands at
(0.5, 1.0) um asked for by type.  The build must produce exactly the band families of the reference, and every
band area must lie within the snap bound of the exact erosion of the same lattice polygons.  The areas are
geometry, not simulation results.  Scenes that take more than a few seconds to build are marked ``slow``.
"""

from __future__ import annotations

import pytest
import sgb_supported_cells as disk_qubit
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import (
    MARGINS,
    Conductor,
    Scene,
    assert_exact_bands,
    die_box,
    ngon,
    rect,
    reference,
    short_band_edges,
    to_lattice,
    turns_degrees,
)

from scgsim.geometry.sgb.cutfamilies.inset import SHORT_ARC_GRID_STEPS

SCENES = {**disk_qubit.SCENES, **pdk.SCENES}
SLOW = {
    "cell02_clearance",
    "disk_in_clearance",
    "cell07_cpw45_ro",
    "cell07_cpw45_xy",
    "ro_through_clearance",
    "xy_tip",
    "d0_die",
    "claw_floating",
    "claw_floating_test_build",
    "claw_grounded",
    "junction_straight",
    "junction_fan",
    "junction_fan_noport",
    "d1_die_test_build",
    "d1_die_defaults",
    "cell11_bumps_flip_chip",
    "cell19_launcher",
    "cell20_comb",
    "meander_long_outline",
}


def scene_params():
    """Every sample scene, the slow ones marked."""
    return [
        pytest.param(name, id=name, marks=pytest.mark.slow if name in SLOW else ())
        for name in SCENES
    ]


def test_the_sample_set_covers_every_supported_unit_cell():
    """Cells 1 to 24 each have at least one sample scene."""
    covered = {cell for build in SCENES.values() for cell in build().cells}
    assert covered == set(range(1, 25))


# The emitted band edges shorter than SHORT_ARC_GRID_STEPS that remain, per scene (Issue #138); every other
# scene has none.  They are exact geometry:
# * the lead of the Disk Qubit junction, 2 um wide, at the 1 um margin: the eroded region closes where the lead
#   is 2 m wide, and the two offsets meet within a grid step (Issue #128);
# * an SA offset between a convex crossing and a reflex arc, 14 nm long, beside the ground tongue's 195 nm cap
#   chord, which is shorter than the margin (junction_fan, d1_die_defaults).
# A band that ends on the die edge leaves none: it runs straight to the die edge, with no cap (cells 18 and 21).
SHORT_BAND_EDGES = {
    "junction_straight": 8,
    "d1_die_test_build": 8,
    "junction_fan": 5,
    "junction_fan_noport": 5,
    "d1_die_defaults": 5,
}


@pytest.mark.parametrize("name", scene_params())
def test_a_supported_cell_has_the_bands_of_its_exact_erosion(name):
    """The scene builds, with every band family it should have and no other, at the exact areas.

    The band seams emit no edge shorter than 25 grid steps beyond the recorded exact ones.
    """
    model = assert_exact_bands(SCENES[name]())
    short = short_band_edges(model, SHORT_ARC_GRID_STEPS)
    assert len(short) == SHORT_BAND_EDGES.get(name, 0), short


@pytest.mark.parametrize("name", sorted(pdk.DRAWN_SPLINES))
def test_a_spline_strip_drawn_at_258_points_builds_exact_bands(name):
    """The PDK's spline strips turn 0.012 and 0.058 degrees at their flattest vertices.

    The offsets of two lines end at their crossing on both sides of a vertex (Issue #138), so the curve closes
    at turns this small and the bands are those of the exact erosion.
    """
    scene = pdk.DRAWN_SPLINES[name]()
    assert min(abs(t) for t in turns_degrees(scene.conductors[0].outer)) < 0.06
    assert_exact_bands(scene)


@pytest.mark.parametrize(
    "name", ["cell21_interpolation_resampled", "cell21_bspline_resampled"]
)
def test_a_resampled_spline_strip_turns_at_least_2_8125_degrees_at_every_vertex(name):
    """The resampled strips that build above keep every lattice vertex at 2.8125 degrees or more."""
    ring = SCENES[name]().conductors[0].outer
    assert min(abs(t) for t in turns_degrees(ring)) >= 360 / 128


def test_the_long_meander_has_outlines_of_500_to_900_vertices():
    """The stress case of 500 to 900 vertices in one outline, each vertex turning at least 2.8125 degrees.

    The scene builds with exact bands in the parametrised test above (``meander_long_outline``).
    """
    scene = SCENES["meander_long_outline"]()
    sizes = {c.entity: len(c.outer) for c in scene.conductors}
    assert 500 <= sizes["signal"] <= 900
    for c in scene.conductors:
        assert min(abs(t) for t in turns_degrees(c.outer)) >= 360 / 128, c.entity


def test_a_band_that_erodes_away_has_no_reference_area_and_builds_empty():
    """A 16-gon of radius 1 um: its inradius, 0.98 um, is below the 1 um margin, so the last band is empty.

    The reference must say so (a buffered ring leaves a 0.0012 um^2 hole here) and the build must agree.
    """
    scene = Scene(
        "collapse",
        (),
        die_box(21.0),
        [Conductor("disk", "disk", to_lattice(ngon(1.0, 16)))],
    )
    expected = reference(scene)
    for family in ("ma::disk::top", "ms::disk::bottom"):
        assert expected[family].areas[-1] == pytest.approx(0.0, abs=1e-12)
        assert expected[family].bounds[-1] == 0.0
    assert_exact_bands(scene)


@pytest.mark.parametrize(("side", "last_band"), [(2.4, 0.4**2), (1.8, 0.0)])
def test_the_reference_of_a_square_is_its_hand_calculated_erosion(side, last_band):
    """An axis-aligned square of side s: band k holds the points between margins, (s - 2 m_k)^2 at most."""
    half = side / 2
    scene = Scene(
        "square",
        (),
        die_box(10.0),
        [Conductor("pad", "pad", to_lattice(rect(-half, -half, half, half)))],
    )
    inner = [max(side - 2 * m, 0.0) ** 2 for m in MARGINS]
    expected = (side**2 - inner[0], inner[0] - inner[1], inner[1])
    assert reference(scene)["ma::pad::top"].areas == pytest.approx(expected, abs=1e-9)
    assert expected[-1] == pytest.approx(last_band, abs=1e-12)
