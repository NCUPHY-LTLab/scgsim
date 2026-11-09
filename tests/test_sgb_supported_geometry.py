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
    TURN_REFUSAL,
    assert_exact_bands,
    catalogue,
    turns_degrees,
    type_bands,
)

from scgsim.geometry.sgb.errors import ValidationError
from scgsim.geometry.sgb.model import build_model_from_catalogue

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


@pytest.mark.parametrize("name", scene_params())
def test_a_supported_cell_has_the_bands_of_its_exact_erosion(name):
    """The scene builds, with every band family it should have and no other, at the exact areas."""
    assert_exact_bands(SCENES[name]())


@pytest.mark.parametrize("name", sorted(pdk.DRAWN_SPLINES))
def test_a_spline_strip_drawn_at_258_points_is_refused_by_its_turn_per_vertex(name):
    """The PDK's spline strips turn 0.012 and 0.058 degrees at their flattest vertices: the bands are refused."""
    scene = pdk.DRAWN_SPLINES[name]()
    assert min(abs(t) for t in turns_degrees(scene.conductors[0].outer)) < 0.06
    with pytest.raises(ValidationError, match=TURN_REFUSAL):
        build_model_from_catalogue(catalogue(scene), "B", insets=type_bands())


@pytest.mark.parametrize(
    "name", ["cell21_interpolation_resampled", "cell21_bspline_resampled"]
)
def test_a_resampled_spline_strip_turns_at_least_the_rule_at_every_vertex(name):
    """The resampled strips that build above keep every lattice vertex at 2.8125 degrees or more."""
    ring = SCENES[name]().conductors[0].outer
    assert min(abs(t) for t in turns_degrees(ring)) >= 360 / 128
