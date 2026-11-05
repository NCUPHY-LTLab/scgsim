"""The a priori sizing rules of rules/1, on the cases the benchmark measured."""

from __future__ import annotations

import dataclasses
import math

import pytest
from meshing_support import feature, features_of, film

from scgsim.meshing import RULES_VERSION, RuleSet
from scgsim.meshing.policy import BoundingBox, coarse_size, ladder_size, size_requests


def box_with_diagonal(diagonal: float) -> BoundingBox:
    return BoundingBox((0.0, 0.0, 0.0), (diagonal, 0.0, 0.0))


def test_the_default_rule_set_is_the_measured_one():
    rules = RuleSet()
    assert rules.version == RULES_VERSION == "scgsim.meshing.rules/1"
    assert (rules.max_size_divisor, rules.gradation) == (20.0, 1.0)
    assert rules.film_sidewall_thickness_multiple == 25.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.gradation = 2.0  # type: ignore[misc]


def test_the_coarse_size_is_the_diagonal_over_twenty():
    box = BoundingBox((0.0, 0.0, 0.0), (3.0, 4.0, 12.0))  # diagonal 13
    assert coarse_size(box, RuleSet()) == pytest.approx(13 / 20)


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        (204.2548411472296, 2.553185514340370),  # the 100 um chip: 25 x 0.2 = 5 sits just under 10.21 / 2
        (2042.1067654067651, 3.190791),  # the 1 mm chip: ladder step 5 of 102.1
        (4084.164546469703, 3.190754),  # the 2 mm chip
    ],
)
def test_a_point_two_micron_film_gets_the_ladder_size_under_five_microns(diagonal, expected):
    requests = size_requests(features_of(film([1, 2, 3])), box_with_diagonal(diagonal), RuleSet())
    assert set(requests.curve_sizes) == {1, 2, 3}
    for size in requests.curve_sizes.values():
        assert size == pytest.approx(expected, rel=1e-5)
    assert requests.max_size == pytest.approx(diagonal / 20)
    assert requests.gradation == 1.0


def test_a_thick_film_is_no_coarser_than_half_the_coarse_size():
    requests = size_requests(
        features_of(film([1], thickness_um=10.0)), box_with_diagonal(400.0), RuleSet()
    )
    assert requests.curve_sizes[1] == pytest.approx(10.0)  # 400 / 20 / 2, not 25 x 10


def test_the_size_is_never_above_the_thickness_multiple_and_always_on_the_ladder():
    max_size = 100.0
    for thickness in (0.05, 0.2, 0.7, 1.0, 3.3):
        size = size_requests(
            features_of(film([1], thickness_um=thickness)),
            box_with_diagonal(max_size * 20),
            RuleSet(),
        ).curve_sizes[1]
        assert size <= 25 * thickness
        assert math.log2(max_size / size) == pytest.approx(round(math.log2(max_size / size)))


def test_a_curve_covered_by_two_films_takes_the_finer_size():
    requests = size_requests(
        features_of(film([1, 2], 0.2), film([2, 3], 0.05)), box_with_diagonal(2000.0), RuleSet()
    )
    assert requests.curve_sizes[2] == requests.curve_sizes[3] < requests.curve_sizes[1]


def test_kinds_without_an_a_priori_rule_ask_for_no_size():
    kinds = features_of(
        feature("band_boundary", [1], {"margins_um": [0.5, 1.0], "narrowest_band_um": 0.5}),
        feature("arc", [2], {"radius_um": 10.0}, source=None),
        feature("refinement_curve", [3]),
        feature("port_edge", [4]),
    )
    assert size_requests(kinds, box_with_diagonal(2000.0), RuleSet()).curve_sizes == {}


def test_a_film_curve_that_is_also_a_band_edge_keeps_the_film_size():
    requests = size_requests(
        features_of(
            film([1]), feature("band_boundary", [1], {"margins_um": [0.5], "narrowest_band_um": 0.5})
        ),
        box_with_diagonal(2000.0),
        RuleSet(),
    )
    assert set(requests.curve_sizes) == {1}


def test_ladder_size_is_the_largest_rung_not_above_the_target():
    assert ladder_size(60.0, 100.0) == 50.0
    assert ladder_size(50.0, 100.0) == 50.0
    assert ladder_size(49.9, 100.0) == 25.0
    assert ladder_size(1000.0, 100.0) == 50.0  # the first rung is max / 2
    with pytest.raises(ValueError, match="positive target"):
        ladder_size(0.0, 100.0)
