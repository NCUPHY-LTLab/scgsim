"""The three sizing recipes, on the cases the benchmark measured."""

from __future__ import annotations

import dataclasses
import math

import pytest
from meshing_support import feature, features_of, film

from scgsim.meshing import (
    FeatureRecipe,
    GlobalSizeRecipe,
    UniformBoundaryRecipe,
)
from scgsim.meshing.recipes.feature import conductor_outline_curves
from scgsim.meshing.sizing import BoundingBox, coarse_size, ladder_size


def box_with_diagonal(diagonal: float) -> BoundingBox:
    """A box whose diagonal is ``diagonal``."""
    return BoundingBox((0.0, 0.0, 0.0), (diagonal, 0.0, 0.0))


def test_the_default_feature_recipe_is_the_measured_one():
    """The default feature recipe is the measured one."""
    recipe = FeatureRecipe()
    assert recipe.name == "features"
    assert recipe.version == "scgsim.meshing.features/1"
    assert (recipe.max_size_divisor, recipe.gradation) == (20.0, 1.0)
    assert recipe.film_sidewall_thickness_multiple == 25.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        recipe.gradation = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("recipe", "name", "version"),
    [
        (FeatureRecipe(), "features", "scgsim.meshing.features/1"),
        (GlobalSizeRecipe(), "global-size", "scgsim.meshing.global-size/1"),
        (
            UniformBoundaryRecipe(),
            "uniform-boundary",
            "scgsim.meshing.uniform-boundary/1",
        ),
    ],
)
def test_every_recipe_has_its_own_name_and_version_that_are_not_parameters(
    recipe, name, version
):
    """Name and version identify the strategy; the dataclass fields are only its parameters."""
    assert (recipe.name, recipe.version) == (name, version)
    assert "name" not in dataclasses.asdict(recipe)
    assert "version" not in dataclasses.asdict(recipe)


def test_the_coarse_size_is_the_diagonal_over_twenty():
    """The coarse size is the diagonal over twenty."""
    box = BoundingBox((0.0, 0.0, 0.0), (3.0, 4.0, 12.0))  # diagonal 13
    assert coarse_size(box, FeatureRecipe().max_size_divisor) == pytest.approx(13 / 20)


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        (
            204.2548411472296,
            2.553185514340370,
        ),  # the 100 um chip: 25 x 0.2 = 5 sits just under 10.21 / 2
        (2042.1067654067651, 3.190791),  # the 1 mm chip: ladder step 5 of 102.1
        (4084.164546469703, 3.190754),  # the 2 mm chip
    ],
)
def test_a_point_two_micron_film_gets_the_ladder_size_under_five_microns(
    diagonal, expected
):
    """A point two micron film gets the ladder size under five microns."""
    requests = FeatureRecipe().size_requests(
        features_of(film([1, 2, 3])), box_with_diagonal(diagonal)
    )
    assert set(requests.curve_sizes) == {1, 2, 3}
    for size in requests.curve_sizes.values():
        assert size == pytest.approx(expected, rel=1e-5)
    assert requests.max_size == pytest.approx(diagonal / 20)
    assert requests.gradation == 1.0


def test_a_film_wall_whose_rule_would_not_bind_gets_no_size():
    """A wall at or above the ceiling is already resolved relative to the coarse size."""
    rules = FeatureRecipe()
    box = box_with_diagonal(400.0)  # coarse size 20, ceiling 10
    thick = rules.size_requests(features_of(film([1], thickness_um=10.0)), box)
    assert thick.curve_sizes == {}  # 25 x 10 = 250, far above the ceiling
    on_the_ceiling = rules.size_requests(features_of(film([1], thickness_um=0.4)), box)
    assert on_the_ceiling.curve_sizes == {}  # 25 x 0.4 = 10 = ceiling: not binding
    just_under = rules.size_requests(features_of(film([1], thickness_um=0.39)), box)
    assert just_under.curve_sizes == {
        1: pytest.approx(5.0)
    }  # 9.75 -> the rung below 10


def test_a_ten_micron_substrate_in_a_millimetre_model_is_left_alone():
    """The case that cost 10 % of the tetrahedra: 25 x 10 um is far above half the coarse size."""
    box = box_with_diagonal(2042.1)
    assert (
        FeatureRecipe()
        .size_requests(features_of(film([1], thickness_um=10.0)), box)
        .curve_sizes
        == {}
    )


def test_the_size_is_never_above_the_thickness_multiple_and_always_on_the_ladder():
    """Every thin wall that gets a size is on the max/2^k ladder and at most 25 times its thickness."""
    max_size = 100.0
    for thickness in (0.05, 0.2, 0.7, 1.0):
        size = (
            FeatureRecipe()
            .size_requests(
                features_of(film([1], thickness_um=thickness)),
                box_with_diagonal(max_size * 20),
            )
            .curve_sizes[1]
        )
        assert size <= 25 * thickness
        assert math.log2(max_size / size) == pytest.approx(
            round(math.log2(max_size / size))
        )


def test_a_curve_covered_by_two_films_takes_the_finer_size():
    """A curve covered by two films takes the finer size."""
    requests = FeatureRecipe().size_requests(
        features_of(film([1, 2], 0.2), film([2, 3], 0.05)),
        box_with_diagonal(2000.0),
    )
    assert requests.curve_sizes[2] == requests.curve_sizes[3] < requests.curve_sizes[1]


def test_kinds_without_an_a_priori_rule_ask_for_no_size():
    """Kinds without an a priori rule ask for no size."""
    kinds = features_of(
        feature(
            "band_boundary", [1], {"margins_um": [0.5, 1.0], "narrowest_band_um": 0.5}
        ),
        feature("arc", [2], {"radius_um": 10.0}, source=None),
        feature("refinement_curve", [3]),
        feature("port_edge", [4]),
    )
    assert (
        FeatureRecipe().size_requests(kinds, box_with_diagonal(2000.0)).curve_sizes
        == {}
    )


def test_a_film_curve_that_is_also_a_band_edge_keeps_the_film_size():
    """A film curve that is also a band edge keeps the film size."""
    requests = FeatureRecipe().size_requests(
        features_of(
            film([1]),
            feature(
                "band_boundary", [1], {"margins_um": [0.5], "narrowest_band_um": 0.5}
            ),
        ),
        box_with_diagonal(2000.0),
    )
    assert set(requests.curve_sizes) == {1}


def test_ladder_size_is_the_largest_rung_not_above_the_target():
    """Ladder size is the largest rung not above the target."""
    assert ladder_size(60.0, 100.0) == 50.0
    assert ladder_size(50.0, 100.0) == 50.0
    assert ladder_size(49.9, 100.0) == 25.0
    assert ladder_size(1000.0, 100.0) == 50.0  # the first rung is max / 2
    with pytest.raises(ValueError, match="positive target"):
        ladder_size(0.0, 100.0)


def outline_features():
    """Features with owners: pad film 1-2, pad arc 3, substrate film 4-5, band 6, unowned arc 7, port 8."""
    pad_film = film([1, 2])
    pad_film["source"] = "pad"
    substrate_film = film([4, 5], 100.0)
    substrate_film["source"] = "substrate"
    pad_arc = feature("arc", [3], {"radius_um": 10.0}, source="pad")
    shared_arc = feature("arc", [7], {"radius_um": 1.0}, source=None)
    band = feature(
        "band_boundary", [6], {"margins_um": [0.5], "narrowest_band_um": 0.5}
    )
    port = feature("port_edge", [8], source="port")
    return features_of(pad_film, pad_arc, substrate_film, band, shared_arc, port)


def test_the_conductor_outline_rule_is_off_by_default():
    """The conductor outline rule is off by default."""
    rules = FeatureRecipe()
    assert rules.conductor_outline is False
    assert (rules.conductor_edge_size_um, rules.conductor_edge_grow_um) == (5.0, 300.0)
    assert (
        rules.size_requests(outline_features(), box_with_diagonal(2000.0)).fields == ()
    )


def test_the_conductor_outline_is_film_walls_and_arcs_of_conductors_and_the_port_edges():
    """The conductor outline is film walls and arcs of conductors and the port edges."""
    assert conductor_outline_curves(outline_features()) == (1, 2, 3, 8)


def test_the_outline_field_has_the_edge_size_and_grows_over_its_own_distance():
    """The outline field has the edge size and grows over its own distance."""
    rules = FeatureRecipe(conductor_outline=True)
    requests = rules.size_requests(outline_features(), box_with_diagonal(2000.0))
    (outline,) = requests.fields
    assert (outline.curves, outline.size, outline.distance) == (
        (1, 2, 3, 8),
        5.0,
        300.0,
    )
    assert requests.gradation == 1.0  # the film sizes keep their own grading


def test_the_outline_rule_adds_to_the_a_priori_film_sizes_instead_of_replacing_them():
    """The outline rule adds to the a priori film sizes instead of replacing them."""
    plain = FeatureRecipe().size_requests(outline_features(), box_with_diagonal(2000.0))
    outlined = FeatureRecipe(conductor_outline=True).size_requests(
        outline_features(), box_with_diagonal(2000.0)
    )
    assert dict(outlined.curve_sizes) == dict(plain.curve_sizes)


def test_an_outline_size_not_below_the_coarse_size_asks_for_nothing():
    """An outline size not below the coarse size asks for nothing."""
    rules = FeatureRecipe(conductor_outline=True)
    assert (
        rules.size_requests(outline_features(), box_with_diagonal(100.0)).fields == ()
    )


def test_the_outline_sizes_must_be_positive():
    """The outline sizes must be positive."""
    with pytest.raises(ValueError, match="positive"):
        FeatureRecipe(conductor_edge_size_um=0.0)
    with pytest.raises(ValueError, match="positive"):
        FeatureRecipe(conductor_edge_grow_um=-1.0)


def test_the_global_size_recipe_asks_only_for_the_coarse_size():
    """The global-size recipe asks only for the coarse size: no curve sizes, no field, whatever the facts."""
    requests = GlobalSizeRecipe().size_requests(
        features_of(film([1, 2, 3])), box_with_diagonal(2000.0)
    )
    assert requests.max_size == pytest.approx(100.0)
    assert requests.gradation == 1.0
    assert (dict(requests.curve_sizes), requests.fields) == ({}, ())


def test_the_global_size_divisor_is_a_parameter():
    """The divisor sets the coarse size from the diagonal."""
    requests = GlobalSizeRecipe(max_size_divisor=10.0).size_requests(
        features_of(), box_with_diagonal(2000.0)
    )
    assert requests.max_size == pytest.approx(200.0)


def test_the_uniform_boundary_recipe_sizes_every_curve_of_the_model_alike():
    """Every curve of the binding gets the near size, and the size reaches the far size over the far distance."""
    requests = UniformBoundaryRecipe().size_requests(
        features_of(film([1])), box_with_diagonal(2000.0)
    )
    assert dict(requests.curve_sizes) == {tag: 5.0 for tag in range(1, 13)}
    assert requests.max_size == 300.0
    assert requests.fields == ()
    # DistMax = (max - size) / gradation: the sizes of the benchmark's former RP reference.
    assert requests.gradation == pytest.approx(295.0 / 300.0)
    assert (requests.max_size - 5.0) / requests.gradation == pytest.approx(300.0)


def test_the_uniform_boundary_recipe_ignores_the_extent_and_the_features():
    """The same binding gives the same sizes in a small model and in a large one."""
    recipe = UniformBoundaryRecipe(near_um=2.0, far_um=100.0)
    small = recipe.size_requests(features_of(), box_with_diagonal(10.0))
    large = recipe.size_requests(features_of(film([1, 2])), box_with_diagonal(9000.0))
    assert small == large
    assert small.max_size == 100.0 and set(small.curve_sizes.values()) == {2.0}


@pytest.mark.parametrize(("near", "far"), [(0.0, 300.0), (5.0, 5.0), (400.0, 300.0)])
def test_the_uniform_boundary_sizes_must_grow(near, far):
    """The near size must be positive and below the far size."""
    with pytest.raises(ValueError, match="near_um < far_um"):
        UniformBoundaryRecipe(near_um=near, far_um=far)
