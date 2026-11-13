"""The nearest-origin grid of a support region returns the same float as a scan over every origin.

``SupportRegion.distance`` searches its origins cell ring by cell ring and stops once no unseen origin can
be nearer.  Every distance a real banded build asks for is compared here with the minimum over all origins,
and a large interface measured from one small conductor (origins sparse, queries far from them) costs a
number of cells tied to the origin count, not to how far a query lies from the origins.
"""

from __future__ import annotations

import math

import pytest
import sgb_supported_cells as disk_qubit
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import (
    Conductor,
    Scene,
    catalogue,
    die_box,
    rect,
    to_lattice,
    type_bands,
)

from scgsim.geometry.sgb.bands.requests import InsetRequest, SolutionInterface
from scgsim.geometry.sgb.exact import support
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.exact.support import (  # private-import: the cells a query lists are the cost under test
    SupportRegion,
    _ring_cells,
    distance_to_primitive,
)
from scgsim.geometry.sgb.model import build_model_from_catalogue

SCENES = {**disk_qubit.SCENES, **pdk.SCENES, **pdk.NOT_WRITTEN}


def sparse_origins() -> tuple[PrismCatalogue, tuple[InsetRequest, ...]]:
    """A 4 mm die with one centred 1 um conductor, SA bands (0.1, 0.2) um measured from its edges."""
    scene = Scene(
        "sparse_origins",
        (),
        die_box(2000.0),
        [Conductor("pad", "Signal", to_lattice(rect(-0.5, -0.5, 0.5, 0.5)))],
    )
    bands = (InsetRequest("sa", SolutionInterface("substrate", "vacuum"), (0.1, 0.2)),)
    return catalogue(scene), bands


def build_input(name: str) -> tuple[PrismCatalogue, tuple[InsetRequest, ...]]:
    """The catalogue and band requests of a named scene."""
    if name == "sparse_origins":
        return sparse_origins()
    return catalogue(SCENES[name]()), type_bands()


@pytest.mark.parametrize(
    "name, at_least",
    [
        # Floors that keep the comparison from being vacuous, not a cost bound.  Ending the offsets of a
        # convex junction at their crossing (Issue #138) leaves fewer cells to sample: 56 to 92 queries here.
        pytest.param("cell20_comb", 50, marks=pytest.mark.slow),
        pytest.param("d1_die_defaults", 50, marks=pytest.mark.slow),
        ("cell22_tsv_signal", 50),
        ("cell24_sheet_circular_pad", 50),
        (
            "cell18_handoff",
            40,
        ),  # every query on a support whose origins continue past the die edge
        (
            "sparse_origins",
            10,
        ),  # one small support boundary: few queries, one of them 1 mm away
    ],
)
def test_every_distance_of_a_banded_build_equals_the_scan_over_all_origins(
    monkeypatch, name, at_least
):
    """Every distance the build asks for, to straight and arc origins alike, is the float of the scan."""
    grid_distance = SupportRegion.distance
    queries = []

    def compared(region: SupportRegion, point: tuple[float, float]) -> float:
        found = grid_distance(region, point)
        scanned = min(
            distance_to_primitive(p, point[0], point[1]) for p in region.origins
        )
        assert found == scanned, (name, point)
        queries.append(point)
        return found

    monkeypatch.setattr(SupportRegion, "distance", compared)
    catalogue_, bands = build_input(name)
    build_model_from_catalogue(catalogue_, "B", insets=bands)
    assert len(queries) >= at_least  # the build asks, so the comparison is not vacuous


def test_a_query_far_from_sparse_origins_visits_no_more_cells_than_the_grid_holds(
    monkeypatch,
):
    """Queries across a 4 mm interface, origins on a 1 um square: the cells per query stay within the grid.

    The grid has about sqrt(n) cells a side for n origins (at least four, plus one for each box edge that
    falls inside a cell), so no query may list more than that square, however far it lies from them.
    """
    distance = SupportRegion.distance
    listed = [0]
    per_query = []

    def counted_cells(*args):
        cells = _ring_cells(*args)
        listed[0] += len(cells)
        return cells

    def counted_distance(region: SupportRegion, point: tuple[float, float]) -> float:
        listed[0] = 0
        found = distance(region, point)
        side = max(4, math.isqrt(len(region.origins))) + 2
        per_query.append((listed[0], side * side, point))
        return found

    monkeypatch.setattr(support, "_ring_cells", counted_cells)
    monkeypatch.setattr(SupportRegion, "distance", counted_distance)
    catalogue_, bands = sparse_origins()
    build_model_from_catalogue(catalogue_, "B", insets=bands)
    far = [point for _, _, point in per_query if max(map(abs, point)) > 100_000]
    assert far  # a query 0.1 mm and more from the conductor, so the bound is tested where it matters
    assert all(cells <= bound for cells, bound, _ in per_query), max(per_query)
