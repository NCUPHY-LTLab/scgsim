"""The nearest-origin grid of a support region returns the same float as a scan over every origin.

``SupportRegion.distance`` searches its origins cell ring by cell ring and stops once no unseen origin can
be nearer.  Every distance a real banded build asks for is compared here with the minimum over all origins.
"""

from __future__ import annotations

import pytest
import sgb_supported_cells as disk_qubit
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import catalogue, type_bands

from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.support import SupportRegion, distance_to_primitive

SCENES = {**disk_qubit.SCENES, **pdk.SCENES}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("cell20_comb", marks=pytest.mark.slow),
        pytest.param("d1_die_defaults", marks=pytest.mark.slow),
        "cell22_tsv_signal",
        "cell24_sheet_circular_pad",
    ],
)
def test_every_distance_of_a_banded_build_equals_the_scan_over_all_origins(
    monkeypatch, name
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
    build_model_from_catalogue(catalogue(SCENES[name]()), "B", insets=type_bands())
    assert len(queries) > 100  # the build asks, so the comparison is not vacuous
