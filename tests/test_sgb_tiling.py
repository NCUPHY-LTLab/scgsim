"""Tile-parallel derivation is identical to the single-tile result (design brief section 9)."""

from __future__ import annotations

import pytest
from sgb_scenes import airbridge, pad_in_ground, random_layout, random_segments

from scgsim.geometry.sgb.derivation.labelled import derive_from_catalogue
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.noding.snap_rounding import snap_round
from scgsim.geometry.sgb.noding.tiling import (
    TiledEngine,
    tile_boxes,
    tile_cut_lines,
    tile_index,
)


@pytest.mark.parametrize("tiles", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(8))
def test_noded_edges_identical_for_random_segments(seed, tiles):
    """Noded edges identical for random segments."""
    segments = random_segments(seed, 80, [60, 400, 20_000][seed % 3])
    assert snap_round(segments, TiledEngine(tiles)).edges == snap_round(segments).edges


@pytest.mark.parametrize("tiles", [2, 3, 4])
def test_running_examples_have_identical_ids_for_every_tile_count(tiles):
    """Running examples have identical ids for every tile count."""
    for catalogue in (pad_in_ground("A"), pad_in_ground("B"), airbridge()):
        reference = derive_from_catalogue(catalogue).identity_sets()
        assert (
            derive_from_catalogue(catalogue, tiles=tiles).identity_sets() == reference
        )


@pytest.mark.parametrize("tiles", [2, 3, 4])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("route", ["A", "B"])
def test_random_layouts_have_identical_ids_for_every_tile_count(seed, route, tiles):
    """Random layouts have identical ids for every tile count."""
    catalogue = random_layout(30 + 7 * seed, seed=seed, route=route)
    reference = derive_from_catalogue(catalogue).identity_sets()
    assert derive_from_catalogue(catalogue, tiles=tiles).identity_sets() == reference


@pytest.mark.slow
@pytest.mark.parametrize(("tiles", "workers"), [(2, 2), (3, 4), (4, 8)])
def test_worker_processes_give_the_same_ids_for_any_tile_and_worker_count(
    tiles, workers
):
    """Worker processes give the same ids for any tile and worker count."""
    catalogue = random_layout(60, seed=5, route="B")
    reference = derive_from_catalogue(catalogue).identity_digest()
    assert (
        derive_from_catalogue(catalogue, tiles=tiles, workers=workers).identity_digest()
        == reference
    )


def test_parallel_and_serial_stages_are_booked_separately():
    """Parallel and serial stages are booked separately."""
    result = derive_from_catalogue(
        random_layout(40, seed=2, route="B"), tiles=2, workers=2
    )
    assert set(result.stage_kind.values()) == {"serial", "parallel"}
    assert 0.0 < result.amdahl["amdahl_parallel_fraction"] < 1.0
    assert {"labelling", "atomic_ids", "noding_kernels"} <= set(result.timings)


def test_every_pixel_belongs_to_exactly_one_tile_and_cuts_lie_between_pixels():
    """Every pixel belongs to exactly one tile and cuts lie between pixels."""
    x_cuts, y_cuts = tile_cut_lines(-50, 49, 4), tile_cut_lines(0, 99, 4)
    boxes = tile_boxes(x_cuts, y_cuts)
    assert len(boxes) == 16
    for x in range(-50, 50):
        for y in (0, 37, 99):
            owners = [
                i
                for i, (x0, y0, x1, y1) in enumerate(boxes)
                if x0 <= x < x1 and y0 <= y < y1
            ]
            assert owners == [tile_index(x_cuts, y_cuts, 4, (x, y))]


def test_a_tiled_engine_needs_at_least_one_tile():
    """A tiled engine needs at least one tile."""
    with pytest.raises(ValueError, match="tiles must be >= 1"):
        TiledEngine(0)


@pytest.mark.slow
def test_a_large_layout_has_one_identity_for_serial_tiled_and_worker_builds():
    """400 features in both routes: serial, 4x4 tiles in one process, and 4x4 tiles on 4 workers agree."""
    for route in ("A", "B"):
        catalogue = random_layout(400, seed=11, route=route)
        digests = {
            derive_from_catalogue(
                catalogue, tiles=tiles, workers=workers
            ).identity_digest()
            for tiles, workers in ((1, 1), (4, 1), (4, 4))
        }
        assert len(digests) == 1


@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("tiles", [2, 3])
def test_the_whole_model_has_one_identity_for_every_tile_count(route, tiles):
    """Assembly, groups and validation are tile-independent too: vertices, edges, faces, volumes and groups."""
    catalogue = (
        pad_in_ground("A" if route == "A" else "B") if route != "C" else airbridge()
    )
    reference = build_model_from_catalogue(catalogue, route)
    tiled = build_model_from_catalogue(catalogue, route, tiles=tiles)
    assert tiled.identity_sets() == reference.identity_sets()


@pytest.mark.slow
def test_the_whole_model_has_one_identity_for_worker_processes():
    """A random Route B layout: serial, tiled in one process, tiled on four workers."""
    catalogue = random_layout(120, seed=4, route="B")
    digests = {
        build_model_from_catalogue(
            catalogue, "B", tiles=tiles, workers=workers
        ).identity_digest()
        for tiles, workers in ((1, 1), (3, 1), (3, 4))
    }
    assert len(digests) == 1
