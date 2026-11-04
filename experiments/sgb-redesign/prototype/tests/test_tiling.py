"""Tile-parallel noding is identical to the single-tile result (brief section 9)."""

from __future__ import annotations

import pytest

from helpers import random_segments
from sgbproto.pipeline import build, id_sets, ids_digest
from sgbproto.scenes import airbridge, pad_in_ground, random_layout
from sgbproto.snap_rounding import snap_round
from sgbproto.tiling import TiledEngine


@pytest.mark.parametrize("tiles", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(8))
def test_noded_edges_identical_for_random_segments(seed, tiles):
    segments = random_segments(seed, 80, [60, 400, 20_000][seed % 3])
    assert snap_round(segments, TiledEngine(tiles)).edges == snap_round(segments).edges


@pytest.mark.parametrize("tiles", [1, 2, 3, 4])
def test_running_examples_have_identical_ids_for_every_tile_count(tiles):
    for spec in (pad_in_ground("A"), pad_in_ground("B"), airbridge()):
        reference = id_sets(build(spec))
        assert id_sets(build(spec, tiles=tiles)) == reference


@pytest.mark.parametrize("tiles", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("route", ["A", "B"])
def test_random_layouts_have_identical_ids_for_every_tile_count(seed, route, tiles):
    spec = random_layout(30 + 7 * seed, seed=seed, route=route)
    assert id_sets(build(spec, tiles=tiles)) == id_sets(build(spec))


@pytest.mark.parametrize("tiles,workers", [(2, 2), (3, 4), (4, 8)])
def test_worker_processes_give_the_same_ids_for_any_tile_and_worker_count(tiles, workers):
    spec = random_layout(60, seed=5, route="B")
    assert ids_digest(build(spec, tiles=tiles, workers=workers)) == ids_digest(build(spec))


def test_parallel_stages_are_booked_and_serial_stages_are_booked():
    result = build(random_layout(40, seed=2, route="B"), tiles=2, workers=2)
    kinds = set(result.stage_kind.values())
    assert kinds == {"serial", "parallel"}
    assert 0.0 < result.amdahl["amdahl_parallel_fraction"] < 1.0
    assert {"labelling", "atomic_ids", "wall_runs", "shells", "noding_kernels"} <= set(result.timings)
