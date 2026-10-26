"""Verify opt-in work counts on a public authored circular-hole construction."""

import json
from collections import Counter
from dataclasses import replace

import pytest
from sgb_method_cases import circular_hole_case

from scgsim.diagnostics import preparation_diagnostics
from scgsim.geometry import interpret_physical_geometry
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.construction.gmsh_occ import write_occ_geometry_from_plan


def _build(tmp_path):
    source, stack, profile = circular_hole_case()
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    plan = build_route_construction_plan(
        effective, route="A", representation_context=physical.context
    )
    return write_occ_geometry_from_plan(plan, xao_path=tmp_path / "geometry.xao")


def test_curved_work_counts_and_native_phase_boundaries(tmp_path):
    pytest.importorskip("gmsh")
    plain = _build(tmp_path / "plain")
    with preparation_diagnostics(tmp_path / "diagnostics"):
        observed = _build(tmp_path / "observed")
    # XAO includes native encoding; compare the complete source geometry records.
    assert replace(observed, metadata={}) == replace(plain, metadata={})
    rows = [
        json.loads(line)
        for path in (tmp_path / "diagnostics").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    starts = [row for row in rows if row["event"] == "start"]
    stages = Counter(row["stage"] for row in starts)
    assert stages["geometry.curved.global_planar_arrangement"] == 1
    slabs = [
        row for row in starts if row["stage"] == "geometry.curved.classify_one_z_slab"
    ]
    sweep = next(
        row["counts"]
        for row in starts
        if row["stage"] == "geometry.curved.retained_edge_sweep"
    )
    work = next(
        row["counts"] for row in starts if row["stage"] == "geometry.curved.sweep_work"
    )
    assert [row["counts"]["slab_index"] for row in slabs] == list(range(sweep["slabs"]))
    assert work["swept_walls"] == sweep["edge_slab_sweeps"]
    assert sweep["edge_slab_sweeps"] == sweep["retained_edges"] * sweep["slabs"]
    assert 0 < work["used_walls"] <= work["swept_walls"]
    repairs = stages["geometry.occ.build_shells_and_volume"]
    assert repairs == stages["geometry.occ.synchronize_before_shell"]
    assert repairs == stages["geometry.occ.synchronize_after_shell"]
    assert repairs == stages["geometry.occ.match_repaired_faces"]
    assert repairs == stages["geometry.occ.rebind_affected_faces"]
    assert stages["geometry.occ.index_shared_maps"] == 2
    ends = Counter(row["stage"] for row in rows if row["event"] == "end")
    assert stages == ends
