"""Serialized version-1 protocol fixtures; scientific outcomes use separate native evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scgsim.palace.results.amr_computation import (
    AmrAnalysisContext,
    analyze_snapshot_series,
)
from scgsim.palace.results.amr_diagnostic_io import NativeDiagnosticSource
from scgsim.palace.results.amr_diagnostics import (
    bind_native_attempts,
    read_native_amr_diagnostics,
)
from scgsim.palace.results.report_models import AmrPassSnapshot


def _record(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema": "palace.amr_diagnostics",
        "schema_version": 1,
        "record_type": kind,
        **fields,
    }


def _statistics(converged: bool = True) -> dict[str, Any]:
    return dict(
        available=True,
        algorithm="cg",
        residual_norm="preconditioned",
        final_residual_is_estimate=False,
        iterations=2,
        initial_residual=1.0,
        final_residual="NaN",
        reference_residual=1.0,
        reference_residual_source="rhs",
        relative_tolerance=1e-6,
        absolute_tolerance=0.0,
        maximum_iterations=10,
        converged=converged,
    )


def _fixture(root: Path) -> NativeDiagnosticSource:
    """Write exact protocol shapes with deliberately small arithmetic, without native execution."""
    base = root / "output/amr-diagnostics"
    base.mkdir(parents=True)
    records = {}
    for iteration, cells, dofs in ((1, 2, 3), (2, 3, 4)):
        mesh = f"mesh-{iteration}"
        records[f"solve-{iteration}.json"] = _record(
            "solve",
            iteration=iteration,
            mesh_id=mesh,
            state="completed",
            owned_global_cells=cells,
            true_dofs=dofs,
            capacitance_written=True,
            capacitance_iteration=iteration,
            capacitance_terminal_order=[7],
            terminal_records=[f"terminal-{iteration}.json"],
            mesh_paths=[],
            indicator_norm=0.25,
            indicator_semantics="native RMS",
        )
        records[f"terminal-{iteration}.json"] = _record(
            "terminal",
            iteration=iteration,
            mesh_id=mesh,
            terminal=7,
            statistics=dict(
                terminal_index=7,
                capacitance_column=0,
                main=_statistics(),
                recovery=_statistics(),
            ),
        )
    records["transition.json"] = _record(
        "transition",
        pre_iteration=1,
        next_iteration=2,
        pre_mesh_id="mesh-1",
        refined_mesh_id="refined-1",
        next_mesh_id="mesh-2",
        state="completed",
        threshold=0.2,
        theta_requested=0.7,
        eta_squared_total=5.0,
        eta_squared_marked=4.0,
        marked_error_fraction=0.8,
        marked_error_fraction_reason=None,
        mesh_conformity="NC",
        max_nc_levels=1,
        owned_global_cells_pre=2,
        owned_global_cells_post_refinement=3,
        owned_global_cells_post_rebalance=3,
        marked_count=1,
        refined_parent_count=1,
        extra_refined_parent_count=0,
        refined_mesh_paths=[],
        rank_records=["rank.json"],
    )
    records["rank.json"] = _record(
        "rank",
        rank=0,
        pre_mesh_id="mesh-1",
        refined_mesh_id="refined-1",
        owned_cells_pre=2,
        owned_cells_post_refinement=3,
        marked=[0],
        refined_parents=[0],
        fine_to_coarse=[
            dict(fine=0, coarse=0),
            dict(fine=1, coarse=0),
            dict(fine=2, coarse=1),
        ],
        marked_count=1,
        refined_parent_count=1,
        extra_refined_parent_count=0,
    )
    for name, record in records.items():
        (base / name).write_text(json.dumps(record))
    (base / "manifest.json").write_text(
        json.dumps(
            _record(
                "manifest",
                coordinate_units="nondimensional",
                mpi_size=1,
                files=list(records),
                solves=["solve-1.json", "solve-2.json"],
                transitions=["transition.json"],
            )
        )
    )
    return NativeDiagnosticSource(root, root / "output", {7: "net-pad"})


def _edit(source: NativeDiagnosticSource, filename: str, **fields: Any) -> None:
    """Change one protocol record to exercise validation of its declared identity."""
    path = source.output_dir / "amr-diagnostics" / filename
    record = json.loads(path.read_text())
    record.update(fields)
    path.write_text(json.dumps(record))


def _sealed(source: NativeDiagnosticSource) -> NativeDiagnosticSource:
    """Seal fixture bytes using the existing returned-output record format."""
    records = []
    for path in sorted((source.output_dir / "amr-diagnostics").iterdir()):
        data = path.read_bytes()
        records.append(
            dict(
                path=path.relative_to(source.run_dir).as_posix(),
                present=True,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    return replace(source, receipt={"output_files": records}, require_receipt=True)


def _snapshot(iteration: int, value: float) -> AmrPassSnapshot:
    return AmrPassSnapshot(
        iteration - 1,
        f"iteration{iteration}",
        Path(f"iteration{iteration}"),
        None,
        None,
        None,
        ((value,),),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        native_iteration=iteration,
    )


def _analysis(source: NativeDiagnosticSource):
    diagnostics = read_native_amr_diagnostics(source)
    snapshots = bind_native_attempts(
        (_snapshot(1, 4.0), _snapshot(2, 6.0)), diagnostics, source.run_dir
    )
    context = AmrAnalysisContext(
        source.run_dir, "Electrostatic", "completed", diagnostics.integrity
    )
    return analyze_snapshot_series(snapshots, context, diagnostics)


def test_explicit_native_seven_binds_column_zero_and_compact_metrics(tmp_path):
    """Native IDs are explicit identifiers, and final payload contains compact counts only."""
    analysis = _analysis(_sealed(_fixture(tmp_path)))
    assert analysis.passes[0].terminal_ids == ("net-pad",)
    observation = analysis.passes[0].terminal_solve_observation
    assert observation is not None
    assert observation.terminal_statistics[0].terminal_index == 7
    assert observation.terminal_statistics[0].capacitance_column == 0
    assert observation.terminal_statistics[0].main.final_residual == "NaN"
    transition = analysis.transitions[0]
    assert transition.marked_fraction.value == 0.5
    assert transition.marked_squared_indicator_coverage.value == 0.8
    assert transition.new_cell_amplification.value == 1
    assert transition.observations is not None
    assert transition.observations.marked_count == 1
    assert transition.observations.refined_parent_count == 1
    assert transition.dof_increase.units == "DOF"
    assert transition.adjacent_capacitance.relative_frobenius.value == pytest.approx(
        2 / 6
    )
    payload = json.loads(json.dumps(analysis.to_payload(), allow_nan=False))
    assert "marked_parents" not in payload["transitions"][0]["observations"]


def test_started_attempt_without_snapshot_remains_visible_and_stale_c_is_stripped(
    tmp_path,
):
    """A started next solve is represented even when its own C has never been written."""
    source = _fixture(tmp_path)
    _edit(
        source,
        "solve-2.json",
        state="started",
        capacitance_written=False,
        capacitance_iteration=None,
        capacitance_terminal_order=None,
        terminal_records=[],
        true_dofs=None,
    )
    diagnostics = read_native_amr_diagnostics(source)
    for snapshots in ((_snapshot(1, 4.0),), (_snapshot(1, 4.0), _snapshot(2, 4.0))):
        bound = bind_native_attempts(snapshots, diagnostics, source.run_dir)
        assert len(bound) == 2
        assert bound[1].native_iteration == 2
        assert bound[1].capacitance_matrix_f is None
        analysis = analyze_snapshot_series(
            bound,
            AmrAnalysisContext(
                tmp_path, "Electrostatic", "failed", "observed_unsealed"
            ),
            diagnostics,
        )
        assert analysis.passes[1].linear_solve_reason == "native_solve_started"
        assert analysis.passes[1].capacitance_matrix_f is None


@pytest.mark.parametrize("family", ["main", "recovery"])
def test_main_and_recovery_validity_have_distinct_quantity_scope(tmp_path, family):
    """Recovery failure warns about the estimator while only main failure invalidates C comparisons."""
    source = _fixture(tmp_path)
    path = source.output_dir / "amr-diagnostics/terminal-2.json"
    record = json.loads(path.read_text())
    record["statistics"][family]["converged"] = False
    path.write_text(json.dumps(record))
    analysis = _analysis(source)
    current = analysis.passes[1]
    assert current.capacitance_matrix_f == ((6.0,),)
    assert current.native_zz_norm.value == 0.25
    relative = analysis.transitions[0].adjacent_capacitance.relative_frobenius
    if family == "main":
        assert current.linear_solve_state == "unconverged"
        assert relative.reason == "linear_solve_unconverged"
    else:
        assert current.recovery_solve_state == "unconverged"
        assert current.linear_solve_state == "converged"
        assert relative.value == pytest.approx(2 / 6)


def test_partial_missing_terminal_retains_attempt_but_not_validity(tmp_path):
    """An inventoried missing terminal leaves an explicit incomplete unsealed observation."""
    source = _fixture(tmp_path)
    (source.output_dir / "amr-diagnostics/terminal-2.json").unlink()
    diagnostics = read_native_amr_diagnostics(source)
    assert diagnostics.integrity == "observed_unsealed"
    assert diagnostics.reason == "native_diagnostic_records_incomplete"
    assert diagnostics.solves[1].converged == (None,)
    assert _analysis(source).passes[1].linear_solve_state == "unavailable"


def test_strict_reader_rejects_changed_and_missing_consumed_bytes(tmp_path):
    """Sealed input cannot silently become an unsealed or modified native observation."""
    source = _sealed(_fixture(tmp_path))
    _edit(source, "terminal-2.json", terminal=8)
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        read_native_amr_diagnostics(source)
    (source.output_dir / "amr-diagnostics/terminal-2.json").unlink()
    with pytest.raises(FileNotFoundError):
        read_native_amr_diagnostics(source)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"solves": ["unlisted.json"]}, "unlisted"),
        ({"files": ["../escape.json"]}, "confined"),
        ({"schema_version": 2}, "record identity"),
    ],
)
def test_manifest_rejects_unsupported_or_unlisted_sources(tmp_path, mutation, message):
    """Only the authoritative manifest schema and confined explicit inventory are accepted."""
    source = _fixture(tmp_path)
    _edit(source, "manifest.json", **mutation)
    with pytest.raises(ValueError, match=message):
        read_native_amr_diagnostics(source)


def test_fine_map_must_support_declared_refined_parents(tmp_path):
    """The actual owned fine-to-coarse relation validates producer parent counts."""
    source = _fixture(tmp_path)
    _edit(source, "rank.json", refined_parents=[1])
    with pytest.raises(ValueError, match="refined parents disagree"):
        read_native_amr_diagnostics(source)


def test_terminal_column_and_mesh_must_match_solve_identity(tmp_path):
    """A terminal record from another column or mesh cannot qualify this C matrix."""
    source = _fixture(tmp_path)
    _edit(source, "terminal-2.json", mesh_id="another-mesh")
    with pytest.raises(ValueError, match="another solve or mesh"):
        read_native_amr_diagnostics(source)


def test_unlisted_files_are_not_discovered(tmp_path):
    """An unrelated malformed JSON file outside manifest references is never read."""
    source = _fixture(tmp_path)
    (source.output_dir / "amr-diagnostics/unlisted.json").write_text("invalid json")
    assert len(read_native_amr_diagnostics(source).solves) == 2


def test_missing_solve_record_preserves_existing_raw_snapshot(tmp_path):
    """Enabling diagnostics never removes a readable snapshot whose sidecar is missing."""
    source = _fixture(tmp_path)
    (source.output_dir / "amr-diagnostics/solve-2.json").unlink()
    diagnostics = read_native_amr_diagnostics(source)
    original = replace(
        _snapshot(2, 6.0),
        error_norm=0.9,
        elapsed_total_s=12.0,
        peak_node_memory_mb=15.0,
        terminal_ids=("net-pad",),
    )
    bound = bind_native_attempts(
        (_snapshot(1, 4.0), original), diagnostics, source.run_dir
    )
    assert bound[1] == original
    analysis = analyze_snapshot_series(
        bound,
        AmrAnalysisContext(tmp_path, "Electrostatic", "completed", "observed_unsealed"),
        diagnostics,
    )
    current = analysis.passes[1]
    assert current.capacitance_matrix_f == ((6.0,),)
    assert current.native_zz_norm.value == 0.9
    assert current.elapsed_cumulative_s == 12.0
    assert current.peak_node_memory_mb == 15.0
    assert current.linear_solve_state == "unavailable"


def test_started_partial_terminal_set_cannot_establish_whole_pass_convergence(tmp_path):
    """One completed excitation is not proof that a started multi-terminal solve is done."""
    source = _fixture(tmp_path)
    _edit(
        source,
        "solve-2.json",
        state="started",
        capacitance_written=False,
        capacitance_iteration=None,
        capacitance_terminal_order=None,
    )
    analysis = _analysis(source)
    current = analysis.passes[1]
    assert current.linear_solve_state == "unavailable"
    assert current.linear_solve_reason == "capacitance_terminal_order_unavailable"
    assert current.terminal_solve_observation is not None
    assert (
        current.terminal_solve_observation.terminal_statistics[0].main.converged is True
    )


def test_partial_missing_rank_retains_declared_counts_with_validation_state(tmp_path):
    """Global native counts remain visible while set-dependent derived metrics stay unavailable."""
    source = _fixture(tmp_path)
    (source.output_dir / "amr-diagnostics/rank.json").unlink()
    transition = _analysis(source).transitions[0]
    assert transition.observations is not None
    assert transition.observations.marked_count == 1
    assert transition.observations.refined_parent_count == 1
    assert transition.observations.extra_refined_parent_count == 0
    assert transition.observations.counts_validation == "rank_records_incomplete"
    assert transition.closure_amplification.value is None
    assert transition.new_cell_amplification.value is None


def test_refinement_map_cannot_drop_an_owned_coarse_parent(tmp_path):
    """Every old owned cell must contribute a leaf under pure refinement."""
    source = _fixture(tmp_path)
    _edit(
        source, "rank.json", fine_to_coarse=[dict(fine=i, coarse=0) for i in range(3)]
    )
    with pytest.raises(ValueError, match="retain every owned coarse parent"):
        read_native_amr_diagnostics(source)


@pytest.mark.parametrize("missing", [False, True])
def test_strict_reader_verifies_listed_mesh_even_without_parsing_it(tmp_path, missing):
    """Every inventoried file participates in sealed integrity, including opaque mesh bytes."""
    source = _fixture(tmp_path)
    base = source.output_dir / "amr-diagnostics"
    manifest = json.loads((base / "manifest.json").read_text())
    manifest["files"].append("mesh.bin")
    (base / "manifest.json").write_text(json.dumps(manifest))
    mesh = base / "mesh.bin"
    mesh.write_bytes(b"opaque protocol mesh bytes")
    source = _sealed(source)
    if missing:
        mesh.unlink()
        with pytest.raises(FileNotFoundError):
            read_native_amr_diagnostics(source)
    else:
        mesh.write_bytes(b"modified mesh bytes")
        with pytest.raises(ValueError, match="receipt hash mismatch"):
            read_native_amr_diagnostics(source)


def test_partial_missing_listed_mesh_is_explicitly_incomplete(tmp_path):
    """An unsealed manifest cannot hide missing opaque inventory behind readable JSON."""
    source = _fixture(tmp_path)
    base = source.output_dir / "amr-diagnostics"
    manifest = json.loads((base / "manifest.json").read_text())
    manifest["files"].append("absent-mesh.bin")
    (base / "manifest.json").write_text(json.dumps(manifest))
    observations = read_native_amr_diagnostics(source)
    assert observations.integrity == "observed_unsealed"
    assert observations.reason == "native_diagnostic_records_incomplete"
    assert len(observations.solves) == 2
    assert observations.missing_files == ("absent-mesh.bin",)
    assert _analysis(source).diagnostic_missing_files == ("absent-mesh.bin",)


def test_serialized_started_transition_after_final_solve_has_no_synthetic_pass(
    tmp_path,
):
    """The manifest can publish a started refinement before publishing its next solve."""
    source = _fixture(tmp_path)
    base = source.output_dir / "amr-diagnostics"
    manifest = json.loads((base / "manifest.json").read_text())
    manifest["solves"] = ["solve-1.json"]
    manifest["files"] = [
        name
        for name in manifest["files"]
        if name not in {"solve-2.json", "terminal-2.json", "rank.json"}
    ]
    (base / "manifest.json").write_text(json.dumps(manifest))
    transition_record = json.loads((base / "transition.json").read_text())
    transition_record["state"] = "started"
    for name in (
        "owned_global_cells_post_refinement",
        "owned_global_cells_post_rebalance",
        "marked_count",
        "refined_parent_count",
        "extra_refined_parent_count",
        "refined_mesh_paths",
        "rank_records",
    ):
        transition_record.pop(name)
    (base / "transition.json").write_text(json.dumps(transition_record))
    diagnostics = read_native_amr_diagnostics(_sealed(source))
    snapshots = bind_native_attempts((_snapshot(1, 4.0),), diagnostics, tmp_path)
    analysis = analyze_snapshot_series(
        snapshots,
        AmrAnalysisContext(tmp_path, "Electrostatic", "failed", "receipt_bound"),
        diagnostics,
    )
    assert len(analysis.passes) == 1
    assert len(analysis.transitions) == 1
    transition = analysis.transitions[0]
    assert (transition.before_iteration, transition.after_iteration) == (1, 2)
    assert transition.reason == "native_transition_after_pass_unavailable"
    assert transition.marked_squared_indicator_coverage.value == 0.8
    assert transition.closure_extra_parents.reason == "native_transition_started"
    assert transition.dof_growth_ratio.value is None
    assert transition.adjacent_capacitance.relative_frobenius.value is None
    assert transition.observations is not None
    assert transition.observations.state == "started"
    assert transition.observations.premesh_id == "mesh-1"
    assert transition.observations.next_mesh_id == "mesh-2"
    assert transition.observations.source.endswith("transition.json")
    assert transition.observations.true_dofs_after is None
