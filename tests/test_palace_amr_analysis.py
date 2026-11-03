"""Hand-calculated AMR metrics, terminal alignment and unavailable observations."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from scgsim.palace.results.amr_analysis import analyze_amr, indexed_terminal_ids
from scgsim.palace.results.amr_capacitance import (
    compare_adjacent_capacitance,
    compare_reference_capacitance,
)
from scgsim.palace.results.amr_computation import (
    AmrAnalysisContext,
    analyze_snapshot_series,
)
from scgsim.palace.results.amr_metrics import refinement_metrics
from scgsim.palace.results.amr_models import (
    AmrCellIdentity,
    AmrDiagnosticObservations,
    AmrRefinementObservation,
    AmrSolveObservation,
    CapacitanceMatrix,
    CapacitanceReference,
)
from scgsim.palace.results.report_models import AmrPassSnapshot, PalaceResultSelection


def _snapshot(
    iteration: int | None,
    matrix: CapacitanceMatrix | None = ((6.0,),),
    terminal_ids: tuple[str, ...] | None = ("net-pad",),
) -> AmrPassSnapshot:
    return AmrPassSnapshot(
        pass_index=0 if iteration is None else iteration - 1,
        source=f"iteration{iteration}",
        path=Path(f"iteration{iteration}"),
        frequencies_ghz=None,
        eig_columns=None,
        port_epr=None,
        capacitance_matrix_f=matrix,
        surface_epr=None,
        error_indicators={"Norm": 0.25},
        error_norm=0.25,
        degrees_of_freedom=100,
        mesh_elements=10,
        elapsed_total_s=None,
        peak_node_memory_mb=None,
        native_iteration=iteration,
        terminal_ids=terminal_ids,
    )


def _solves(*snapshots: AmrPassSnapshot) -> tuple[AmrSolveObservation, ...]:
    records = []
    for snapshot in snapshots:
        assert snapshot.native_iteration is not None
        assert snapshot.terminal_ids is not None
        records.append(
            AmrSolveObservation(
                snapshot.native_iteration,
                snapshot.terminal_ids,
                (True,) * len(snapshot.terminal_ids),
                "explicit-native-solve-record",
            )
        )
    return tuple(records)


def _context(status: str = "completed") -> AmrAnalysisContext:
    return AmrAnalysisContext(Path("run"), "Electrostatic", status, "receipt_bound")


def _cells(*indices: int) -> frozenset[AmrCellIdentity]:
    return frozenset(AmrCellIdentity("mesh-1", 0, index) for index in indices)


def _refinement() -> AmrRefinementObservation:
    return AmrRefinementObservation(
        before_iteration=1,
        after_iteration=2,
        premesh_id="mesh-1",
        cells_before=10,
        cells_after=16,
        marked_parents=_cells(0, 2),
        refined_parents=_cells(0, 1),
        marked_squared_indicator_sum=5.0,
        total_squared_indicator_sum=10.0,
        source="explicit-native-refinement-record",
    )


def test_one_terminal_adjacent_and_explicit_reference_use_different_denominators():
    """Adjacent change is 2/6; the supplied reference difference is 4/2."""
    previous = _snapshot(1, ((4.0,),))
    current = _snapshot(2, ((6.0,),))
    diagnostics = AmrDiagnosticObservations(solves=_solves(previous, current))
    reference = CapacitanceReference(
        ((2.0,),), ("net-pad",), "F", "independent-reference"
    )
    analysis = analyze_snapshot_series(
        (previous, current), _context(), diagnostics, reference
    )
    adjacent = analysis.transitions[0].adjacent_capacitance
    reference_change = analysis.passes[1].reference_capacitance
    assert adjacent.relative_frobenius.value == pytest.approx(1 / 3)
    assert adjacent.absolute_delta_f.value == ((2.0,),)
    assert reference_change is not None
    assert reference_change.relative_frobenius.value == 2
    assert reference_change.absolute_delta_f.value == ((4.0,),)
    assert adjacent.normalization == "current"
    assert reference_change.normalization == "reference"
    assert reference_change.source == "independent-reference"


def test_multiterminal_reference_reorders_both_axes_and_converts_picofarads():
    """The identified current matrix is exactly twice the reordered pF reference."""
    current = _snapshot(2, ((4e-12, -1e-12), (-1e-12, 2e-12)), ("net-A", "net-B"))
    reference = CapacitanceReference(
        ((1.0, -0.5), (-0.5, 2.0)), ("net-B", "net-A"), "pF", "measured-two-port"
    )
    comparison = compare_reference_capacitance(current, reference, _solves(current))
    assert comparison.relative_frobenius.value == pytest.approx(1)
    assert comparison.absolute_delta_f.terminal_ids == ("net-A", "net-B")
    assert comparison.absolute_delta_f.value is not None
    assert comparison.absolute_delta_f.value[0] == pytest.approx((2e-12, 0.5e-12))
    assert comparison.absolute_delta_f.value[1] == pytest.approx((0.5e-12, 1e-12))
    assert comparison.absolute_delta_f.units == "F"


def test_adjacent_terminal_reordering_is_identity_based():
    """Equal matrices with reversed native axes have zero physical change."""
    previous = _snapshot(1, ((1.0, -0.5), (-0.5, 2.0)), ("net-B", "net-A"))
    current = _snapshot(2, ((2.0, -0.5), (-0.5, 1.0)), ("net-A", "net-B"))
    comparison = compare_adjacent_capacitance(
        previous, current, _solves(previous, current)
    )
    assert comparison.relative_frobenius.value == 0
    assert comparison.absolute_delta_f.value == ((0.0, 0.0), (0.0, 0.0))


@pytest.mark.parametrize("units, value", [("F", 2e-12), ("pF", 2.0), ("fF", 2000.0)])
def test_explicit_reference_units_share_the_same_farads(units, value):
    """Unit conversion preserves an explicit two-picofarad reference."""
    snapshot = _snapshot(1, ((2e-12,),))
    reference = CapacitanceReference(((value,),), ("net-pad",), units, "supplied")
    comparison = compare_reference_capacitance(snapshot, reference, _solves(snapshot))
    assert comparison.relative_frobenius.value == pytest.approx(0, abs=1e-15)


def test_refinement_metrics_use_actual_sets_and_squared_indicators():
    """M and P need not be nested: actual extra parents are one, not P-count minus M-count."""
    previous = _snapshot(1)
    current = replace(_snapshot(2), degrees_of_freedom=160, mesh_elements=16)
    diagnostics = AmrDiagnosticObservations(
        solves=_solves(previous, current), refinements=(_refinement(),), reason=None
    )
    transition = analyze_snapshot_series(
        (previous, current), _context(), diagnostics
    ).transitions[0]
    assert transition.marked_fraction.value == 0.2
    assert transition.marked_squared_indicator_coverage.value == 0.5
    assert transition.closure_amplification.value == 1
    assert transition.closure_extra_parents.value == 1
    assert transition.new_cell_amplification.value == 3
    assert transition.dof_growth_ratio.value == 1.6
    assert transition.dof_increase.value == 60


def test_cell_ids_preserve_rank_scope_for_equal_local_indices():
    """Rank-local zero identifies two different coarse cells on the same pre-mesh."""
    marked = frozenset(
        (AmrCellIdentity("mesh-1", 0, 0), AmrCellIdentity("mesh-1", 1, 0))
    )
    refined = marked | frozenset((AmrCellIdentity("mesh-1", 1, 1),))
    metrics = refinement_metrics(
        replace(_refinement(), marked_parents=marked, refined_parents=refined)
    )
    assert metrics.marked_fraction.value == 0.2
    assert metrics.closure_amplification.value == 1.5
    assert metrics.closure_extra_parents.value == 1


def test_wrong_premesh_identity_is_not_merged_into_native_sets():
    """A cell from another pre-refinement mesh makes the set-based metrics unavailable."""
    wrong = frozenset((AmrCellIdentity("other-mesh", 0, 0),))
    metrics = refinement_metrics(replace(_refinement(), marked_parents=wrong))
    assert metrics.marked_fraction.value is None
    assert metrics.closure_extra_parents.reason == "coarse_cell_mesh_identity_mismatch"


def test_missing_squared_indicator_sums_do_not_use_marked_fraction_as_proxy():
    """Native marking counts alone cannot establish captured estimator energy."""
    metrics = refinement_metrics(
        replace(_refinement(), marked_squared_indicator_sum=None)
    )
    assert metrics.marked_fraction.value == 0.2
    assert metrics.marked_squared_indicator_coverage.value is None
    assert (
        metrics.marked_squared_indicator_coverage.reason
        == "native_squared_indicator_sums_unavailable"
    )


def test_zero_marking_and_indicator_denominators_remain_explicit():
    """A zero marked fraction is real; dividing by no marked parents is undefined."""
    observation = replace(
        _refinement(),
        marked_parents=frozenset(),
        refined_parents=frozenset(),
        marked_squared_indicator_sum=0,
        total_squared_indicator_sum=0,
    )
    metrics = refinement_metrics(observation)
    assert metrics.marked_fraction.value == 0
    assert metrics.closure_extra_parents.value == 0
    assert metrics.closure_amplification.reason == "zero_marked_parents"
    assert metrics.new_cell_amplification.reason == "zero_marked_parents"
    assert (
        metrics.marked_squared_indicator_coverage.reason == "zero_squared_indicator_sum"
    )


def test_one_solve_without_amr_has_no_invented_transition_or_reference():
    """A lone native solve has neither adjacent change nor an implicit final reference."""
    analysis = analyze_snapshot_series((_snapshot(1),), _context())
    assert analysis.transitions == ()
    assert analysis.reference is None
    assert analysis.passes[0].reference_capacitance is None


@pytest.mark.parametrize(
    "after, reason",
    [
        (3, "native_iteration_gap"),
        (1, "duplicate_native_iteration"),
        (None, "native_iteration_unavailable"),
    ],
)
def test_missing_or_nongenuine_rounds_are_not_linked(after, reason):
    """A display sequence cannot replace the missing native iteration identity."""
    previous = _snapshot(1)
    current = _snapshot(after)
    transition = analyze_snapshot_series((previous, current), _context()).transitions[0]
    assert transition.availability == "unavailable"
    assert transition.reason == reason
    assert transition.adjacent_capacitance.relative_frobenius.value is None
    assert transition.dof_growth_ratio.reason == reason


def test_identical_capacitance_across_genuine_rounds_keeps_zero_change():
    """Two native rounds remain a real transition even when their C matrices agree."""
    previous, current = _snapshot(1), _snapshot(2)
    diagnostics = AmrDiagnosticObservations(solves=_solves(previous, current))
    analysis = analyze_snapshot_series((previous, current), _context(), diagnostics)
    assert len(analysis.passes) == 2
    assert len(analysis.transitions) == 1
    assert analysis.transitions[0].adjacent_capacitance.relative_frobenius.value == 0


def test_legacy_solve_validity_is_unknown_despite_completed_driver_state():
    """A completed driver does not establish convergence for its terminal solves."""
    previous, current = _snapshot(1), _snapshot(2)
    analysis = analyze_snapshot_series((previous, current), _context())
    assert analysis.run_status == "completed"
    assert analysis.passes[1].capacitance_matrix_f == ((6.0,),)
    assert analysis.passes[1].linear_solve_state == "unavailable"
    assert (
        analysis.transitions[0].adjacent_capacitance.relative_frobenius.reason
        == "linear_solve_validity_unavailable"
    )


def test_unconverged_solve_keeps_raw_capacitance_and_unavailable_change():
    """An unsuccessful terminal KSP solve remains visible beside its raw C value."""
    previous, current = _snapshot(1), _snapshot(2)
    solved = _solves(previous, current)
    diagnostics = AmrDiagnosticObservations(
        solves=(solved[0], replace(solved[1], converged=(False,)))
    )
    reference = CapacitanceReference(((2.0,),), ("net-pad",), "F", "explicit")
    analysis = analyze_snapshot_series(
        (previous, current), _context("failed"), diagnostics, reference
    )
    assert analysis.run_status == "failed"
    assert analysis.passes[1].capacitance_matrix_f == ((6.0,),)
    assert analysis.passes[1].linear_solve_state == "unconverged"
    assert analysis.passes[1].terminal_solve_observation == diagnostics.solves[1]
    assert (
        analysis.transitions[0].adjacent_capacitance.relative_frobenius.reason
        == "linear_solve_unconverged"
    )
    assert analysis.passes[1].reference_capacitance is not None
    assert analysis.passes[1].reference_capacitance.relative_frobenius.value is None


@pytest.mark.parametrize(
    "state, reason",
    [(False, "linear_solve_unconverged"), (None, "linear_solve_validity_unavailable")],
)
def test_every_terminal_requires_explicit_native_validity(state, reason):
    """One unknown or unconverged excitation prevents a whole-matrix comparison."""
    snapshot = _snapshot(1, ((2.0, -0.5), (-0.5, 1.0)), ("net-A", "net-B"))
    assert snapshot.terminal_ids is not None
    assert snapshot.capacitance_matrix_f is not None
    solve = AmrSolveObservation(1, snapshot.terminal_ids, (True, state), "native")
    reference = CapacitanceReference(
        snapshot.capacitance_matrix_f, snapshot.terminal_ids, "F", "reference"
    )
    comparison = compare_reference_capacitance(snapshot, reference, (solve,))
    assert comparison.relative_frobenius.reason == reason


def test_reference_terminal_identity_mismatch_does_not_compare_positions():
    """Equal dimensions and values cannot replace a missing conductor-net identity."""
    snapshot = _snapshot(1)
    reference = CapacitanceReference(((6.0,),), ("another-net",), "F", "explicit")
    comparison = compare_reference_capacitance(snapshot, reference, _solves(snapshot))
    assert (
        comparison.relative_frobenius.reason == "reference_terminal_identity_mismatch"
    )
    assert comparison.absolute_delta_f.value is None


def test_missing_terminal_identity_is_not_filled_from_matrix_width():
    """The matrix dimension provides no conductor-net identity by itself."""
    snapshot = _snapshot(1, terminal_ids=None)
    reference = CapacitanceReference(((6.0,),), ("net-pad",), "F", "explicit")
    comparison = compare_reference_capacitance(snapshot, reference, ())
    assert comparison.relative_frobenius.reason == "terminal_identity_unavailable"


def test_zero_current_norm_keeps_entry_delta_but_not_normalized_change():
    """An absolute one-farad delta exists while normalization by zero does not."""
    previous, current = _snapshot(1, ((1.0,),)), _snapshot(2, ((0.0,),))
    comparison = compare_adjacent_capacitance(
        previous, current, _solves(previous, current)
    )
    assert comparison.relative_frobenius.reason == "zero_current_matrix_norm"
    assert comparison.absolute_delta_f.value == ((1.0,),)


def test_zero_reference_norm_keeps_entry_delta_but_not_relative_error():
    """An explicitly zero reference never produces a fabricated relative error."""
    snapshot = _snapshot(1, ((1.0,),))
    reference = CapacitanceReference(((0.0,),), ("net-pad",), "F", "zero-reference")
    comparison = compare_reference_capacitance(snapshot, reference, _solves(snapshot))
    assert comparison.relative_frobenius.reason == "zero_reference_matrix_norm"
    assert comparison.absolute_delta_f.value == ((1.0,),)


def test_zero_previous_dofs_do_not_fabricate_growth_ratio():
    """The observed increase can exist while its normalized growth is undefined."""
    previous, current = replace(_snapshot(1), degrees_of_freedom=0), _snapshot(2)
    transition = analyze_snapshot_series((previous, current), _context()).transitions[0]
    assert transition.dof_growth_ratio.reason == "zero_previous_true_dofs"
    assert transition.dof_increase.value == 100


def test_native_zz_norm_is_the_recorded_value_not_a_marking_proxy():
    """The estimator norm preserves a recorded zero independently of diagnostics."""
    snapshot = replace(_snapshot(1), error_norm=0.0)
    assert (
        analyze_snapshot_series((snapshot,), _context()).passes[0].native_zz_norm.value
        == 0
    )
    missing = replace(snapshot, error_norm=None)
    assert (
        analyze_snapshot_series((missing,), _context()).passes[0].native_zz_norm.reason
        == "native_zz_norm_unavailable"
    )


def test_transition_cell_count_mismatch_is_not_accepted_as_refinement_evidence():
    """A matching iteration pair must also match the observed owned-cell counts."""
    previous, current = _snapshot(1), replace(_snapshot(2), mesh_elements=99)
    diagnostics = AmrDiagnosticObservations(refinements=(_refinement(),), reason=None)
    transition = analyze_snapshot_series(
        (previous, current), _context(), diagnostics
    ).transitions[0]
    assert transition.marked_fraction.reason == "native_transition_cell_count_mismatch"


def test_indexed_terminal_ids_use_structured_net_ids_in_native_order():
    """Display aliases never replace canonical indexed conductor-net identities."""
    index = {
        "entries": [
            {
                "section": "Boundaries.Terminal",
                "index": 2,
                "net_id": "net-B",
                "terminal_name": "alias-A",
            },
            {
                "section": "Boundaries.Terminal",
                "index": 1,
                "net_id": "net-A",
                "terminal_name": "alias-B",
            },
        ]
    }
    assert indexed_terminal_ids(index) == ("net-A", "net-B")
    index["entries"][0].pop("net_id")
    assert indexed_terminal_ids(index) is None


@dataclass(frozen=True)
class _InspectedReport:
    """Minimal detached inspection record exercising the structural input contract."""

    run_dir: Path
    problem: str
    passes: tuple[AmrPassSnapshot, ...]
    selection: PalaceResultSelection
    failure: None
    provenance: dict
    completeness: str


def test_partial_inspection_input_retains_run_state_and_json_payload(tmp_path):
    """Partial reports remain explicitly unsealed and retain an unsuccessful attempt."""
    selection = PalaceResultSelection(
        "missing",
        "iteration1",
        Path("iteration1"),
        0,
        "latest_complete_iteration_after_failed_attempt",
        "observed_unsealed",
    )
    result = _InspectedReport(
        tmp_path,
        "Electrostatic",
        (_snapshot(1),),
        selection,
        None,
        {"returned_receipt": {"status": "failed"}},
        "partial",
    )
    analysis = analyze_amr(result)
    assert analysis.run_status == "failed"
    assert analysis.integrity == "observed_unsealed"
    payload = analysis.to_payload()
    assert payload["run_dir"] == str(tmp_path)
    assert json.loads(json.dumps(payload, allow_nan=False))["passes"][0][
        "capacitance_matrix_f"
    ] == [[6.0]]


def test_path_only_input_does_not_bypass_existing_result_resolution(tmp_path):
    """A directory cannot impersonate a validated or inspected source object."""
    with pytest.raises(TypeError, match="resolved result or an inspected report"):
        analyze_amr(tmp_path)


@pytest.mark.parametrize(
    "values, ids, units, source",
    [
        (((1.0,),), ("same", "same"), "F", "ref"),
        (((1.0,),), ("one", "two"), "F", "ref"),
        (((True,),), ("one",), "F", "ref"),
        (((1.0,),), ("one",), "unknown", "ref"),
        (((1.0,),), ("one",), "F", ""),
    ],
)
def test_reference_requires_explicit_shape_identity_units_and_source(
    values, ids, units, source
):
    """Malformed reference contracts fail without inventing missing reference facts."""
    with pytest.raises((TypeError, ValueError)):
        CapacitanceReference(values, ids, units, source)


def test_nonuniform_multiterminal_change_uses_the_full_frobenius_norm():
    """Two diagonal changes use all four current entries in the normalization."""
    previous = _snapshot(1, ((2.0, -1.0), (-1.0, 3.0)), ("net-A", "net-B"))
    current = _snapshot(2, ((3.0, -1.0), (-1.0, 4.0)), ("net-A", "net-B"))
    comparison = compare_adjacent_capacitance(
        previous, current, _solves(previous, current)
    )
    assert comparison.relative_frobenius.value == pytest.approx((2 / 27) ** 0.5)
    assert comparison.absolute_delta_f.value == ((1.0, 0.0), (0.0, 1.0))


def test_explicit_reference_remains_separate_from_missing_adjacent_round():
    """An iteration gap blocks adjacent change, not a valid identified reference."""
    previous = _snapshot(1, ((4.0,),))
    current = _snapshot(3, ((6.0,),))
    reference = CapacitanceReference(((2.0,),), ("net-pad",), "F", "independent")
    diagnostics = AmrDiagnosticObservations(solves=_solves(previous, current))
    analysis = analyze_snapshot_series(
        (previous, current), _context(), diagnostics, reference
    )
    assert (
        analysis.transitions[0].adjacent_capacitance.relative_frobenius.reason
        == "native_iteration_gap"
    )
    assert analysis.passes[1].reference_capacitance is not None
    assert analysis.passes[1].reference_capacitance.relative_frobenius.value == 2


def test_reference_detaches_mutable_input_without_relabeling_its_source():
    """Later caller edits cannot change the explicitly supplied reference matrix."""
    values = [[2.0]]
    ids = ["net-pad"]
    reference = CapacitanceReference(
        cast(CapacitanceMatrix, values),
        cast(tuple[str, ...], ids),
        "pF",
        "external-measurement",
    )
    values[0][0] = 9.0
    ids[0] = "changed"
    assert reference.values == ((2.0,),)
    assert reference.terminal_ids == ("net-pad",)
    assert reference.source == "external-measurement"


def test_started_native_solve_without_capacitance_retains_its_iteration_warning():
    """An explicit started round is not lost just because it has no C table yet."""
    previous = _snapshot(1, ((4.0,),))
    started = _snapshot(2, None)
    diagnostics = AmrDiagnosticObservations(solves=_solves(previous))
    analysis = analyze_snapshot_series(
        (previous, started), _context("running"), diagnostics
    )
    assert len(analysis.passes) == 2
    assert analysis.passes[1].native_iteration == 2
    assert analysis.passes[1].capacitance_matrix_f is None
    assert analysis.passes[1].linear_solve_state == "unavailable"
    assert (
        analysis.transitions[0].adjacent_capacitance.relative_frobenius.reason
        == "capacitance_unavailable"
    )


@pytest.mark.parametrize(
    "ids, reason",
    [
        (None, "terminal_identity_unavailable"),
        (("net-A",), "linear_solve_terminal_identity_mismatch"),
    ],
)
def test_pass_convergence_is_bound_to_the_observed_matrix_terminal_ids(ids, reason):
    """A valid solve of another conductor cannot certify this pass as converged."""
    snapshot = _snapshot(1, terminal_ids=ids)
    observation = AmrSolveObservation(1, ("net-B",), (True,), "native-for-net-B")
    diagnostics = AmrDiagnosticObservations(solves=(observation,))
    analyzed = analyze_snapshot_series((snapshot,), _context(), diagnostics).passes[0]
    assert analyzed.linear_solve_state == "unavailable"
    assert analyzed.linear_solve_reason == reason
    assert analyzed.terminal_solve_observation == observation
    assert analyzed.capacitance_matrix_f == ((6.0,),)


def test_true_dof_increase_has_a_count_unit_instead_of_a_dimensionless_unit():
    """Growth ratio is dimensionless; the observed increase counts native DOFs."""
    previous, current = _snapshot(1), replace(_snapshot(2), degrees_of_freedom=160)
    transition = analyze_snapshot_series((previous, current), _context()).transitions[0]
    assert transition.dof_growth_ratio.units == "1"
    assert transition.dof_increase.units == "DOF"
    assert transition.dof_increase.value == 60


def test_payload_encodes_raw_nonfinite_observations_without_numerical_replacement():
    """RFC JSON preserves IEEE observations as explicit native-style string markers."""
    snapshot = replace(
        _snapshot(1, ((float("nan"),),)),
        elapsed_total_s=float("inf"),
        peak_node_memory_mb=-float("inf"),
    )
    reference = CapacitanceReference(
        ((float("inf"),),), ("net-pad",), "F", "explicit-nonfinite-observation"
    )
    analysis = analyze_snapshot_series((snapshot,), _context(), reference=reference)
    assert analysis.passes[0].elapsed_cumulative_s == float("inf")
    payload = json.loads(json.dumps(analysis.to_payload(), allow_nan=False))
    assert payload["passes"][0]["capacitance_matrix_f"] == [["NaN"]]
    assert payload["passes"][0]["elapsed_cumulative_s"] == "Infinity"
    assert payload["passes"][0]["peak_node_memory_mb"] == "-Infinity"
    assert payload["reference"]["values"] == [["Infinity"]]
    assert (
        payload["passes"][0]["reference_capacitance"]["relative_frobenius"]["value"]
        is None
    )


def test_declared_started_transition_after_last_solve_keeps_indicator_observations():
    """A native refinement attempt survives without inventing its next solve or C table."""
    snapshot = _snapshot(1)
    transition = replace(
        _refinement(),
        state="started",
        cells_after=None,
        marked_parents=None,
        refined_parents=None,
        next_mesh_id="mesh-2",
        refined_mesh_id="mesh-1-refined",
        counts_validation="transition_started",
    )
    diagnostics = AmrDiagnosticObservations(refinements=(transition,))
    analysis = analyze_snapshot_series((snapshot,), _context("failed"), diagnostics)
    assert len(analysis.passes) == 1
    assert len(analysis.transitions) == 1
    result = analysis.transitions[0]
    assert (result.before_iteration, result.after_iteration) == (1, 2)
    assert result.reason == "native_transition_after_pass_unavailable"
    assert result.marked_squared_indicator_coverage.value == 0.5
    assert result.closure_amplification.reason == "native_transition_started"
    assert result.new_cell_amplification.value is None
    assert result.dof_increase.reason == "native_transition_after_pass_unavailable"
    assert result.dof_increase.units == "DOF"
    assert result.adjacent_capacitance.relative_frobenius.value is None
    assert result.observations is not None
    assert result.observations.source == transition.source
    assert result.observations.state == "started"
    assert result.observations.next_mesh_id == "mesh-2"
    assert result.observations.true_dofs_before == snapshot.degrees_of_freedom
    assert result.observations.true_dofs_after is None


def test_declared_transition_with_missing_before_snapshot_retains_native_evidence():
    """An absent previous solve invalidates C and DOF comparisons rather than erasing the transition."""
    snapshot = replace(_snapshot(2), mesh_elements=16)
    diagnostics = AmrDiagnosticObservations(refinements=(_refinement(),))
    result = analyze_snapshot_series((snapshot,), _context(), diagnostics).transitions[
        0
    ]
    assert result.reason == "native_transition_before_pass_unavailable"
    assert result.adjacent_capacitance.relative_frobenius.reason == result.reason
    assert result.marked_fraction.value == 0.2
    assert result.observations is not None
    assert result.observations.true_dofs_before is None
    assert result.observations.true_dofs_after == snapshot.degrees_of_freedom


def test_transition_union_keeps_existing_gap_and_declared_endpoints_without_duplicates():
    """Recorded pass gaps and explicit native rounds coexist without repeating a completed pair."""
    first = _snapshot(1)
    second = replace(_snapshot(2), mesh_elements=16)
    third = _snapshot(3)
    diagnostics = AmrDiagnosticObservations(refinements=(_refinement(),))
    complete = analyze_snapshot_series((first, second), _context(), diagnostics)
    assert [
        (item.before_iteration, item.after_iteration) for item in complete.transitions
    ] == [(1, 2)]
    gapped = analyze_snapshot_series((first, third), _context(), diagnostics)
    assert [
        (item.before_iteration, item.after_iteration) for item in gapped.transitions
    ] == [(1, 3), (1, 2)]
    assert gapped.transitions[0].reason == "native_iteration_gap"
    assert gapped.transitions[1].reason == "native_transition_after_pass_unavailable"


def test_declared_transition_without_either_snapshot_keeps_explicit_identity():
    """Native transition identity remains reportable even with no readable solve snapshots."""
    diagnostics = AmrDiagnosticObservations(refinements=(_refinement(),))
    analysis = analyze_snapshot_series((), _context("failed"), diagnostics)
    assert analysis.passes == ()
    assert len(analysis.transitions) == 1
    assert analysis.transitions[0].reason == "native_transition_both_passes_unavailable"
    assert analysis.transitions[0].marked_squared_indicator_coverage.value == 0.5
