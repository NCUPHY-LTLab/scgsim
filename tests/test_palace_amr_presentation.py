"""Display and benchmark integration for detached AMR analysis."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

import scgsim.palace.presentation.trust as trust_module
from scgsim.palace.presentation.amr import show_amr_analysis
from scgsim.palace.presentation.trust import PalaceTrustReport
from scgsim.palace.results.amr_models import (
    AmrMatrixMetric,
    AmrMetric,
    AmrPassAnalysis,
    AmrTransitionAnalysis,
    CapacitanceComparison,
    CapacitanceReference,
    PalaceAmrAnalysis,
)
from scgsim.palace.results.report_models import (
    AmrPassSnapshot,
    PalaceResultSelection,
)


def test_amr_report_keeps_zero_values_gaps_raw_capacitance_and_solve_warnings() -> None:
    """Render observed zeros, retain missing intervals, and show their reasons."""
    analysis = _analysis()

    report = show_amr_analysis(analysis)
    figures = report.figures

    assert report.analysis is analysis
    assert report.data["amr_analysis"] == analysis.to_payload()
    assert figures["marking_and_coverage"].data[0].y == (0.0, None)
    assert figures["marking_and_coverage"].data[0].connectgaps is False
    assert "adjacent_capacitance_change" in figures
    assert "explicit_reference_capacitance_error" in figures
    assert (
        figures["adjacent_capacitance_change"]
        is not figures["explicit_reference_capacitance_error"]
    )
    assert "linear_solve_unconverged" in report.tables_html()
    assert (
        "&quot;terminal_ids&quot;: [&quot;A&quot;, &quot;B&quot;]"
        in report.tables_html()
    )
    assert "not observed in source data" in report.tables_html()


def test_pass_figures_break_lines_across_missing_native_iterations() -> None:
    """Keep real observations visible without connecting iteration one to three."""
    source = _analysis()
    passes = (
        replace(source.passes[0], native_iteration=1),
        replace(
            source.passes[1],
            native_iteration=3,
            native_zz_norm=AmrMetric(0.1, "available", units="1"),
            reference_capacitance=source.passes[0].reference_capacitance,
        ),
    )

    report = show_amr_analysis(replace(source, passes=passes))

    zz = report.figures["native_zz_norm"].data[0]
    reference = report.figures["explicit_reference_capacitance_error"].data[0]
    assert zz.x == (1, 2, 3)
    assert zz.y == (0.2, None, 0.1)
    assert zz.connectgaps is False
    assert reference.x == (1, 2, 3)
    assert reference.y[1] is None


def test_trust_benchmark_reuses_the_supplied_analysis_instance(tmp_path: Path) -> None:
    """Keep benchmark fields and serialized AMR data attached to one run analysis."""
    analysis = _analysis(run_dir=tmp_path)
    trust = _trust_report(tmp_path)

    benchmark = trust.show_simulation_benchmark(amr_analysis=analysis)

    assert benchmark.amr_analysis is analysis
    assert benchmark.data["amr_analysis"] == analysis.to_payload()


def test_partial_benchmark_analyzes_once_and_reuses_that_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute one analysis for a partial trust report and reuse it in report data."""
    trust = _trust_report(tmp_path)
    analysis = _analysis(run_dir=tmp_path)
    calls: list[tuple[PalaceTrustReport, CapacitanceReference | None]] = []

    def analyze(
        source: PalaceTrustReport,
        reference: CapacitanceReference | None = None,
    ) -> PalaceAmrAnalysis:
        calls.append((source, reference))
        return analysis

    monkeypatch.setattr(trust_module, "analyze_amr", analyze)

    benchmark = trust.show_simulation_benchmark()

    assert calls == [(trust, None)]
    assert benchmark.amr_analysis is analysis
    assert benchmark.data["amr_analysis"] == analysis.to_payload()


def test_partial_trust_report_is_an_analysis_source(tmp_path: Path) -> None:
    """Allow native snapshot analysis directly from an inspected trust report."""
    trust = _trust_report(tmp_path)

    benchmark = trust.show_simulation_benchmark()

    assert benchmark.amr_analysis is not None
    assert benchmark.amr_analysis.run_dir == tmp_path
    assert benchmark.amr_analysis.integrity == "observed_unsealed"
    assert benchmark.data["amr_analysis"] == benchmark.amr_analysis.to_payload()


def test_benchmark_exports_native_cost_records_without_crossing_iteration_gaps(
    tmp_path: Path,
) -> None:
    """Export exact per-pass identities and leave the 1-to-3 timer delta unavailable."""
    snapshots = (
        _cost_snapshot(0, 1, 10.0),
        _cost_snapshot(1, 3, 35.0),
        _cost_snapshot(2, 4, 40.0),
    )
    trust = replace(_trust_report(tmp_path), passes=snapshots)

    payload = trust.show_simulation_benchmark().data["pass_cost_records"]

    assert [item["native_iteration"] for item in payload] == [1, 3, 4]
    assert [item["elapsed_pass_s"] for item in payload] == [10.0, None, 5.0]
    assert [item["elapsed_pass_reason"] for item in payload] == [
        None,
        "native_iteration_gap",
        None,
    ]
    assert [item["elapsed_cumulative_s"] for item in payload] == [10.0, 35.0, 40.0]


def test_benchmark_cost_reasons_cover_missing_iteration_counter_and_reset(
    tmp_path: Path,
) -> None:
    """Retain raw cumulative times when native identity or timer continuity is absent."""
    snapshots = (
        _cost_snapshot(0, 2, 20.0),
        _cost_snapshot(1, None, 25.0),
        _cost_snapshot(2, 1, 30.0),
        _cost_snapshot(3, 2, 28.0),
    )
    trust = replace(_trust_report(tmp_path), passes=snapshots)

    payload = trust.show_simulation_benchmark().data["pass_cost_records"]

    assert [item["elapsed_pass_s"] for item in payload] == [None, None, 30.0, None]
    assert [item["elapsed_pass_reason"] for item in payload] == [
        "missing_previous_native_iteration",
        "native_iteration_unavailable",
        None,
        "cumulative_elapsed_reset",
    ]
    assert [item["elapsed_cumulative_s"] for item in payload] == [
        20.0,
        25.0,
        30.0,
        28.0,
    ]


def test_benchmark_cost_reason_covers_missing_cumulative_counter(
    tmp_path: Path,
) -> None:
    """Keep adjacent per-pass time unavailable when either cumulative timer is absent."""
    snapshots = (
        _cost_snapshot(0, 1, 10.0),
        _cost_snapshot(1, 2, None),
        _cost_snapshot(2, 3, 30.0),
        _cost_snapshot(3, 4, 35.0),
    )
    trust = replace(_trust_report(tmp_path), passes=snapshots)

    payload = trust.show_simulation_benchmark().data["pass_cost_records"]

    assert [item["elapsed_pass_s"] for item in payload] == [10.0, None, None, 5.0]
    assert [item["elapsed_pass_reason"] for item in payload] == [
        None,
        "cumulative_elapsed_unavailable",
        "previous_cumulative_elapsed_unavailable",
        None,
    ]


@pytest.mark.parametrize("invalid_elapsed", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("invalid_iteration", [1, 2])
def test_benchmark_cost_preserves_nonfinite_counters_but_rejects_deltas(
    tmp_path: Path,
    invalid_elapsed: float,
    invalid_iteration: int,
) -> None:
    """Keep raw nonfinite timers while refusing contaminated pass-time arithmetic."""
    iterations = (1, 2, 3) if invalid_iteration == 1 else (1, 2, 3, 4)
    elapsed = (
        (invalid_elapsed, 20.0, 25.0)
        if invalid_iteration == 1
        else (10.0, invalid_elapsed, 30.0, 35.0)
    )
    trust = replace(
        _trust_report(tmp_path),
        passes=tuple(
            _cost_snapshot(index, iteration, elapsed[index])
            for index, iteration in enumerate(iterations)
        ),
    )

    payload = trust.show_simulation_benchmark().data["pass_cost_records"]

    if invalid_iteration == 1:
        assert [item["elapsed_pass_s"] for item in payload] == [None, None, 5.0]
        assert [item["elapsed_pass_reason"] for item in payload] == [
            "nonfinite_cumulative_elapsed",
            "previous_cumulative_elapsed_nonfinite",
            None,
        ]
    else:
        assert [item["elapsed_pass_s"] for item in payload] == [
            10.0,
            None,
            None,
            5.0,
        ]
        assert [item["elapsed_pass_reason"] for item in payload] == [
            None,
            "nonfinite_cumulative_elapsed",
            "previous_cumulative_elapsed_nonfinite",
            None,
        ]

    raw_timer = payload[invalid_iteration - 1]["elapsed_cumulative_s"]
    if math.isnan(invalid_elapsed):
        assert math.isnan(raw_timer)
    else:
        assert raw_timer == invalid_elapsed


def test_trust_benchmark_rejects_analysis_from_a_different_run(tmp_path: Path) -> None:
    """Prevent analysis from being combined with unrelated benchmark observations."""
    analysis = _analysis(run_dir=tmp_path / "other")
    trust = _trust_report(tmp_path)

    with pytest.raises(ValueError, match="run identity, status, and integrity"):
        trust.show_simulation_benchmark(amr_analysis=analysis)

    wrong_problem = replace(_analysis(run_dir=tmp_path), problem="Eigenmode")
    with pytest.raises(ValueError, match="run identity, status, and integrity"):
        trust.show_simulation_benchmark(amr_analysis=wrong_problem)

    wrong_status = replace(_analysis(run_dir=tmp_path), run_status="completed")
    with pytest.raises(ValueError, match="run identity, status, and integrity"):
        trust.show_simulation_benchmark(amr_analysis=wrong_status)

    wrong_integrity = replace(_analysis(run_dir=tmp_path), integrity="receipt_bound")
    with pytest.raises(ValueError, match="run identity, status, and integrity"):
        trust.show_simulation_benchmark(amr_analysis=wrong_integrity)


def _analysis(*, run_dir: Path = Path("/tmp/amr-presentation")) -> PalaceAmrAnalysis:
    """Build native-shaped passes with one measured zero and one unavailable transition."""
    available_zero = AmrMetric(0.0, "available", units="1")
    unavailable = AmrMetric(None, "unavailable", "not observed in source data", "1")
    capacitance_change = CapacitanceComparison(
        relative_frobenius=AmrMetric(0.2, "available", units="1"),
        absolute_delta_f=unavailable_comparison_matrix(),
        normalization="current",
        source=None,
    )
    explicit_reference_change = CapacitanceComparison(
        relative_frobenius=AmrMetric(0.05, "available", units="1"),
        absolute_delta_f=AmrMatrixMetric(
            value=((0.1, 0.0), (0.0, 0.1)),
            terminal_ids=("A", "B"),
            availability="available",
        ),
        normalization="reference",
        source="explicit test reference",
    )
    passes = (
        AmrPassAnalysis(
            native_iteration=0,
            source="results/palace/palace.json",
            capacitance_matrix_f=((1.0, 0.0), (0.0, 2.0)),
            terminal_ids=("A", "B"),
            native_zz_norm=AmrMetric(0.2, "available", units="1"),
            reference_capacitance=explicit_reference_change,
            linear_solve_state="unconverged",
            linear_solve_reason="linear_solve_unconverged",
        ),
        AmrPassAnalysis(
            native_iteration=2,
            source="results/palace/iteration01/palace.json",
            capacitance_matrix_f=None,
            terminal_ids=None,
            native_zz_norm=unavailable,
            reference_capacitance=None,
        ),
    )
    transitions = (
        AmrTransitionAnalysis(
            before_iteration=0,
            after_iteration=1,
            availability="available",
            reason=None,
            marked_fraction=available_zero,
            marked_squared_indicator_coverage=available_zero,
            closure_amplification=AmrMetric(1.0, "available", units="1"),
            closure_extra_parents=AmrMetric(0, "available", units="parent cells"),
            new_cell_amplification=AmrMetric(2.0, "available", units="1"),
            dof_growth_ratio=AmrMetric(1.1, "available", units="1"),
            dof_increase=AmrMetric(100, "available", units="DOFs"),
            adjacent_capacitance=capacitance_change,
        ),
        AmrTransitionAnalysis(
            before_iteration=1,
            after_iteration=None,
            availability="unavailable",
            reason="not observed in source data",
            marked_fraction=unavailable,
            marked_squared_indicator_coverage=unavailable,
            closure_amplification=unavailable,
            closure_extra_parents=unavailable,
            new_cell_amplification=unavailable,
            dof_growth_ratio=unavailable,
            dof_increase=unavailable,
            adjacent_capacitance=capacitance_change,
        ),
    )
    return PalaceAmrAnalysis(
        run_dir=run_dir,
        problem="Electrostatic",
        run_status="partial",
        integrity="observed_unsealed",
        passes=passes,
        transitions=transitions,
        reference=CapacitanceReference(
            values=((1.0, 0.0), (0.0, 2.0)),
            terminal_ids=("A", "B"),
            units="F",
            source="test reference",
        ),
    )


def unavailable_comparison_matrix():
    """Build an explicitly unavailable elementwise capacitance comparison."""
    return AmrMatrixMetric(
        value=None,
        terminal_ids=None,
        availability="unavailable",
        reason="not observed in source data",
    )


def _cost_snapshot(
    pass_index: int,
    native_iteration: int | None,
    elapsed_total_s: float | None,
) -> AmrPassSnapshot:
    """Build one native cost snapshot with no synthetic physics observations."""
    return AmrPassSnapshot(
        pass_index=pass_index,
        source=f"iteration{native_iteration}",
        path=Path(f"iteration{native_iteration}"),
        frequencies_ghz=None,
        eig_columns=None,
        port_epr=None,
        capacitance_matrix_f=None,
        surface_epr=None,
        error_indicators=None,
        error_norm=None,
        degrees_of_freedom=1_000_000,
        mesh_elements=10,
        elapsed_total_s=elapsed_total_s,
        peak_node_memory_mb=None,
        native_iteration=native_iteration,
    )


def _trust_report(run_dir: Path) -> PalaceTrustReport:
    """Build the minimum partial-run context required by benchmark projection."""
    return PalaceTrustReport(
        run_dir=run_dir,
        problem="Electrostatic",
        route="route_a",
        profile="test",
        completeness="partial",
        latest_source="none",
        selection=PalaceResultSelection(
            final_snapshot_status="missing",
            selected_source=None,
            selected_path=None,
            selected_pass_index=None,
            reason="no_complete_snapshot",
            integrity="observed_unsealed",
        ),
        failure=None,
        identity={},
        passes=(),
        amr_tolerance=None,
        amr_max_passes=None,
        durations={},
        cost={},
        provenance={"mesh_summary": {}},
        theme="light",
    )
