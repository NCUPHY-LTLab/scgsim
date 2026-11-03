"""Export observed SimplePad AMR series and scientific comparison figures.

The Palace analysis owns metric formulas. This module selects its recorded
values for the four-case study; it never supplies a reference or fills gaps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class CaseSeries:
    """An identified study arm with the analysis and Benchmark it exported."""

    case: Mapping[str, Any]
    analysis: Mapping[str, Any]
    benchmark: Mapping[str, Any]


@dataclass(frozen=True)
class PlotSelection:
    """Select two recorded columns and a display-only unit conversion."""

    xkey: str
    ykey: str
    scale: float = 1.0


_CSV_FIELDS: tuple[str, ...] = (
    "case",
    "route",
    "native_iteration",
    "true_dofs",
    "mesh_elements",
    "elapsed_cumulative_s",
    "elapsed_pass_s",
    "elapsed_pass_reason",
    "peak_node_memory_mb",
    "capacitance_f",
    "native_zz_norm",
    "linear_solve_state",
    "recovery_solve_state",
    "main_iterations",
    "main_final_residual",
    "main_reference_residual",
    "main_residual_norm",
    "recovery_iterations",
    "recovery_final_residual",
    "recovery_reference_residual",
    "recovery_residual_norm",
    "closure_factor",
    "extra_refined_parents",
    "new_cell_amplification",
    "marked_fraction",
    "indicator_coverage",
    "relative_capacitance_change",
    "dof_growth_ratio",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the measured exports, write their table, and draw sampled points."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args(argv)
    cases = load_case_series(arguments.study_dir)
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_observations(cases, output / "observations.csv")
    write_transitions(cases, output / "refinement-transitions.csv")
    figures = render_figures(cases, output)
    summary = {
        "reference": None,
        "figures": figures,
        "cases": [case_summary(case) for case in cases],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


def load_case_series(study_dir: Path) -> tuple[CaseSeries, ...]:
    """Read only explicitly listed exports, verifying their recorded digests."""
    root = study_dir.expanduser().resolve()
    manifest = _read_mapping(root / "study-manifest.json")
    if manifest.get("status") != "analysis_attempts_complete":
        raise ValueError("comparison requires the completed analysis-attempt stage")
    cases = []
    for case in manifest["cases"]:
        analysis = _read_export(root, case, "amr_analysis")
        benchmark = _read_export(root, case, "benchmark")
        if analysis.get("reference") is not None:
            raise ValueError("the first SimplePad batch does not supply a reference")
        cases.append(CaseSeries(case, analysis, benchmark))
    return tuple(cases)


def _read_export(root: Path, case: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Keep an explicitly reported export error separate from available data."""
    if case.get("analysis_status") in {
        "analysis_error",
        "not_run_preparation_failed",
    }:
        return {}
    relative = case[f"{name}_path"]
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("study export path leaves its study directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = case.get(f"{name}_sha256")
    if (
        expected is not None
        and hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise ValueError(f"study export digest mismatch: {relative}")
    return _read_mapping(path)


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return payload


def write_observations(cases: Sequence[CaseSeries], destination: Path) -> None:
    """Export raw observations and existing metrics; missing values stay blank."""
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerows(observation_rows(case))


def observation_rows(case: CaseSeries) -> list[dict[str, Any]]:
    """Join a transition to its exact native after-iteration, never its position."""
    transitions = {
        item["after_iteration"]: item
        for item in case.analysis.get("transitions", ())
        if item.get("after_iteration") is not None
    }
    costs = {
        item["native_iteration"]: item
        for item in case.benchmark.get("pass_cost_records", ())
        if item.get("native_iteration") is not None
    }
    rows = []
    for solve in case.analysis.get("passes", ()):
        transition = transitions.get(solve.get("native_iteration"), {})
        adjacent = transition.get("adjacent_capacitance") or {}
        cost = costs.get(solve.get("native_iteration"), {})
        statistics = _single_terminal_statistics(solve)
        main = statistics.get("main") or {}
        recovery = statistics.get("recovery") or {}
        rows.append(
            {
                "case": case.case["case_id"],
                "route": case.case["route_id"],
                "native_iteration": solve.get("native_iteration"),
                "true_dofs": solve.get("degrees_of_freedom"),
                "mesh_elements": solve.get("mesh_elements"),
                "elapsed_cumulative_s": solve.get("elapsed_cumulative_s"),
                "elapsed_pass_s": cost.get("elapsed_pass_s"),
                "elapsed_pass_reason": cost.get("elapsed_pass_reason"),
                "peak_node_memory_mb": solve.get("peak_node_memory_mb"),
                "capacitance_f": _single_terminal_capacitance(solve),
                "native_zz_norm": metric_value(solve, "native_zz_norm"),
                "linear_solve_state": solve.get("linear_solve_state"),
                "recovery_solve_state": solve.get("recovery_solve_state"),
                "main_iterations": main.get("iterations"),
                "main_final_residual": main.get("final_residual"),
                "main_reference_residual": main.get("reference_residual"),
                "main_residual_norm": main.get("residual_norm"),
                "recovery_iterations": recovery.get("iterations"),
                "recovery_final_residual": recovery.get("final_residual"),
                "recovery_reference_residual": recovery.get("reference_residual"),
                "recovery_residual_norm": recovery.get("residual_norm"),
                "closure_factor": metric_value(transition, "closure_amplification"),
                "extra_refined_parents": metric_value(
                    transition, "closure_extra_parents"
                ),
                "new_cell_amplification": metric_value(
                    transition, "new_cell_amplification"
                ),
                "marked_fraction": metric_value(transition, "marked_fraction"),
                "indicator_coverage": metric_value(
                    transition, "marked_squared_indicator_coverage"
                ),
                "relative_capacitance_change": metric_value(
                    adjacent, "relative_frobenius"
                ),
                "dof_growth_ratio": metric_value(transition, "dof_growth_ratio"),
            }
        )
    return rows


def _single_terminal_statistics(solve: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = solve.get("terminal_solve_observation") or {}
    statistics = observation.get("terminal_statistics") or ()
    if not statistics:
        return {}
    if len(statistics) != 1:
        raise ValueError("the SimplePad study expects one Terminal's solve statistics")
    return statistics[0]


def write_transitions(cases: Sequence[CaseSeries], destination: Path) -> None:
    """Keep refinement costs even when the next solve has no readable matrix."""
    fields: tuple[str, ...] = (
        "case",
        "route",
        "before_iteration",
        "after_iteration",
        "marked_count",
        "refined_parent_count",
        "extra_refined_parent_count",
        "cells_before",
        "cells_after",
        "true_dofs_before",
        "true_dofs_after",
        "dof_growth_ratio",
        "closure_factor",
        "new_cell_amplification",
        "marked_fraction",
        "indicator_coverage",
        "availability",
        "reason",
    )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer: csv.DictWriter[str] = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerows(transition_rows(case))


def transition_rows(case: CaseSeries) -> list[dict[str, Any]]:
    """Select native parent counts and the analysis owner's transition metrics."""
    rows = []
    for transition in case.analysis.get("transitions", ()):
        observations = transition.get("observations") or {}
        rows.append(
            {
                "case": case.case["case_id"],
                "route": case.case["route_id"],
                "before_iteration": transition.get("before_iteration"),
                "after_iteration": transition.get("after_iteration"),
                "marked_count": observations.get("marked_count"),
                "refined_parent_count": observations.get("refined_parent_count"),
                "extra_refined_parent_count": observations.get(
                    "extra_refined_parent_count"
                ),
                "cells_before": observations.get("cells_before"),
                "cells_after": observations.get("cells_after"),
                "true_dofs_before": observations.get("true_dofs_before"),
                "true_dofs_after": observations.get("true_dofs_after"),
                "dof_growth_ratio": metric_value(transition, "dof_growth_ratio"),
                "closure_factor": metric_value(transition, "closure_amplification"),
                "new_cell_amplification": metric_value(
                    transition, "new_cell_amplification"
                ),
                "marked_fraction": metric_value(transition, "marked_fraction"),
                "indicator_coverage": metric_value(
                    transition, "marked_squared_indicator_coverage"
                ),
                "availability": transition.get("availability"),
                "reason": transition.get("reason"),
            }
        )
    return rows


def metric_value(record: Mapping[str, Any], key: str) -> float | int | None:
    """Select an available scalar without replacing unavailable data by zero."""
    metric = record.get(key)
    if not isinstance(metric, Mapping) or metric.get("availability") != "available":
        return None
    return metric.get("value")


def _single_terminal_capacitance(solve: Mapping[str, Any]) -> float | str | None:
    matrix = solve.get("capacitance_matrix_f")
    if matrix is None:
        return None
    if len(matrix) != 1 or len(matrix[0]) != 1:
        raise ValueError("the SimplePad study expects a one-Terminal matrix")
    return matrix[0][0]


def case_summary(case: CaseSeries) -> dict[str, Any]:
    """Retain the last recorded observation and actual attempt state for a table."""
    rows = observation_rows(case)
    metadata = case.benchmark.get("performance_metadata", {})
    error = case.case.get("analysis_error") or {}
    failure = metadata.get("failure") or {}
    return {
        "case": case.case["case_id"],
        "route": case.case["route_id"],
        "preparation_status": case.case.get("preparation_status"),
        "exit_code": case.case.get("run_exit_code"),
        "run_status": case.analysis.get("run_status", case.case.get("run_status")),
        "integrity": case.analysis.get("integrity"),
        "analysis_status": case.case.get("analysis_status"),
        "analysis_error_type": error.get("type"),
        "recorded_solve_count": len(rows),
        "last_observation": rows[-1] if rows else None,
        "failure_category": failure.get("category"),
    }


def observed_route_ids(cases: Sequence[CaseSeries]) -> tuple[str, ...]:
    """Facet only routes with recorded solve or refinement observations."""
    return tuple(
        route
        for route in ("route-a", "route-b")
        if any(
            case.case["route_id"] == route
            and (case.analysis.get("passes") or case.analysis.get("transitions"))
            for case in cases
        )
    )


def render_figures(cases: Sequence[CaseSeries], output: Path) -> list[str]:
    """Produce standalone SVGs from observed points, grouped by geometry route."""
    from matplotlib import pyplot as plt  # noqa: PLC0415 -- CSV export stays usable without the optional plotting runtime.

    routes = observed_route_ids(cases)
    if not routes:
        return []
    figures = []
    specifications = (
        ("zz-work", "native_zz_norm", "Native ZZ norm", 1.0),
        ("capacitance-work", "capacitance_f", "Observed pad capacitance (fF)", 1e15),
        (
            "capacitance-change-work",
            "relative_capacitance_change",
            "Adjacent capacitance change",
            1.0,
        ),
        ("dof-growth-work", "dof_growth_ratio", "True DOFs after / before", 1.0),
    )
    with plt.rc_context(
        {
            "figure.facecolor": "#161616",
            "axes.facecolor": "#161616",
            "axes.edgecolor": "#aaaaaa",
            "axes.labelcolor": "#eeeeee",
            "text.color": "#eeeeee",
            "xtick.color": "#dddddd",
            "ytick.color": "#dddddd",
            "grid.color": "#444444",
            "svg.fonttype": "none",
            "font.size": 10,
        }
    ):
        for stem, quantity, label, scale in specifications:
            figure, axes = plt.subplots(
                len(routes),
                2,
                figsize=(11, 3.5 * len(routes)),
                layout="constrained",
                squeeze=False,
            )
            for row, route in enumerate(routes):
                for column, (xkey, xlabel) in enumerate(
                    (
                        ("true_dofs", "Global true DOFs"),
                        ("elapsed_cumulative_s", "Native cumulative time (s)"),
                    )
                ):
                    axis = axes[row, column]
                    for case in cases:
                        if case.case["route_id"] == route:
                            _plot_series(
                                axis, case, PlotSelection(xkey, quantity, scale)
                            )
                    axis.set(title=route.upper(), xlabel=xlabel, ylabel=label)
                    axis.grid(alpha=0.5)
                    if axis.has_data():
                        axis.legend()
            if any(axis.has_data() for axis in axes.flat):
                filename = f"{stem}.svg"
                figure.savefig(output / filename)
                figures.append(filename)
            plt.close(figure)
        figures.extend(_render_refinement(cases, output))
        figures.extend(_render_parent_counts(cases, output))
        figures.extend(_render_marking(cases, output))
    return figures


def _plot_series(
    axis: Axes, case: CaseSeries, selection: PlotSelection, *, transitions: bool = False
) -> None:
    rows = transition_rows(case) if transitions else observation_rows(case)
    xkey, ykey, scale = selection.xkey, selection.ykey, selection.scale
    # Nonfinite observations remain in CSV; a finite coordinate is needed to draw a point.
    points = [
        row for row in rows if _plot_number(row[xkey]) and _plot_number(row[ykey])
    ]
    if not points:
        return
    nonconformal = case.case["amr_nonconformal"]
    color = "#f0b95b" if nonconformal else "#66b5ff"
    label = "NC(1)" if nonconformal else "Conformal"
    axis.scatter(
        [row[xkey] for row in points],
        [row[ykey] * scale for row in points],
        color=color,
        marker="^" if nonconformal else "o",
        label=label,
        s=28,
    )
    if ykey == "capacitance_f":
        unconverged = [
            row for row in points if row["linear_solve_state"] == "unconverged"
        ]
        if unconverged:
            axis.scatter(
                [row[xkey] for row in unconverged],
                [row[ykey] * scale for row in unconverged],
                marker="x",
                color="#ff6e6e",
                label=f"{label}: linear solve unconverged",
                s=55,
            )
    if ykey == "native_zz_norm":
        unconverged = [
            row for row in points if row["recovery_solve_state"] == "unconverged"
        ]
        if unconverged:
            axis.scatter(
                [row[xkey] for row in unconverged],
                [row[ykey] for row in unconverged],
                marker="x",
                color="#ff6e6e",
                s=55,
                label=f"{label}: recovery unconverged",
            )


def _render_refinement(cases: Sequence[CaseSeries], output: Path) -> list[str]:
    from matplotlib import pyplot as plt  # noqa: PLC0415 -- native refinement plots use the optional plotting runtime.

    routes = observed_route_ids(cases)
    figure, axes = plt.subplots(
        len(routes),
        2,
        figsize=(11, 3.5 * len(routes)),
        layout="constrained",
        squeeze=False,
    )
    for row, route in enumerate(routes):
        for column, (quantity, label) in enumerate(
            (
                ("closure_factor", "Refined parents / marked cells"),
                ("new_cell_amplification", "New cells / marked cells"),
            )
        ):
            axis = axes[row, column]
            for case in cases:
                if case.case["route_id"] == route:
                    _plot_series(
                        axis,
                        case,
                        PlotSelection("after_iteration", quantity),
                        transitions=True,
                    )
            axis.set(
                title=route.upper(),
                xlabel="Native solve iteration after refinement",
                ylabel=label,
            )
            axis.grid(alpha=0.5)
            if axis.has_data():
                axis.legend()
    written = []
    if any(axis.has_data() for axis in axes.flat):
        figure.savefig(output / "refinement.svg")
        written.append("refinement.svg")
    plt.close(figure)
    return written


def _plot_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _render_parent_counts(cases: Sequence[CaseSeries], output: Path) -> list[str]:
    from matplotlib import pyplot as plt  # noqa: PLC0415 -- standalone scientific figures use the optional plotting runtime.

    routes = observed_route_ids(cases)
    figure, axes = plt.subplots(
        1,
        len(routes),
        figsize=(5.5 * len(routes), 4),
        layout="constrained",
        squeeze=False,
    )
    for axis, route in zip(axes.flat, routes, strict=True):
        for case in cases:
            if case.case["route_id"] != route:
                continue
            rows = [
                row
                for row in transition_rows(case)
                if row["after_iteration"] is not None
                and row["marked_count"] is not None
                and row["extra_refined_parent_count"] is not None
            ]
            if not rows:
                continue
            nonconformal = case.case["amr_nonconformal"]
            offset = 0.18 if nonconformal else -0.18
            color = "#f0b95b" if nonconformal else "#66b5ff"
            label = "NC(1)" if nonconformal else "Conformal"
            xs = [row["after_iteration"] + offset for row in rows]
            marked = [row["marked_count"] for row in rows]
            axis.bar(xs, marked, width=0.34, color=color, label=f"{label}: marked")
            axis.bar(
                xs,
                [row["extra_refined_parent_count"] for row in rows],
                bottom=marked,
                width=0.34,
                color=color,
                alpha=0.45,
                hatch="///",
                label=f"{label}: extra parents",
            )
        axis.set(
            title=route.upper(),
            xlabel="Native next-solve iteration",
            ylabel="Original parent cells",
        )
        axis.grid(axis="y", alpha=0.5)
        if axis.has_data():
            axis.legend(fontsize=8)
    written = []
    if any(axis.has_data() for axis in axes.flat):
        figure.savefig(output / "parent-counts.svg")
        written.append("parent-counts.svg")
    plt.close(figure)
    return written


def _render_marking(cases: Sequence[CaseSeries], output: Path) -> list[str]:
    from matplotlib import pyplot as plt  # noqa: PLC0415 -- standalone scientific figures use the optional plotting runtime.

    routes = observed_route_ids(cases)
    figure, axes = plt.subplots(
        len(routes),
        2,
        figsize=(11, 3.5 * len(routes)),
        layout="constrained",
        squeeze=False,
    )
    for row, route in enumerate(routes):
        for column, (quantity, label) in enumerate(
            (
                ("marked_fraction", "Fraction of pre-step cells marked"),
                ("indicator_coverage", "Fraction of squared indicator covered"),
            )
        ):
            axis = axes[row, column]
            for case in cases:
                if case.case["route_id"] == route:
                    _plot_series(
                        axis,
                        case,
                        PlotSelection("after_iteration", quantity),
                        transitions=True,
                    )
            observed = axis.has_data()
            if observed and quantity == "indicator_coverage":
                axis.axhline(
                    0.7, color="#bbbbbb", linestyle=":", label="Requested theta = 0.7"
                )
            axis.set(
                title=route.upper(),
                xlabel="Native next-solve iteration",
                ylabel=label,
                ylim=(0, 1),
            )
            axis.grid(alpha=0.5)
            if observed:
                axis.legend(fontsize=8)
    written = []
    if any(axis.has_data() for axis in axes.flat):
        figure.savefig(output / "marking.svg")
        written.append("marking.svg")
    plt.close(figure)
    return written


if __name__ == "__main__":
    raise SystemExit(main())
