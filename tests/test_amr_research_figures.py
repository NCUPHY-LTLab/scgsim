"""Keep scientific exports faithful to recorded zero, missing and failed data."""

import csv
import hashlib
import json

import pytest

from research.palace_amr_simplepad.figures import (
    CaseSeries,
    case_summary,
    load_case_series,
    observation_rows,
    observed_route_ids,
    write_observations,
    write_transitions,
)


def test_observed_zero_is_retained_without_inventing_an_adjacent_change(tmp_path):
    """A zero observation remains data, while the initial comparison stays absent."""
    case = CaseSeries(
        {"case_id": "a-c", "route_id": "route-a"},
        {
            "passes": [
                {
                    "native_iteration": 1,
                    "capacitance_matrix_f": [[0.0]],
                    "native_zz_norm": {"availability": "available", "value": 0.0},
                    "linear_solve_state": "unconverged",
                }
            ],
            "transitions": [],
        },
        {},
    )
    row = observation_rows(case)[0]
    assert row["capacitance_f"] == row["native_zz_norm"] == 0.0
    assert row["linear_solve_state"] == "unconverged"
    assert row["relative_capacitance_change"] is None
    assert row["true_dofs"] is None
    destination = tmp_path / "observations.csv"
    write_observations([case], destination)
    exported = next(csv.DictReader(destination.open()))
    assert exported["capacitance_f"] == "0.0"
    assert exported["relative_capacitance_change"] == ""
    assert exported["true_dofs"] == ""


def test_refinement_cost_survives_when_the_next_solve_has_no_matrix(tmp_path):
    """Export actual parents even when no solved pass can receive the transition."""
    case = CaseSeries(
        {"case_id": "b-nc", "route_id": "route-b"},
        {
            "passes": [],
            "transitions": [
                {
                    "before_iteration": 2,
                    "after_iteration": 3,
                    "availability": "available",
                    "reason": None,
                    "observations": {
                        "marked_count": 10,
                        "refined_parent_count": 30,
                        "extra_refined_parent_count": 20,
                        "cells_before": 100,
                        "cells_after": 200,
                    },
                    "closure_amplification": {
                        "availability": "available",
                        "value": 3.0,
                    },
                }
            ],
        },
        {},
    )
    destination = tmp_path / "transitions.csv"
    write_transitions([case], destination)
    row = next(csv.DictReader(destination.open()))
    assert row["after_iteration"] == "3"
    assert row["marked_count"] == "10"
    assert row["refined_parent_count"] == "30"
    assert row["closure_factor"] == "3.0"
    assert row["indicator_coverage"] == ""


def test_comparison_rejects_changed_export_bytes(tmp_path):
    """A source file changed after export cannot become a trusted report input."""
    analysis = tmp_path / "analysis.json"
    analysis.write_text('{"passes": [], "transitions": [], "reference": null}')
    digest = hashlib.sha256(analysis.read_bytes()).hexdigest()
    manifest = {
        "status": "analysis_attempts_complete",
        "cases": [
            {
                "case_id": "a-c",
                "amr_analysis_path": "analysis.json",
                "amr_analysis_sha256": digest,
            }
        ],
    }
    (tmp_path / "study-manifest.json").write_text(json.dumps(manifest))
    analysis.write_text('{"passes": [], "transitions": [], "reference": "changed"}')
    with pytest.raises(ValueError, match="digest mismatch"):
        load_case_series(tmp_path)


def test_unprepared_route_has_status_without_invented_exports_or_plot_facets(tmp_path):
    """An explicitly skipped route needs no result files and adds no empty facet."""
    skipped = {
        "case_id": "a-c",
        "route_id": "route-a",
        "preparation_status": "failed",
        "run_status": "not_run_preparation_failed",
        "analysis_status": "not_run_preparation_failed",
        "run_exit_code": None,
    }
    manifest = {"status": "analysis_attempts_complete", "cases": [skipped]}
    (tmp_path / "study-manifest.json").write_text(json.dumps(manifest))
    cases = load_case_series(tmp_path)
    assert observation_rows(cases[0]) == []
    assert case_summary(cases[0])["last_observation"] is None
    assert case_summary(cases[0])["run_status"] == "not_run_preparation_failed"
    assert observed_route_ids(cases) == ()
    observed = CaseSeries(
        {"case_id": "b-c", "route_id": "route-b"},
        {"passes": [{"native_iteration": 1}], "transitions": []},
        {},
    )
    assert observed_route_ids((*cases, observed)) == ("route-b",)


def test_recovery_warning_and_native_cost_identity_remain_separate():
    """A converged potential solve can coexist with an unconverged recovery solve."""
    case = CaseSeries(
        {"case_id": "b-c", "route_id": "route-b"},
        {
            "passes": [
                {
                    "native_iteration": 4,
                    "linear_solve_state": "converged",
                    "recovery_solve_state": "unconverged",
                    "terminal_solve_observation": {
                        "terminal_statistics": [
                            {
                                "main": {"iterations": 77, "final_residual": 2e-5},
                                "recovery": {
                                    "iterations": 10000,
                                    "final_residual": 8e-5,
                                },
                            }
                        ]
                    },
                }
            ]
        },
        {"pass_cost_records": [{"native_iteration": 3, "elapsed_pass_s": 7.0}]},
    )
    row = observation_rows(case)[0]
    assert row["linear_solve_state"] == "converged"
    assert row["recovery_solve_state"] == "unconverged"
    assert row["main_iterations"] == 77
    assert row["recovery_iterations"] == 10000
    assert row["main_final_residual"] == 2e-5
    assert row["recovery_final_residual"] == 8e-5
    assert row["elapsed_pass_s"] is None
