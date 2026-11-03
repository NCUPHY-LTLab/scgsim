"""Pure contract checks for the four-case Palace AMR research driver."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from research.palace_amr_simplepad.contract import (
    AMR_MAX_ITERATIONS,
    AMR_NC_LEVELS,
    AMR_THETA,
    AMR_TOLERANCE,
    CASES,
    FEM_ORDER,
    FILM_THICKNESS_UM,
    GAP_UM,
    LINEAR_MAX_ITERATIONS,
    LINEAR_TOLERANCE,
    LOCATOR_WIDTH_UM,
    MESH_ALGORITHM_3D,
    MESH_GEOMETRY_ORDER,
    MESH_SIZE_POLICY,
    MESH_SURFACE_THREADS,
    MESH_THREADS,
    MPI_PROCESSES,
    OPENMP_THREADS,
    PAD_SIDE_UM,
    PDK_REVISION,
    SUBSTRATE_RELATIVE_PERMITTIVITY,
    SUBSTRATE_THICKNESS_UM,
    VACUUM_PADDING_UM,
)
from research.palace_amr_simplepad.driver import (
    analyze_study,
    launcher_script,
    new_study_dir,
    require_prepared_manifest,
)
from research.palace_amr_simplepad.provenance import (
    capture_host_resources,
    capture_scgsim_provenance,
)


def test_human_selected_source_mesh_and_solver_values_are_preserved():
    """Keep the Human-selected source dimensions and four-case solve controls exact."""
    assert PDK_REVISION == "c75a1d4d2ee07e86a3b5d9b491609e5a1a5b850c"
    assert (PAD_SIDE_UM, GAP_UM, LOCATOR_WIDTH_UM) == (200.0, 20.0, 2.0)
    assert (SUBSTRATE_THICKNESS_UM, SUBSTRATE_RELATIVE_PERMITTIVITY) == (500.0, 11.9)
    assert FILM_THICKNESS_UM == 0.1
    assert VACUUM_PADDING_UM == (250.0, 250.0, 1000.0)
    assert (MESH_SIZE_POLICY, MESH_ALGORITHM_3D) == ("gmsh_default", "Delaunay")
    assert (MESH_GEOMETRY_ORDER, MESH_THREADS, MESH_SURFACE_THREADS) == (1, 1, 1)
    assert (FEM_ORDER, LINEAR_TOLERANCE, LINEAR_MAX_ITERATIONS) == (2, 1e-6, 200)
    assert (AMR_MAX_ITERATIONS, AMR_TOLERANCE, AMR_THETA, AMR_NC_LEVELS) == (
        20,
        0.01,
        0.7,
        1,
    )
    assert (MPI_PROCESSES, OPENMP_THREADS) == (30, 2)


def test_case_table_pairs_refinement_modes_with_each_route():
    """Keep exactly one conformal and nonconformal arm for each representation."""
    assert [
        (case.case_id, case.route_id, case.representation, case.nonconformal)
        for case in CASES
    ] == [
        ("a-c", "route-a", "A", False),
        ("a-nc", "route-a", "A", True),
        ("b-c", "route-b", "B", False),
        ("b-nc", "route-b", "B", True),
    ]


@pytest.mark.parametrize(
    "args",
    [
        {"status": "preparing", "cases": []},
        {"status": "prepared", "cases": [{"case_id": "a-c"}]},
    ],
)
def test_run_requires_all_four_prepared_case_identities(args):
    """Reject a partial or unfinished study before starting any handoff."""
    with pytest.raises(ValueError):
        require_prepared_manifest(args)


def test_run_rejects_reused_handoff_identity():
    """Require one independently generated handoff ID per prepared case."""
    with pytest.raises(ValueError, match="independent handoff identity"):
        require_prepared_manifest(
            {
                "status": "prepared",
                "cases": [
                    {"case_id": case.case_id, "handoff_id": "shared"} for case in CASES
                ],
            }
        )


def test_prepared_manifest_accepts_exact_ordered_four_case_set():
    """Accept the complete prepared case set without adding outcome criteria."""
    require_prepared_manifest(
        {
            "status": "prepared",
            "cases": [
                {"case_id": case.case_id, "handoff_id": f"id-{case.case_id}"}
                for case in CASES
            ],
        }
    )


def test_launcher_executes_serial_probe_and_fixed_mpi_binding_with_spaced_paths(
    tmp_path,
):
    """Check wrapper argv with test executables; this does not run Palace or MPI."""
    binary = tmp_path / "Palace test binary"
    mpi_launcher = tmp_path / "OpenMPI test launcher"
    wrapper = tmp_path / "wrapper with spaces.sh"
    config = tmp_path / "case config.json"
    palace_capture = tmp_path / "Palace received argv"
    mpi_capture = tmp_path / "MPI received argv"
    binary.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$PALACE_CAPTURE"
""",
        encoding="utf-8",
    )
    mpi_launcher.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$MPI_CAPTURE"
shift 8
exec "$@"
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    mpi_launcher.chmod(0o755)
    wrapper.write_text(launcher_script(binary, mpi_launcher), encoding="utf-8")
    wrapper.chmod(0o755)
    environment = os.environ | {
        "PALACE_CAPTURE": str(palace_capture),
        "MPI_CAPTURE": str(mpi_capture),
    }

    subprocess.run([str(wrapper), "-serial", "--version"], env=environment, check=True)
    assert palace_capture.read_text(encoding="utf-8").splitlines() == ["--version"]

    subprocess.run(
        [str(wrapper), "-np", "30", "-nt", "2", str(config)],
        env=environment,
        check=True,
    )
    assert mpi_capture.read_text(encoding="utf-8").splitlines() == [
        "--use-hwthread-cpus",
        "--map-by",
        "slot:PE=2",
        "--bind-to",
        "hwthread",
        "--report-bindings",
        "-np",
        "30",
        str(binary),
        str(config),
    ]
    assert palace_capture.read_text(encoding="utf-8").splitlines() == [str(config)]


def test_default_study_directory_is_unique_and_under_selected_root(tmp_path):
    """Allocate each default study below the selected data root with a unique name."""
    first = new_study_dir(tmp_path, None)
    second = new_study_dir(tmp_path, None)
    assert first.parent == tmp_path.resolve()
    assert second.parent == tmp_path.resolve()
    assert first != second


def test_failed_case_exports_partial_amr_and_benchmark_without_resolving(tmp_path):
    """Export observed incomplete cases while retaining their failed run states."""
    study_dir = tmp_path / "study"
    case_entries = []
    for case_id, route in (("a-c", "A"), ("a-nc", "A"), ("b-c", "B"), ("b-nc", "B")):
        run_dir = (
            study_dir / "runs" / case_id / ("route-a" if route == "A" else "route-b")
        )
        metadata_dir = run_dir / "metadata"
        metadata_dir.mkdir(parents=True)
        handoff_id = f"partial-{case_id}"
        (metadata_dir / "palace_handoff_metadata.json").write_text(
            json.dumps(
                {
                    "problem": "Electrostatic",
                    "route": route,
                    "profile": "direct-local",
                    "handoff_id": handoff_id,
                    "status": "prepared",
                    "kind": "manual_handoff",
                    "manual_intent": True,
                }
            ),
            encoding="utf-8",
        )
        case_entries.append(
            {
                "case_id": case_id,
                "route_id": "route-a" if route == "A" else "route-b",
                "handoff_id": handoff_id,
                "run_dir": run_dir.relative_to(study_dir).as_posix(),
                "run_status": "returned_nonzero",
                "run_exit_code": 137,
                "receipt_present": False,
                "benchmark_path": f"analysis/{case_id}-benchmark.json",
                "amr_analysis_path": f"analysis/{case_id}-amr.json",
            }
        )
    (study_dir / "study-manifest.json").write_text(
        json.dumps(
            {
                "schema": "scgsim.palace_amr_simplepad.study.v1",
                "status": "run_attempts_complete",
                "failed_run_count": len(case_entries),
                "cases": case_entries,
            }
        ),
        encoding="utf-8",
    )

    assert analyze_study(study_dir) == 1
    manifest = json.loads((study_dir / "study-manifest.json").read_text())
    assert manifest["status"] == "analysis_attempts_complete"
    assert [case["run_status"] for case in manifest["cases"]] == [
        "returned_nonzero"
    ] * 4
    for case in manifest["cases"]:
        analysis = json.loads((study_dir / case["amr_analysis_path"]).read_text())
        benchmark = json.loads((study_dir / case["benchmark_path"]).read_text())
        assert analysis["run_status"] is not None
        assert analysis["integrity"] is not None
        assert analysis["passes"] == []
        assert analysis["transitions"] == []
        assert analysis["reference"] is None
        assert benchmark["amr_analysis"] == analysis
        assert benchmark["performance_metadata"]["completeness"] == "partial"
        assert case["analysis_status"] == "exported_from_observed_run"


def test_scgsim_provenance_reports_unavailable_fields_without_guessing():
    """Record repository, install, and VCS identities or explicit unavailable reasons."""
    repository_root = Path(__file__).resolve().parents[1]
    provenance = capture_scgsim_provenance(repository_root)
    repository = provenance["repository"]
    installed = provenance["installed_distribution"]
    assert set(repository) == {"head", "tree", "working_tree_dirty"}
    for observation in repository.values():
        assert set(observation) == {"value", "reason"}
        assert observation["value"] is not None or observation["reason"]
    assert isinstance(installed["version"], str) or installed["version"] is None
    if installed["vcs_commit_id"] is None:
        assert installed["vcs_commit_reason"]
    assert str(repository_root) not in json.dumps(provenance)


def test_host_resources_record_cpu_affinity_without_eligibility_rules():
    """Capture available stdlib host facts without deriving hardware gates."""
    resources = capture_host_resources()
    assert set(resources) == {
        "logical_cpu_count",
        "process_affinity_cpu_ids",
        "machine_architecture",
        "processor_description",
    }
    logical_cpu_count = resources["logical_cpu_count"]
    if logical_cpu_count["value"] is None:
        assert logical_cpu_count["reason"]
    else:
        assert isinstance(logical_cpu_count["value"], int)
    affinity = resources["process_affinity_cpu_ids"]
    if affinity["value"] is not None:
        assert affinity["value"] == sorted(set(affinity["value"]))
    else:
        assert affinity["reason"]
