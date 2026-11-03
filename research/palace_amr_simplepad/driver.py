"""Prepare, run, and analyze the four selected Palace AMR study cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, Sequence, cast

from .contract import (
    AMR_MAX_ITERATIONS,
    AMR_NC_LEVELS,
    AMR_THETA,
    AMR_TOLERANCE,
    CASES,
    DEFAULT_STUDY_ROOT,
    DIE_SIDE_UM,
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
    MPI_LAUNCHER,
    MPI_PROCESSES,
    OPENMP_THREADS,
    PAD_SIDE_UM,
    PDK_REVISION,
    SUBSTRATE_RELATIVE_PERMITTIVITY,
    SUBSTRATE_THICKNESS_UM,
    VACUUM_PADDING_UM,
    CaseDefinition,
)
from .provenance import capture_host_resources, capture_scgsim_provenance

if TYPE_CHECKING:
    from scgsim.geometry.source.plan import GeometryPlanSnapshot
    from scgsim.palace import ElectrostaticSim
    from scgsim.palace.execution.handoff import HandoffPlan


@dataclass(frozen=True)
class PreparationContext:
    """Collect measured source and toolchain identities for the study manifest."""

    study_dir: Path
    palace_binary: Path
    mpi_launcher: Path
    wrapper_path: Path
    source_gds_sha256: str
    source_preparation_seconds: float
    scgsim_provenance: dict[str, Any]
    host_resources: dict[str, Any]


class _PdkModule(Protocol):
    """Narrow runtime surface consumed from the pinned public PDK."""

    def activate(self) -> None: ...


class _PdkFactoryModule(Protocol):
    """Public geometry factory used to create the shared study source."""

    def square_pad_capacitor(
        self,
        *,
        pad_side_um: float,
        gap_um: float,
        die_side_um: float,
        junction_width_um: float,
    ) -> object: ...


class _PdkMaterialsModule(Protocol):
    """Public material-record provider used by GeometryPlan."""

    def get_material_records(self) -> dict[str, dict[str, Any]]: ...


class _PdkTechModule(Protocol):
    """Public single-die stack provider used by GeometryPlan."""

    def get_single_die_layer_stack(self) -> object: ...


MANIFEST_NAME = "study-manifest.json"
PROFILE_NAME = "direct-local"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a stage command and dispatch to its named workflow."""
    arguments = _parse_arguments(argv)
    if arguments.stage == "prepare":
        study_dir = _prepare_study(arguments)
        print(study_dir)
        return 0
    if arguments.stage == "run":
        return _run_study(arguments.study_dir)
    return analyze_study(arguments.study_dir)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the explicit prepare, run, and analyze command surfaces."""
    parser = argparse.ArgumentParser(
        description="Prepare and analyze the four-case SimplePad Palace AMR study."
    )
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare", help="build meshes and sealed handoffs")
    prepare.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    prepare.add_argument("--study-dir", type=Path)
    prepare.add_argument("--palace-binary", type=Path, required=True)
    prepare.add_argument("--mpi-launcher", type=Path, default=MPI_LAUNCHER)
    run = commands.add_parser("run", help="run all prepared cases sequentially")
    run.add_argument("study_dir", type=Path)
    analyze = commands.add_parser("analyze", help="resolve and export measured results")
    analyze.add_argument("study_dir", type=Path)
    return parser.parse_args(argv)


def _prepare_study(arguments: argparse.Namespace) -> Path:
    """Verify inputs, create one GeometryPlan snapshot, mesh each route once, and seal cases."""
    from gdsfactory import clear_cache  # noqa: PLC0415 -- GDS preparation runs only in the prepare stage.

    from scgsim.geometry import GeometryPlan  # noqa: PLC0415 -- GeometryPlan is needed only to create the shared source snapshot.

    orcapen_sc_pdk = cast(_PdkModule, import_module("orcapen_sc_pdk"))
    pdk_factory = cast(
        _PdkFactoryModule,
        import_module("orcapen_sc_pdk.cells.simulation.scgsim"),
    )
    pdk_materials = cast(_PdkMaterialsModule, import_module("orcapen_sc_pdk.materials"))
    pdk_tech = cast(_PdkTechModule, import_module("orcapen_sc_pdk.tech"))

    palace_binary = _required_executable(arguments.palace_binary, "Palace binary")
    mpi_launcher = _required_executable(
        arguments.mpi_launcher, "OpenMPI 5.0.10 launcher"
    )
    _require_pdk_revision(PDK_REVISION)
    study_dir = new_study_dir(arguments.study_root, arguments.study_dir)
    study_dir.mkdir(parents=True, exist_ok=False)
    wrapper_path = study_dir / "launcher" / "palace-mpi-wrapper.sh"
    wrapper_path.parent.mkdir(parents=True)
    wrapper_path.write_text(
        launcher_script(palace_binary, mpi_launcher), encoding="utf-8"
    )
    wrapper_path.chmod(0o755)

    scgsim_provenance = capture_scgsim_provenance(Path(__file__).resolve().parents[2])
    source_preparation_started = perf_counter()
    orcapen_sc_pdk.activate()
    clear_cache()
    component = pdk_factory.square_pad_capacitor(
        pad_side_um=PAD_SIDE_UM,
        gap_um=GAP_UM,
        die_side_um=DIE_SIDE_UM,
        junction_width_um=LOCATOR_WIDTH_UM,
    )
    plan = GeometryPlan(
        component=component,
        root_id="chip",
        layer_stack=pdk_tech.get_single_die_layer_stack(),
        material_records=pdk_materials.get_material_records(),
        coupon_padding_um=0.0,
    )
    plan.set_nets({"pad": ("chip/PAD",), "ground": ("chip/GROUND",)})
    plan.set_vacuum_region(padding=VACUUM_PADDING_UM)
    snapshot = plan.prepare()
    source_preparation_seconds = perf_counter() - source_preparation_started

    preparation = PreparationContext(
        study_dir=study_dir,
        palace_binary=palace_binary,
        mpi_launcher=mpi_launcher,
        wrapper_path=wrapper_path,
        source_gds_sha256=snapshot.gds_sha256,
        source_preparation_seconds=source_preparation_seconds,
        scgsim_provenance=scgsim_provenance,
        host_resources=capture_host_resources(),
    )
    manifest = _initial_manifest(preparation)
    _write_manifest(study_dir, manifest)
    prepared_cases: list[dict[str, Any]] = []
    for route_id in ("route-a", "route-b"):
        route_cases = tuple(case for case in CASES if case.route_id == route_id)
        prepared_cases.extend(
            _prepare_route(
                study_dir=study_dir,
                snapshot=snapshot,
                route_cases=route_cases,
                wrapper_path=wrapper_path,
                preparation_timings=manifest["preparation_timings"],
            )
        )
        manifest["cases"] = list(prepared_cases)
        _write_manifest(study_dir, manifest)
    manifest["status"] = "prepared"
    require_prepared_manifest(manifest)
    _write_manifest(study_dir, manifest)
    return study_dir


def _prepare_route(
    *,
    study_dir: Path,
    snapshot: GeometryPlanSnapshot,
    route_cases: tuple[CaseDefinition, ...],
    wrapper_path: Path,
    preparation_timings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one route mesh and export both independently sealed AMR arms."""
    from scgsim.geometry import ApproximationProfile  # noqa: PLC0415 -- physical profiles are needed only during preparation.
    from scgsim.palace import ElectrostaticSim  # noqa: PLC0415 -- the Palace facade is needed only during preparation.

    first = route_cases[0]
    simulation = ElectrostaticSim()
    simulation.set_plan(snapshot)
    simulation.set_output_dir(study_dir / "preparation" / first.route_id)
    simulation.set_surface_epr(
        representation=first.representation,
        specs={},
        approximation_profile=ApproximationProfile(first.approximation, "v1", {}),
    )
    simulation.add_terminal("pad", net_id="pad")
    simulation.add_ground(net_id="ground")
    simulation.set_electrostatic(
        save_fields=1,
        unassigned_conductor_policy="error",
        exterior_boundary_policy="none",
    )
    simulation.set_mesh(
        size_policy=MESH_SIZE_POLICY,
        algorithm_3d=MESH_ALGORITHM_3D,
        threads=MESH_THREADS,
        surface_threads=MESH_SURFACE_THREADS,
        geometry_order=MESH_GEOMETRY_ORDER,
    )
    mesh_started = perf_counter()
    mesh_path = simulation.mesh()
    preparation_timings["route_geometry_and_mesh_seconds"][first.route_id] = (
        perf_counter() - mesh_started
    )
    route_outputs: list[dict[str, Any]] = []
    for case in route_cases:
        config_started = perf_counter()
        _set_case_controls(simulation, case)
        config_path = simulation.write_config()
        preparation_timings["case_config_seconds"][case.case_id] = (
            perf_counter() - config_started
        )
        handoff_started = perf_counter()
        handoff = simulation.prepare_handoff(
            profile=PROFILE_NAME,
            executable=str(wrapper_path),
            resources={
                "command_style": "wrapper",
                "processes": MPI_PROCESSES,
                "threads": OPENMP_THREADS,
            },
        )
        _verify_handoff_contract(handoff)
        preparation_timings["case_handoff_package_seconds"][case.case_id] = (
            perf_counter() - handoff_started
        )
        target_run = study_dir / "runs" / case.case_id / first.route_id
        target_archive = target_run.with_suffix(".tar.gz")
        export_started = perf_counter()
        _export_handoff(
            handoff.run_dir, handoff.archive_path, target_run, target_archive
        )
        _verify_exported_inputs(config_path, mesh_path, target_run)
        preparation_timings["case_export_seconds"][case.case_id] = (
            perf_counter() - export_started
        )
        route_outputs.append(
            _case_manifest_entry(
                case=case,
                handoff=handoff,
                run_dir=target_run,
                study_dir=study_dir,
            )
        )
    return route_outputs


def _set_case_controls(simulation: ElectrostaticSim, case: CaseDefinition) -> None:
    """Set the Human-selected FEM, solver, AMR, and output controls for one arm."""
    simulation.set_numerical(
        order=FEM_ORDER,
        tolerance=LINEAR_TOLERANCE,
        max_iterations=LINEAR_MAX_ITERATIONS,
        solver_type="Default",
        preconditioner="BoomerAMG",
        ksp_type="CG",
        device="CPU",
        amr_max_passes=AMR_MAX_ITERATIONS,
        amr_nonconformal=case.nonconformal,
        amr_tolerance=AMR_TOLERANCE,
        amr_update_fraction=AMR_THETA,
        amr_nc_levels=AMR_NC_LEVELS,
        save_adapt_iterations=True,
        save_adapt_mesh=True,
        save_adapt_diagnostics=True,
        output_paraview=True,
        output_grid_function=False,
    )


def _run_study(study_dir: Path) -> int:
    """Attempt all four prepared handoffs sequentially and retain each outcome."""
    root = study_dir.expanduser().resolve()
    manifest = _read_manifest(root)
    require_prepared_manifest(manifest)
    _verify_toolchain_identities(root, manifest)
    manifest["run_host_resources"] = capture_host_resources()
    manifest["status"] = "running"
    _write_manifest(root, manifest)
    failed_cases = 0
    for case in manifest["cases"]:
        run_dir = _study_path(root, case["run_dir"])
        try:
            completed = subprocess.run(
                ["bash", str(run_dir / "run_palace.sh")], cwd=run_dir, check=False
            )
        except OSError as exc:
            case["run_status"] = "launch_error"
            case["run_exit_code"] = None
            case["run_error"] = {"type": type(exc).__name__, "message": str(exc)}
            case["receipt_present"] = _record_receipt(run_dir, case)
            failed_cases += 1
        else:
            case["run_exit_code"] = completed.returncode
            case["receipt_present"] = _record_receipt(run_dir, case)
            if completed.returncode == 0 and case["receipt_present"]:
                case["run_status"] = "returned_zero"
            elif completed.returncode == 0:
                case["run_status"] = "missing_receipt"
                failed_cases += 1
            else:
                case["run_status"] = "returned_nonzero"
                failed_cases += 1
        _write_manifest(root, manifest)
    manifest["status"] = "run_attempts_complete"
    manifest["failed_run_count"] = failed_cases
    _write_manifest(root, manifest)
    return int(failed_cases > 0)


def _record_receipt(run_dir: Path, case: dict[str, Any]) -> bool:
    """Record a returned receipt when present without assigning a success state."""
    receipt = run_dir / "metadata" / "palace_returned_run_receipt.json"
    if not receipt.is_file():
        case["receipt_sha256"] = None
        return False
    case["receipt_sha256"] = _sha256(receipt)
    return True


def analyze_study(study_dir: Path) -> int:
    """Export available results and preserve per-case analysis errors."""
    root = study_dir.expanduser().resolve()
    manifest = _read_manifest(root)
    if manifest.get("status") != "run_attempts_complete":
        raise ValueError("analyze requires all four handoff attempts to finish")
    failed_cases = 0
    for case in manifest["cases"]:
        try:
            benchmark_data, analysis_data = analyze_case(root, case)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            case["analysis_status"] = "analysis_error"
            case["analysis_error"] = {"type": type(exc).__name__, "message": str(exc)}
            failed_cases += 1
        else:
            benchmark_path = _study_path(root, case["benchmark_path"])
            amr_path = _study_path(root, case["amr_analysis_path"])
            _write_json(benchmark_path, benchmark_data)
            _write_json(amr_path, analysis_data)
            case["analysis_status"] = "exported_from_observed_run"
            case["benchmark_sha256"] = _sha256(benchmark_path)
            case["amr_analysis_sha256"] = _sha256(amr_path)
        _write_manifest(root, manifest)
    manifest["status"] = "analysis_attempts_complete"
    manifest["analysis_error_count"] = failed_cases
    _write_manifest(root, manifest)
    return int(failed_cases > 0 or manifest.get("failed_run_count", 0) > 0)


def analyze_case(
    study_dir: Path, case: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use strict resolution for returned runs and explicit inspection for failures."""
    from scgsim.palace import (  # noqa: PLC0415 -- Palace result reading is needed only during analysis.
        analyze_amr,
        inspect_run_trustworthiness,
        resolve_palace_result,
    )

    run_dir = _study_path(study_dir, case["run_dir"])
    returned = case.get("run_exit_code") == 0 and case.get("receipt_present") is True
    if returned:
        observed = resolve_palace_result(
            run_dir, expected_handoff_id=case["handoff_id"]
        )
        analysis = analyze_amr(observed, reference=None)
        benchmark = observed.show_simulation_benchmark(amr_analysis=analysis)
    else:
        observed = inspect_run_trustworthiness(run_dir)
        analysis = analyze_amr(observed, reference=None)
        benchmark = observed.show_simulation_benchmark(amr_analysis=analysis)
    return benchmark.data, analysis.to_payload()


def _verify_handoff_contract(handoff: HandoffPlan) -> None:
    """Check public handoff status, records, archive, and sealed input hashes."""
    if (
        handoff.profile != PROFILE_NAME
        or handoff.status != "prepared"
        or handoff.kind != "manual_handoff"
        or not handoff.handoff_id
    ):
        raise ValueError("Palace facade returned an unexpected handoff plan")
    for path in (
        handoff.script_path,
        handoff.metadata_path,
        handoff.run_metadata_path,
        handoff.resource_record_path,
        handoff.manifest_path,
        handoff.archive_path,
        handoff.returned_receipt_recorder_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"prepared handoff artifact is missing: {path}")

    metadata = json.loads(handoff.metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "prepared"
        or metadata.get("kind") != "manual_handoff"
        or metadata.get("manual_intent") is not True
        or metadata.get("problem") != "Electrostatic"
        or metadata.get("handoff_id") != handoff.handoff_id
    ):
        raise ValueError("prepared handoff metadata does not match its public plan")
    hashes = metadata.get("hashes")
    if not isinstance(hashes, list):
        raise ValueError("prepared handoff does not declare its input hashes")
    recorded: dict[str, str] = {}
    for item in hashes:
        if not isinstance(item, dict):
            raise TypeError("prepared handoff input hash entry is malformed")
        relative = item.get("path")
        digest = item.get("sha256")
        if isinstance(relative, str) and isinstance(digest, str):
            recorded[relative] = digest
    for name in ("config.json", "palace.msh"):
        if recorded.get(name) != _sha256(handoff.run_dir / name):
            raise ValueError(f"prepared handoff {name} hash does not match its input")


def _verify_exported_inputs(
    source_config: Path, source_mesh: Path, target_run: Path
) -> None:
    """Confirm the exported case retains the route mesh and prepared config bytes."""
    for source in (source_config, source_mesh):
        exported = target_run / source.name
        if _sha256(source) != _sha256(exported):
            raise ValueError(f"export changed prepared input bytes: {source.name}")


def _export_handoff(
    source_run: Path, source_archive: Path, target_run: Path, target_archive: Path
) -> None:
    """Copy the complete prepared directory and sibling archive without relabeling."""
    if target_run.exists() or target_archive.exists():
        raise FileExistsError(f"case output already exists: {target_run.parent}")
    target_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run, target_run)
    shutil.copy2(source_archive, target_archive)


def _case_manifest_entry(
    *,
    case: CaseDefinition,
    handoff: HandoffPlan,
    run_dir: Path,
    study_dir: Path,
) -> dict[str, Any]:
    """Record each case's independent identity and measured input digests."""
    archive_path = run_dir.with_suffix(".tar.gz")
    return {
        "case_id": case.case_id,
        "route_id": case.route_id,
        "representation": case.representation,
        "approximation": case.approximation,
        "amr_nonconformal": case.nonconformal,
        "handoff_id": handoff.handoff_id,
        "run_dir": run_dir.relative_to(study_dir).as_posix(),
        "archive_path": archive_path.relative_to(study_dir).as_posix(),
        "config_sha256": _sha256(run_dir / "config.json"),
        "mesh_sha256": _sha256(run_dir / "palace.msh"),
        "archive_sha256": _sha256(archive_path),
        "benchmark_path": f"analysis/{case.case_id}-benchmark.json",
        "amr_analysis_path": f"analysis/{case.case_id}-amr.json",
    }


def _initial_manifest(preparation: PreparationContext) -> dict[str, Any]:
    """Describe the selected study inputs and launcher identities before cases."""
    study_dir = preparation.study_dir
    palace_binary = preparation.palace_binary
    mpi_launcher = preparation.mpi_launcher
    wrapper_path = preparation.wrapper_path
    return {
        "schema": "scgsim.palace_amr_simplepad.study.v1",
        "status": "preparing",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pdk_revision": PDK_REVISION,
        "scgsim_provenance": preparation.scgsim_provenance,
        "preparation_host_resources": preparation.host_resources,
        "preparation_timings": {
            "scope": (
                "Client-side source snapshot, route geometry and mesh preparation, "
                "case config, handoff packaging, and export wall times; excludes "
                "native Palace dry-run and solve times."
            ),
            "common_layout_and_geometry_plan_snapshot_seconds": preparation.source_preparation_seconds,
            "route_geometry_and_mesh_seconds": {},
            "case_config_seconds": {},
            "case_handoff_package_seconds": {},
            "case_export_seconds": {},
        },
        "source": {
            "factory": "orcapen_sc_pdk.cells.simulation.scgsim.square_pad_capacitor",
            "gds_sha256": preparation.source_gds_sha256,
            "pad_side_um": PAD_SIDE_UM,
            "gap_um": GAP_UM,
            "die_side_um": DIE_SIDE_UM,
            "locator_width_um": LOCATOR_WIDTH_UM,
            "substrate_thickness_um": SUBSTRATE_THICKNESS_UM,
            "substrate_relative_permittivity": SUBSTRATE_RELATIVE_PERMITTIVITY,
            "film_thickness_um": FILM_THICKNESS_UM,
            "vacuum_padding_um": list(VACUUM_PADDING_UM),
            "nets": {"pad": ["chip/PAD"], "ground": ["chip/GROUND"]},
        },
        "mesh": {
            "size_policy": MESH_SIZE_POLICY,
            "algorithm_3d": MESH_ALGORITHM_3D,
            "geometry_order": MESH_GEOMETRY_ORDER,
            "threads": MESH_THREADS,
            "surface_threads": MESH_SURFACE_THREADS,
            "route_meshes": 2,
        },
        "solver": {
            "fem_order": FEM_ORDER,
            "ksp_type": "CG",
            "preconditioner": "BoomerAMG",
            "linear_tolerance": LINEAR_TOLERANCE,
            "linear_max_iterations": LINEAR_MAX_ITERATIONS,
            "amr_max_iterations": AMR_MAX_ITERATIONS,
            "amr_tolerance": AMR_TOLERANCE,
            "amr_theta": AMR_THETA,
            "amr_nc_levels": AMR_NC_LEVELS,
            "save_adapt_iterations": True,
            "save_adapt_mesh": True,
            "save_adapt_diagnostics": True,
            "save_fields": 1,
            "output_paraview": True,
            "output_grid_function": False,
            "unassigned_conductor_policy": "error",
            "exterior_boundary_policy": "none",
            "openmp_threads": OPENMP_THREADS,
            "mpi_processes": MPI_PROCESSES,
        },
        "toolchain": {
            "palace_binary": str(palace_binary),
            "palace_binary_sha256": _sha256(palace_binary),
            "mpi_launcher": str(mpi_launcher),
            "mpi_launcher_sha256": _sha256(mpi_launcher),
            "wrapper": str(wrapper_path.relative_to(study_dir)),
            "wrapper_sha256": _sha256(wrapper_path),
            "mpi_arguments": [
                "--use-hwthread-cpus",
                "--map-by",
                "slot:PE=2",
                "--bind-to",
                "hwthread",
                "--report-bindings",
            ],
        },
        "cases": [],
    }


def launcher_script(palace_binary: Path, mpi_launcher: Path) -> str:
    """Create a serial compatibility-probe and fixed 30-by-2 MPI launcher."""
    binary = shlex.quote(str(palace_binary))
    launcher = shlex.quote(str(mpi_launcher))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'if [[ "${1:-}" == "-serial" ]]; then',
        "    shift",
        f'    exec {binary} "$@"',
        "fi",
        f'if [[ "$#" -ne 5 || "$1" != "-np" || "$2" != "{MPI_PROCESSES}" || "$3" != "-nt" || "$4" != "{OPENMP_THREADS}" ]]; then',
        "    printf '%s\\n' 'unexpected Palace wrapper arguments' >&2",
        "    exit 64",
        "fi",
        f"if [[ ! -x {binary} ]]; then",
        "    printf '%s\\n' 'Palace binary is no longer executable' >&2",
        "    exit 126",
        "fi",
        f"export OMP_NUM_THREADS={OPENMP_THREADS}",
        "export OMP_PLACES=threads",
        "export OMP_PROC_BIND=close",
        f'exec {launcher} --use-hwthread-cpus --map-by slot:PE=2 --bind-to hwthread --report-bindings -np {MPI_PROCESSES} {binary} "$5"',
    ]
    return "\n".join(lines) + "\n"


def new_study_dir(study_root: Path, explicit_study_dir: Path | None) -> Path:
    """Choose a fresh study directory under the caller-selected root."""
    if explicit_study_dir is not None:
        return explicit_study_dir.expanduser().resolve()
    root = study_root.expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{stamp}-{uuid.uuid4().hex[:8]}"


def _required_executable(path: Path, label: str) -> Path:
    """Resolve a caller-selected executable and require it to be executable."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"{label} is not an executable file: {resolved}")
    return resolved


def _require_pdk_revision(expected_revision: str) -> None:
    """Require the installed public PDK distribution to identify the pinned commit."""
    try:
        distribution = importlib.metadata.distribution("orcapen-sc-pdk")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned orcapen-sc-pdk before prepare.") from exc
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError(
            "orcapen-sc-pdk has no VCS source identity in direct_url.json"
        )
    identity = json.loads(direct_url)
    actual = identity.get("vcs_info", {}).get("commit_id")
    if actual != expected_revision:
        raise RuntimeError(
            "orcapen-sc-pdk revision must "
            f"be {expected_revision}; found {actual or 'no VCS commit identity'}"
        )


def _verify_toolchain_identities(study_dir: Path, manifest: dict[str, Any]) -> None:
    """Require solver and launcher bytes to match the prepared study identities."""
    toolchain = manifest["toolchain"]
    for key, digest_key in (
        ("palace_binary", "palace_binary_sha256"),
        ("mpi_launcher", "mpi_launcher_sha256"),
    ):
        path = Path(toolchain[key])
        if _sha256(path) != toolchain[digest_key]:
            raise ValueError(f"prepared toolchain identity changed: {path}")
    wrapper = _study_path(study_dir, toolchain["wrapper"])
    if _sha256(wrapper) != toolchain["wrapper_sha256"]:
        raise ValueError(f"prepared MPI wrapper identity changed: {wrapper}")


def require_prepared_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed unless exactly the four independently sealed cases are present."""
    if manifest.get("status") != "prepared":
        raise ValueError("run requires a completely prepared study")
    cases = manifest.get("cases", [])
    actual_ids = [case.get("case_id") for case in cases]
    if actual_ids != [case.case_id for case in CASES]:
        raise ValueError("prepared study does not contain the ordered four-case set")
    handoff_ids = [case.get("handoff_id") for case in cases]
    if any(not isinstance(identity, str) or not identity for identity in handoff_ids):
        raise ValueError("prepared cases must retain their handoff identities")
    if len(set(handoff_ids)) != len(CASES):
        raise ValueError("each prepared case must have an independent handoff identity")


def _study_path(study_dir: Path, relative_path: str) -> Path:
    """Resolve one manifest path while keeping it inside the selected study."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("study manifest paths must be non-empty relative paths")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("study manifest paths must be relative")
    root = study_dir.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("study manifest path escapes the selected study directory")
    return resolved


def _read_manifest(study_dir: Path) -> dict[str, Any]:
    """Read the driver-owned study manifest from its selected directory."""
    path = study_dir / MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "scgsim.palace_amr_simplepad.study.v1"
    ):
        raise ValueError(f"unsupported study manifest: {path}")
    return value


def _write_manifest(study_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the study's measured preparation and run identities."""
    _write_json(study_dir / MANIFEST_NAME, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic JSON record via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one existing file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
