"""Manual-only Palace handoff adapted from gsim portable fixes @ 8cf5fa79fa3abb176940dbfc520ff34a44f4770e."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

from ..mesh.build import MeshBuildResult
from ..mesh.manifest import SGB_RUNTIME_AUTHORITY
from ..preparation.transforms import SCHEMA_VERSION
from .archive import (
    run_archive_arcname,
    sibling_run_archive_path,
    sibling_run_archive_relative,
)
from .script import positive_int, render_script, single_line

HandoffProfile = Literal["direct-local", "slurm-single-node", "slurm-multi-node"]
_PROFILES = {"direct-local", "slurm-single-node", "slurm-multi-node"}
PORTABLE_HANDOFF_SHA = "8cf5fa79fa3abb176940dbfc520ff34a44f4770e"


@dataclass(frozen=True)
class HandoffPlan:
    """Prepared, manual-only files. This library never executes or submits them."""

    profile: HandoffProfile
    status: Literal["prepared"]
    kind: Literal["manual_handoff"]
    handoff_id: str
    run_dir: Path
    script_path: Path
    metadata_path: Path
    run_metadata_path: Path
    resource_record_path: Path
    manifest_path: Path
    archive_path: Path
    returned_receipt_path: Path
    returned_receipt_recorder_path: Path


@dataclass(frozen=True)
class _PreparedArtifacts:
    route: str
    thin_film: dict[str, Any] | None
    hashes: list[dict[str, str | int]]
    source_revisions: dict[str, Any]
    palace_identity: dict[str, Any]
    execution_identity: dict[str, Any]
    handoff_id: str


def prepare_handoff(  # noqa: PLR0913 -- established public handoff API
    *,
    profile: str,
    mesh_result: MeshBuildResult,
    config_path: Path,
    executable: str,
    resources: Mapping[str, Any] | None = None,
    setup_commands: Sequence[str] = (),
    petsc_options: Sequence[str] = (),
    problem: Literal["Electrostatic", "Eigenmode"] = "Electrostatic",
) -> HandoffPlan:
    """Create only portable handoff artifacts; caller owns executable and resources."""
    if profile not in _PROFILES:
        raise ValueError(f"Unsupported handoff profile {profile!r}.")
    if not config_path.is_file() or not mesh_result.mesh_path.is_file():
        raise FileNotFoundError("handoff requires existing config.json and palace.msh.")

    executable = single_line(executable, "executable")
    setup = tuple(single_line(command, "setup command") for command in setup_commands)
    petsc = tuple(single_line(option, "PETSc option") for option in petsc_options)
    resource_map = _validate_resources(profile, resources)
    run_dir = mesh_result.output_dir
    for directory in (
        run_dir / "logs",
        run_dir / "results" / "palace",
        run_dir / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    script_path = run_dir / (
        "run_palace.sh" if profile == "direct-local" else "run_palace.sbatch"
    )
    (
        run_dir
        / ("run_palace.sbatch" if profile == "direct-local" else "run_palace.sh")
    ).unlink(missing_ok=True)

    metadata_path = run_dir / "metadata" / "palace_handoff_metadata.json"
    run_metadata_path = run_dir / "metadata" / "palace_run_metadata.json"
    resource_record_path = run_dir / "metadata" / "palace_resource_record.json"
    manifest_path = run_dir / "metadata" / "palace_handoff_archive_manifest.json"
    archive_path = sibling_run_archive_path(run_dir)
    archive_relative = sibling_run_archive_relative(run_dir)
    returned_receipt_path = run_dir / "metadata" / "palace_returned_run_receipt.json"
    returned_receipt_recorder_path = run_dir / "metadata" / "record_palace_return.py"
    index_map_path = run_dir / "metadata" / "palace_index_map.json"

    _clear_previous_returned_outputs(
        run_dir=run_dir,
        returned_receipt_path=returned_receipt_path,
    )
    artifacts = _prepare_artifacts(
        profile=profile,
        mesh_result=mesh_result,
        config_path=config_path,
        executable=executable,
        resources=resource_map,
        setup=setup,
        petsc=petsc,
        problem=problem,
        run_dir=run_dir,
        index_map_path=index_map_path,
        script_path=script_path,
        recorder_path=returned_receipt_recorder_path,
    )
    _write_handoff_metadata(
        artifacts=artifacts,
        profile=profile,
        problem=problem,
        executable=executable,
        resources=resource_map,
        setup=setup,
        petsc=petsc,
        run_dir=run_dir,
        metadata_path=metadata_path,
        run_metadata_path=run_metadata_path,
        script_path=script_path,
        config_path=config_path,
        mesh_path=mesh_result.mesh_path,
        returned_receipt_path=returned_receipt_path,
        recorder_path=returned_receipt_recorder_path,
    )

    resource_record_path.unlink(missing_ok=True)
    archive_path.unlink(missing_ok=True)

    _create_archive(
        run_dir=run_dir,
        archive_path=archive_path,
        archive_relative=archive_relative,
        manifest_path=manifest_path,
        resource_record_path=resource_record_path,
        artifacts=artifacts,
        profile=profile,
        problem=problem,
        resources=resource_map,
    )

    return HandoffPlan(
        cast(HandoffProfile, profile),
        "prepared",
        "manual_handoff",
        artifacts.handoff_id,
        run_dir,
        script_path,
        metadata_path,
        run_metadata_path,
        resource_record_path,
        manifest_path,
        archive_path,
        returned_receipt_path,
        returned_receipt_recorder_path,
    )


def _prepare_artifacts(  # noqa: PLR0913 -- preparation inputs remain explicit
    *,
    profile: str,
    mesh_result: MeshBuildResult,
    config_path: Path,
    executable: str,
    resources: Mapping[str, Any],
    setup: Sequence[str],
    petsc: Sequence[str],
    problem: str,
    run_dir: Path,
    index_map_path: Path,
    script_path: Path,
    recorder_path: Path,
) -> _PreparedArtifacts:
    """Bind inputs and prepare the standalone script and recorder before metadata."""
    if not index_map_path.is_file():
        raise FileNotFoundError("handoff requires metadata/palace_index_map.json.")

    route = _route(mesh_result)
    thin_film = _route_a_thin_film(mesh_result, route)
    hashes = _hashes(
        run_dir,
        (
            mesh_result.mesh_path,
            config_path,
            mesh_result.mesh_manifest_path,
            index_map_path,
        ),
    )
    source_revisions = _identity()
    palace_identity = {
        "config_schema": SCHEMA_VERSION,
        "config_compatibility_probe": "native_dry_run",
        "config_compatibility_semantics": "required_before_solve",
        "runtime_identity_semantics": "observed_provenance_only",
    }

    if recorder_path.is_file():
        recorder_path.unlink()
    shutil.copy2(Path(__file__).with_name("returned_receipt.py"), recorder_path)
    recorder_path.chmod(0o755)
    script_path.write_text(
        render_script(
            profile,
            executable,
            resources,
            setup,
            petsc,
            returned_receipt_recorder_name=recorder_path.name,
            palace_identity=palace_identity,
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | 0o755)

    execution_identity = _execution_identity(
        profile=profile,
        resources=resources,
        executable=executable,
        setup=setup,
        petsc=petsc,
        script_path=script_path,
        recorder_path=recorder_path,
    )
    handoff_id = _compute_handoff_id(
        profile=profile,
        route=route,
        problem=problem,
        source_revisions=source_revisions,
        palace_identity=palace_identity,
        hashes=hashes,
        execution_identity=execution_identity,
        route_a_thin_film=thin_film,
    )
    return _PreparedArtifacts(
        route,
        thin_film,
        hashes,
        source_revisions,
        palace_identity,
        execution_identity,
        handoff_id,
    )


def _write_handoff_metadata(  # noqa: PLR0913 -- fields map to separate JSON records
    *,
    artifacts: _PreparedArtifacts,
    profile: str,
    problem: str,
    executable: str,
    resources: Mapping[str, Any],
    setup: Sequence[str],
    petsc: Sequence[str],
    run_dir: Path,
    metadata_path: Path,
    run_metadata_path: Path,
    script_path: Path,
    config_path: Path,
    mesh_path: Path,
    returned_receipt_path: Path,
    recorder_path: Path,
) -> None:
    """Write the prepared and not-run records after script identities are known."""
    thin_film = artifacts.thin_film
    handoff_id = artifacts.handoff_id
    returned_receipt_relative = returned_receipt_path.relative_to(run_dir).as_posix()
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "status": "prepared",
            "kind": "manual_handoff",
            "manual_intent": True,
            "profile": profile,
            "route": artifacts.route,
            **({"route_a_thin_film": thin_film} if thin_film is not None else {}),
            "problem": problem,
            "handoff_id": handoff_id,
            "returned_receipt_path": returned_receipt_relative,
            "returned_receipt_recorder": recorder_path.relative_to(run_dir).as_posix(),
            "palace_identity": artifacts.palace_identity,
            "source_revisions": artifacts.source_revisions,
            "execution_identity": artifacts.execution_identity,
            "command": _redacted_command(profile, executable, resources),
            "setup_commands": list(setup),
            "petsc_options": list(petsc),
            "requested_resources": resources,
            "resolved_resources": resources,
            "script": script_path.name,
            "config": config_path.name,
            "mesh": mesh_path.name,
            "hashes": artifacts.hashes,
        },
    )
    _write_json(
        run_metadata_path,
        {
            "schema_version": 1,
            "status": "not_run",
            "kind": "manual_handoff",
            "route": artifacts.route,
            **({"route_a_thin_film": thin_film} if thin_film is not None else {}),
            "problem": problem,
            "handoff_id": handoff_id,
            "returned_receipt_path": returned_receipt_relative,
            "palace_identity": artifacts.palace_identity,
            "source_revisions": artifacts.source_revisions,
            "execution_identity": artifacts.execution_identity,
            "hashes": artifacts.hashes,
        },
    )


def _create_archive(  # noqa: PLR0913 -- archive paths and identities stay explicit
    *,
    run_dir: Path,
    archive_path: Path,
    archive_relative: str,
    manifest_path: Path,
    resource_record_path: Path,
    artifacts: _PreparedArtifacts,
    profile: str,
    problem: str,
    resources: Mapping[str, Any],
) -> None:
    """Create the archive and record its digest after all archived inputs settle."""
    excluded = {resource_record_path}
    members = _members(run_dir, exclude=excluded | {manifest_path})
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "archive": archive_relative,
            "input_hashes": artifacts.hashes,
            "members": members,
            "archive_members_excluded_from_self_listing": [
                manifest_path.relative_to(run_dir).as_posix()
            ],
            "archive_members_excluded": [
                resource_record_path.relative_to(run_dir).as_posix()
            ],
        },
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path in excluded:
                continue
            archive.add(
                path,
                arcname=run_archive_arcname(run_dir, path),
                recursive=False,
            )
    _write_json(
        resource_record_path,
        {
            "schema_version": 1,
            "status": "not_submitted",
            "kind": "manual_handoff",
            "profile": profile,
            "route": artifacts.route,
            **(
                {"route_a_thin_film": artifacts.thin_film}
                if artifacts.thin_film is not None
                else {}
            ),
            "problem": problem,
            "handoff_id": artifacts.handoff_id,
            "source_revisions": artifacts.source_revisions,
            "palace_identity": artifacts.palace_identity,
            "execution_identity": artifacts.execution_identity,
            "requested_resources": resources,
            "resolved_resources": resources,
            "archive": {
                "path": archive_relative,
                "sha256": _sha256(archive_path),
            },
        },
    )


def _validate_resources(
    profile: str, resources: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(resources, Mapping) or not resources:
        raise ValueError("handoff requires caller-supplied resources.")
    result = dict(resources)
    if profile == "direct-local":
        positive_int(result.get("threads", 1), "resources.threads")
        positive_int(result.get("processes", 1), "resources.processes")
    else:
        positive_int(result.get("ntasks"), "resources.ntasks")
        positive_int(result.get("cpus_per_task"), "resources.cpus_per_task")
        nodes = positive_int(result.get("nodes"), "resources.nodes")
        if profile == "slurm-single-node" and nodes != 1:
            raise ValueError(
                "slurm-single-node is a single-node profile and requires resources.nodes=1."
            )
        if profile == "slurm-multi-node":
            if nodes < 2:
                raise ValueError(
                    "slurm-multi-node is a multi-node profile and requires resources.nodes >= 2."
                )
            if not any(key in result for key in ("mem", "mem_per_cpu")):
                raise ValueError(
                    "slurm-multi-node requires caller-supplied high-memory resource (mem or mem_per_cpu)."
                )
    style = result.get("command_style", "binary")
    if style not in {"binary", "wrapper"}:
        raise ValueError("resources.command_style must be 'binary' or 'wrapper'.")
    if profile != "direct-local" and style == "wrapper":
        raise ValueError(
            "resources.command_style='wrapper' is supported only for direct-local; "
            "Slurm profiles require 'binary'."
        )
    return result


def _redacted_command(
    profile: str, executable: str, resources: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "manual_only": True,
        "script": "run_palace.sh" if profile == "direct-local" else "run_palace.sbatch",
        "executable_identity": Path(executable).name,
        "launcher": resources.get(
            "launcher", "local" if profile == "direct-local" else "srun"
        ),
        "arguments": ["[executable]", "config.json"],
    }


def _identity() -> dict[str, Any]:
    try:
        scgsim = version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot prepare handoff without installed scgsim package metadata."
        ) from exc
    return {
        "scgsim": scgsim,
        "candidate_state": "converging",
        "gsim_portable_handoff": PORTABLE_HANDOFF_SHA,
        "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
    }


def _compute_handoff_id(  # noqa: PLR0913 -- hash binds each protocol identity input
    *,
    profile: str,
    route: str,
    problem: str,
    source_revisions: Mapping[str, Any],
    palace_identity: Mapping[str, Any],
    hashes: Sequence[Mapping[str, Any]],
    execution_identity: Mapping[str, Any],
    route_a_thin_film: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "route": route,
        "problem": problem,
        "profile": profile,
        "source_revisions": source_revisions,
        "palace_identity": palace_identity,
        "hashes": list(hashes),
        "execution_identity": execution_identity,
        **(
            {"route_a_thin_film": route_a_thin_film}
            if route_a_thin_film is not None
            else {}
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _execution_identity(  # noqa: PLR0913 -- each value is a distinct identity field
    *,
    profile: str,
    resources: Mapping[str, Any],
    executable: str,
    setup: Sequence[str],
    petsc: Sequence[str],
    script_path: Path,
    recorder_path: Path,
) -> dict[str, Any]:
    """Bind the generated execution surface without recording an absolute path."""

    return {
        "profile": profile,
        "resources": dict(resources),
        "setup_commands": list(setup),
        "petsc_options": list(petsc),
        "command": _redacted_command(profile, executable, resources),
        "script": {"path": script_path.name, "sha256": _sha256(script_path)},
        "recorder": {
            "path": f"metadata/{recorder_path.name}",
            "sha256": _sha256(recorder_path),
        },
    }


def _clear_previous_returned_outputs(
    *, run_dir: Path, returned_receipt_path: Path
) -> None:
    """A new manual handoff cannot reuse outputs or a receipt from an earlier run."""

    returned_receipt_path.unlink(missing_ok=True)
    (run_dir / "metadata" / "palace_runtime_compatibility.json").unlink(missing_ok=True)
    results_path = run_dir / "results" / "palace"
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    for log_path in (run_dir / "logs").glob("palace-*.log"):
        log_path.unlink()


def _members(root: Path, *, exclude: set[Path]) -> list[dict[str, str | int]]:
    members: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if path in exclude:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            members.append({"path": relative + "/", "type": "directory"})
        elif path.is_file():
            members.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return members


def _hashes(root: Path, paths: Sequence[Path]) -> list[dict[str, str | int]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route(mesh_result: MeshBuildResult) -> str:
    route = json.loads(mesh_result.provenance_path.read_text(encoding="utf-8")).get(
        "route"
    )
    if route not in {"A", "B"}:
        raise ValueError("mesh provenance must carry Route A or B.")
    return route


def _route_a_thin_film(
    mesh_result: MeshBuildResult, route: str
) -> dict[str, Any] | None:
    provenance = json.loads(mesh_result.provenance_path.read_text(encoding="utf-8"))
    manifest = json.loads(mesh_result.mesh_manifest_path.read_text(encoding="utf-8"))
    provenance_value = provenance.get("route_a_thin_film")
    manifest_value = manifest.get("route_a_thin_film")
    if route == "B":
        if provenance_value is not None or manifest_value is not None:
            raise ValueError("Route B mesh metadata must not define route_a_thin_film.")
        return None
    if not isinstance(provenance_value, Mapping) or provenance_value != manifest_value:
        raise ValueError(
            "Route A mesh provenance and manifest require identical thin-film identity."
        )
    return dict(provenance_value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["HandoffPlan", "HandoffProfile", "prepare_handoff"]
