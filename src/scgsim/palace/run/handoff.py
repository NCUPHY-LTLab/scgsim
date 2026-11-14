"""Write the run script, metadata and archive that hand a meshed and configured run folder to a manual Palace run.

Portions adapted from gsim portable fixes, revision 8cf5fa79fa3abb176940dbfc520ff34a44f4770e; see NOTICE.
"""

from __future__ import annotations

import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast, get_args

from scgsim.contracts.palace_run_folder import (
    ARCHIVE_MANIFEST,
    CONFIG,
    CONFIG_SCHEMA,
    DIRECT_LOCAL_SCRIPT,
    HANDOFF_METADATA,
    INDEX_MAP,
    LOGS_DIR,
    METADATA_DIR,
    OUTPUT_DIR,
    RECORDER,
    RESOURCE_RECORD,
    RETURNED_RECEIPT,
    RUN_METADATA,
    RUN_RECORD_SCHEMA_VERSION,
    RUNTIME_COMPATIBILITY,
    SEALED_WHEN_PRESENT,
    SLURM_SCRIPT,
    HandoffIdentity,
    read_json,
    script_name,
    sha256,
    solver_log,
    write_json,
)
from scgsim.refusal import ENVIRONMENT, Declared, found_text

from ..mesh.build import MeshBuildResult
from ..mesh.manifest import SGB_RUNTIME_AUTHORITY
from .archive import (
    run_archive_arcname,
    sibling_run_archive_path,
    sibling_run_archive_relative,
)
from .recorder_copy import write_standalone_recorder
from .script import ExecutionSettings, positive_int, render_script, single_line

_WHERE_MESH_THEN_CONFIG = (
    "the run folder: call mesh() and write_config() before prepare_handoff()"
)

_WHERE_HANDOFF_CALL = "the call to prepare_handoff"

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
class _HandoffPaths:
    """Where one handoff reads and writes in its run folder, from the run-folder contract."""

    run_dir: Path
    config: Path
    mesh: Path
    index_map: Path
    script: Path
    metadata: Path
    run_metadata: Path
    resource_record: Path
    archive_manifest: Path
    archive: Path
    returned_receipt: Path
    recorder: Path

    @classmethod
    def of(
        cls, mesh_result: MeshBuildResult, config_path: Path, profile: str
    ) -> _HandoffPaths:
        """The paths of a handoff of ``mesh_result`` and ``config_path`` for ``profile``."""
        run_dir = mesh_result.output_dir
        return cls(
            run_dir=run_dir,
            config=config_path,
            mesh=mesh_result.mesh_path,
            index_map=run_dir / INDEX_MAP,
            script=run_dir / script_name(profile),
            metadata=run_dir / HANDOFF_METADATA,
            run_metadata=run_dir / RUN_METADATA,
            resource_record=run_dir / RESOURCE_RECORD,
            archive_manifest=run_dir / ARCHIVE_MANIFEST,
            archive=sibling_run_archive_path(run_dir),
            returned_receipt=run_dir / RETURNED_RECEIPT,
            recorder=run_dir / RECORDER,
        )

    def relative(self, path: Path) -> str:
        """``path`` as a POSIX path relative to the run folder."""
        return path.relative_to(self.run_dir).as_posix()


@dataclass(frozen=True)
class _PreparedArtifacts:
    """What the handoff binds before it writes its records: the identities its id is the hash of."""

    route: str
    thin_film: dict[str, Any] | None
    hashes: list[dict[str, str | int]]
    source_revisions: dict[str, Any]
    palace_identity: dict[str, Any]
    execution_identity: dict[str, Any]
    handoff_id: str

    def thin_film_field(self) -> dict[str, Any]:
        """The ``route_a_thin_film`` field of a record, empty for Route B."""
        return {} if self.thin_film is None else {"route_a_thin_film": self.thin_film}


def prepare_handoff(  # noqa: PLR0913 -- established public handoff API
    *,
    profile: str,
    mesh_result: MeshBuildResult,
    config_path: Path,
    executable: str,
    resources: Mapping[str, Any] | None = None,
    setup_commands: Sequence[str] = (),
    petsc_options: Sequence[str] = (),
    problem: Literal["Electrostatic", "Eigenmode"],
) -> HandoffPlan:
    """Write the run script, metadata, receipt recorder and archive; the caller names the executable and the resources.

    Nothing here starts Palace.
    """
    if profile not in _PROFILES:
        raise ValueError(
            Declared(f"handoff profile {profile!r}", _WHERE_HANDOFF_CALL).refusal(
                f"is not supported: use one of {list(get_args(HandoffProfile))}"
            )
        )
    if not config_path.is_file() or not mesh_result.mesh_path.is_file():
        raise FileNotFoundError(
            Declared("handoff", _WHERE_MESH_THEN_CONFIG).refusal(
                f"requires existing {CONFIG} and {mesh_result.mesh_path.name}"
            )
        )
    executable = single_line(executable, "executable")
    setup = tuple(single_line(command, "setup command") for command in setup_commands)
    petsc = tuple(single_line(option, "PETSc option") for option in petsc_options)
    execution = ExecutionSettings(
        profile=profile,
        executable=executable,
        resources=_validate_resources(profile, resources),
        setup_commands=setup,
        petsc_options=petsc,
    )
    paths = _HandoffPaths.of(mesh_result, config_path, profile)
    _prepare_run_folder(paths)
    artifacts = _prepare_artifacts(execution, problem, mesh_result, paths)
    _write_handoff_metadata(artifacts, execution, problem, paths)
    paths.resource_record.unlink(missing_ok=True)
    paths.archive.unlink(missing_ok=True)
    _create_archive(artifacts, execution, problem, paths)
    return HandoffPlan(
        cast(HandoffProfile, profile),
        "prepared",
        "manual_handoff",
        artifacts.handoff_id,
        paths.run_dir,
        paths.script,
        paths.metadata,
        paths.run_metadata,
        paths.resource_record,
        paths.archive_manifest,
        paths.archive,
        paths.returned_receipt,
        paths.recorder,
    )


def _prepare_run_folder(paths: _HandoffPaths) -> None:
    """Create the run folder's folders and remove what an earlier handoff or run left.

    A new manual handoff cannot reuse outputs, a receipt or the other profile's script from an earlier one.
    """
    run_dir = paths.run_dir
    for directory in (LOGS_DIR, OUTPUT_DIR, METADATA_DIR):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    for stale in {DIRECT_LOCAL_SCRIPT, SLURM_SCRIPT} - {paths.script.name}:
        (run_dir / stale).unlink(missing_ok=True)
    paths.returned_receipt.unlink(missing_ok=True)
    (run_dir / RUNTIME_COMPATIBILITY).unlink(missing_ok=True)
    results_path = run_dir / OUTPUT_DIR
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    for log_path in run_dir.glob(solver_log("*")):
        log_path.unlink()


def _prepare_artifacts(
    execution: ExecutionSettings,
    problem: str,
    mesh_result: MeshBuildResult,
    paths: _HandoffPaths,
) -> _PreparedArtifacts:
    """Bind inputs and prepare the standalone script and recorder before metadata."""
    if not paths.index_map.is_file():
        raise FileNotFoundError(
            Declared("handoff", _WHERE_MESH_THEN_CONFIG).refusal(
                f"requires {INDEX_MAP}"
            )
        )

    route = _route(mesh_result)
    thin_film = _route_a_thin_film(mesh_result, route)
    sealed_when_present = [
        paths.run_dir / relative
        for relative in SEALED_WHEN_PRESENT
        if (paths.run_dir / relative).is_file()
    ]
    hashes = _hashes(
        paths,
        (
            paths.mesh,
            paths.config,
            mesh_result.mesh_manifest_path,
            paths.index_map,
            *sealed_when_present,
        ),
    )
    source_revisions = _identity()
    palace_identity = {
        "config_schema": CONFIG_SCHEMA,
        "config_compatibility_probe": "native_dry_run",
        "config_compatibility_semantics": "required_before_solve",
        "runtime_identity_semantics": "observed_provenance_only",
    }

    write_standalone_recorder(paths.recorder)
    paths.script.write_text(
        render_script(execution, config_schema=CONFIG_SCHEMA) + "\n",
        encoding="utf-8",
    )
    paths.script.chmod(paths.script.stat().st_mode | 0o755)

    execution_identity = _execution_identity(execution, paths)
    handoff_id = HandoffIdentity(
        profile=execution.profile,
        route=route,
        problem=problem,
        source_revisions=source_revisions,
        palace_identity=palace_identity,
        hashes=list(hashes),
        execution_identity=execution_identity,
        route_a_thin_film=thin_film,
    ).digest()
    return _PreparedArtifacts(
        route,
        thin_film,
        hashes,
        source_revisions,
        palace_identity,
        execution_identity,
        handoff_id,
    )


def _write_handoff_metadata(
    artifacts: _PreparedArtifacts,
    execution: ExecutionSettings,
    problem: str,
    paths: _HandoffPaths,
) -> None:
    """Write the prepared and not-run records after script identities are known."""
    returned_receipt_relative = paths.relative(paths.returned_receipt)
    write_json(
        paths.metadata,
        {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "status": "prepared",
            "kind": "manual_handoff",
            "manual_intent": True,
            "profile": execution.profile,
            "route": artifacts.route,
            **artifacts.thin_film_field(),
            "problem": problem,
            "handoff_id": artifacts.handoff_id,
            "returned_receipt_path": returned_receipt_relative,
            "returned_receipt_recorder": paths.relative(paths.recorder),
            "palace_identity": artifacts.palace_identity,
            "source_revisions": artifacts.source_revisions,
            "execution_identity": artifacts.execution_identity,
            "command": _redacted_command(execution),
            "setup_commands": list(execution.setup_commands),
            "petsc_options": list(execution.petsc_options),
            "requested_resources": execution.resources,
            "resolved_resources": execution.resources,
            "script": paths.script.name,
            "config": paths.config.name,
            "mesh": paths.mesh.name,
            "hashes": artifacts.hashes,
        },
    )
    write_json(
        paths.run_metadata,
        {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "status": "not_run",
            "kind": "manual_handoff",
            "route": artifacts.route,
            **artifacts.thin_film_field(),
            "problem": problem,
            "handoff_id": artifacts.handoff_id,
            "returned_receipt_path": returned_receipt_relative,
            "palace_identity": artifacts.palace_identity,
            "source_revisions": artifacts.source_revisions,
            "execution_identity": artifacts.execution_identity,
            "hashes": artifacts.hashes,
        },
    )


def _create_archive(
    artifacts: _PreparedArtifacts,
    execution: ExecutionSettings,
    problem: str,
    paths: _HandoffPaths,
) -> None:
    """Create the archive and record its digest after all archived inputs settle."""
    run_dir = paths.run_dir
    archive_relative = sibling_run_archive_relative(run_dir)
    excluded = {paths.resource_record}
    members = _members(run_dir, exclude=excluded | {paths.archive_manifest})
    write_json(
        paths.archive_manifest,
        {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "archive": archive_relative,
            "input_hashes": artifacts.hashes,
            "members": members,
            "archive_members_excluded_from_self_listing": [
                paths.relative(paths.archive_manifest)
            ],
            "archive_members_excluded": [paths.relative(paths.resource_record)],
        },
    )
    with tarfile.open(paths.archive, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path in excluded:
                continue
            archive.add(
                path,
                arcname=run_archive_arcname(run_dir, path),
                recursive=False,
            )
    write_json(
        paths.resource_record,
        {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "status": "not_submitted",
            "kind": "manual_handoff",
            "profile": execution.profile,
            "route": artifacts.route,
            **artifacts.thin_film_field(),
            "problem": problem,
            "handoff_id": artifacts.handoff_id,
            "source_revisions": artifacts.source_revisions,
            "palace_identity": artifacts.palace_identity,
            "execution_identity": artifacts.execution_identity,
            "requested_resources": execution.resources,
            "resolved_resources": execution.resources,
            "archive": {
                "path": archive_relative,
                "sha256": sha256(paths.archive),
            },
        },
    )


def _validate_resources(
    profile: str, resources: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(resources, Mapping) or not resources:
        raise ValueError(
            Declared(
                "handoff requires caller-supplied resources", _WHERE_HANDOFF_CALL
            ).refusal(found_text(resources))
        )
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
                Declared(
                    "slurm-single-node is a single-node profile and requires resources.nodes=1",
                    _WHERE_HANDOFF_CALL,
                ).refusal(found_text(nodes))
            )
        if profile == "slurm-multi-node":
            if nodes < 2:
                raise ValueError(
                    Declared(
                        "slurm-multi-node is a multi-node profile and requires resources.nodes >= 2",
                        _WHERE_HANDOFF_CALL,
                    ).refusal(found_text(nodes))
                )
            if not any(key in result for key in ("mem", "mem_per_cpu")):
                raise ValueError(
                    Declared(
                        "slurm-multi-node requires caller-supplied high-memory resource (mem or mem_per_cpu)",
                        _WHERE_HANDOFF_CALL,
                    ).refusal(found_text(sorted(result)))
                )
    style = result.get("command_style", "binary")
    if style not in {"binary", "wrapper"}:
        raise ValueError(
            Declared(
                "resources.command_style must be 'binary' or 'wrapper'",
                _WHERE_HANDOFF_CALL,
            ).refusal(found_text(style))
        )
    if profile != "direct-local" and style == "wrapper":
        raise ValueError(
            Declared("resources.command_style='wrapper'", _WHERE_HANDOFF_CALL).refusal(
                f"is supported only for direct-local; profile {profile!r} requires 'binary'"
            )
        )
    return result


def _redacted_command(execution: ExecutionSettings) -> dict[str, Any]:
    """The solver command as the records show it: the script, the executable's name, the launcher."""
    profile = execution.profile
    return {
        "manual_only": True,
        "script": script_name(profile),
        "executable_identity": Path(execution.executable).name,
        "launcher": execution.resources.get(
            "launcher", "local" if profile == "direct-local" else "srun"
        ),
        "arguments": ["[executable]", CONFIG],
    }


def _identity() -> dict[str, Any]:
    try:
        scgsim = version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            Declared(
                "Cannot prepare handoff without installed scgsim package metadata",
                ENVIRONMENT,
            ).refusal(
                "install SCGSim (uv sync, or pip install -e .) rather than running it from a source tree"
            )
        ) from exc
    return {
        "scgsim": scgsim,
        "candidate_state": "converging",
        "gsim_portable_handoff": PORTABLE_HANDOFF_SHA,
        "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
    }


def _execution_identity(
    execution: ExecutionSettings, paths: _HandoffPaths
) -> dict[str, Any]:
    """Bind the generated execution surface without recording an absolute path."""
    return {
        "profile": execution.profile,
        "resources": dict(execution.resources),
        "setup_commands": list(execution.setup_commands),
        "petsc_options": list(execution.petsc_options),
        "command": _redacted_command(execution),
        "script": {"path": paths.script.name, "sha256": sha256(paths.script)},
        "recorder": {"path": RECORDER, "sha256": sha256(paths.recorder)},
    }


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
                    "sha256": sha256(path),
                }
            )
    return members


def _hashes(paths: _HandoffPaths, files: Sequence[Path]) -> list[dict[str, str | int]]:
    """The run-folder path, size and SHA-256 of each sealed input, in the order given."""
    return [
        {
            "path": paths.relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]


def _route(mesh_result: MeshBuildResult) -> str:
    route = read_json(mesh_result.provenance_path).get("route")
    if route not in {"A", "B"}:
        raise ValueError(
            Declared(
                "mesh provenance must carry Route A or B", _WHERE_HANDOFF_CALL
            ).refusal(found_text(route))
        )
    return route


def _route_a_thin_film(
    mesh_result: MeshBuildResult, route: str
) -> dict[str, Any] | None:
    provenance = read_json(mesh_result.provenance_path)
    manifest = read_json(mesh_result.mesh_manifest_path)
    provenance_value = provenance.get("route_a_thin_film")
    manifest_value = manifest.get("route_a_thin_film")
    if route == "B":
        if provenance_value is not None or manifest_value is not None:
            raise ValueError(
                Declared(
                    "Route B mesh metadata must not define route_a_thin_film",
                    _WHERE_HANDOFF_CALL,
                ).refusal(
                    found_text(
                        {"provenance": provenance_value, "manifest": manifest_value}
                    )
                )
            )
        return None
    if not isinstance(provenance_value, Mapping) or provenance_value != manifest_value:
        raise ValueError(
            Declared(
                "Route A mesh provenance and manifest require identical thin-film identity",
                _WHERE_HANDOFF_CALL,
            ).refusal(found_text(provenance_value, manifest_value))
        )
    return dict(provenance_value)


__all__ = ["HandoffPlan", "HandoffProfile", "prepare_handoff"]
