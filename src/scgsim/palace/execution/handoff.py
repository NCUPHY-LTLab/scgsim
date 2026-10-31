"""Prepare a manual Palace handoff: the files someone else runs on a cluster or locally.

``prepare_handoff`` takes a meshed, configured run directory and seals it. It writes
the run script, copies in the receipt recorder, hashes every input, writes the
metadata records, and packs a tar archive. SCGSim never executes or submits any of it.
When the run comes back, ``palace/results`` proves it is the run this module sealed:
the handoff ID written here is the SHA-256 of the inputs (``_handoff_id.py``), so it
can be recomputed.

Adapted from gsim portable fixes @ 8cf5fa79fa3abb176940dbfc520ff34a44f4770e.

Not here: running Palace (the generated script does that), writing the receipt
(``returned_receipt.py``, which is copied into the run directory as a standalone
script), and reading a returned run (``palace/results``).

The bytes of the generated script and recorder are hashed into the handoff ID, so a
change to either one is a change of identity.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, TypedDict

from .._files import sha256_file
from .._handoff_id import compute_handoff_id
from ..mesh.build import (
    GSIM_SHA,
    SGB_DERIVATION_BASE_SHA,
    SGB_DERIVATION_IMPORTED_SHA,
    SGB_RUNTIME_AUTHORITY,
    MeshBuildResult,
)
from ..preparation.transforms import SCHEMA_VERSION
from ._checks import positive_int, single_line
from .archive import (
    run_archive_arcname,
    sibling_run_archive_path,
    sibling_run_archive_relative,
)
from .run_script import render_script

HandoffProfile = Literal["direct-local", "slurm-single-node", "slurm-multi-node"]
_PROFILES = {"direct-local", "slurm-single-node", "slurm-multi-node"}
PORTABLE_HANDOFF_SHA = "8cf5fa79fa3abb176940dbfc520ff34a44f4770e"


class _HashRecord(TypedDict):
    """One sealed input file: ``hashes`` in the metadata records."""

    path: str
    bytes: int
    sha256: str


class _ArchiveMember(TypedDict, total=False):
    """One archive manifest member. ``bytes`` and ``sha256`` only exist for files."""

    path: str
    type: Literal["file", "directory"]
    bytes: int
    sha256: str


class _PalaceIdentity(TypedDict):
    """What the handoff requires of Palace, not which Palace it must be.

    Palace's own ``--dry-run`` has to accept the sealed config before it solves; the
    runtime version is recorded as provenance only.
    """

    config_schema: str
    config_compatibility_probe: Literal["native_dry_run"]
    config_compatibility_semantics: Literal["required_before_solve"]
    runtime_identity_semantics: Literal["observed_provenance_only"]


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
    """Where each handoff file lives inside the run directory."""

    run_dir: Path
    script: Path
    metadata: Path
    run_metadata: Path
    resource_record: Path
    manifest: Path
    archive: Path
    archive_relative: str
    returned_receipt: Path
    recorder: Path
    index_map: Path

    @classmethod
    def for_run(cls, run_dir: Path, profile: str) -> _HandoffPaths:
        metadata_dir = run_dir / "metadata"
        return cls(
            run_dir=run_dir,
            script=run_dir / _script_name(profile),
            metadata=metadata_dir / "palace_handoff_metadata.json",
            run_metadata=metadata_dir / "palace_run_metadata.json",
            resource_record=metadata_dir / "palace_resource_record.json",
            manifest=metadata_dir / "palace_handoff_archive_manifest.json",
            archive=sibling_run_archive_path(run_dir),
            archive_relative=sibling_run_archive_relative(run_dir),
            returned_receipt=metadata_dir / "palace_returned_run_receipt.json",
            recorder=metadata_dir / "record_palace_return.py",
            index_map=metadata_dir / "palace_index_map.json",
        )

    def relative(self, path: Path) -> str:
        return path.relative_to(self.run_dir).as_posix()


# ------------------------------------------------------------ the handoff, as steps


def prepare_handoff(
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
    _make_run_directories(run_dir, profile)
    paths = _HandoffPaths.for_run(run_dir, profile)
    _clear_previous_returned_outputs(
        run_dir=run_dir, returned_receipt_path=paths.returned_receipt
    )
    if not paths.index_map.is_file():
        raise FileNotFoundError("handoff requires metadata/palace_index_map.json.")

    route = _route(mesh_result)
    thin_film = _route_a_thin_film(mesh_result, route)
    hashes = _hashes(
        run_dir,
        (
            mesh_result.mesh_path,
            config_path,
            mesh_result.mesh_manifest_path,
            paths.index_map,
        ),
    )
    identity = _identity()
    palace_identity: _PalaceIdentity = {
        "config_schema": SCHEMA_VERSION,
        "config_compatibility_probe": "native_dry_run",
        "config_compatibility_semantics": "required_before_solve",
        "runtime_identity_semantics": "observed_provenance_only",
    }

    _install_recorder(paths.recorder)
    paths.script.write_text(
        render_script(
            profile,
            executable,
            resource_map,
            setup,
            petsc,
            returned_receipt_recorder_name=paths.recorder.name,
            palace_identity=palace_identity,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.script.chmod(paths.script.stat().st_mode | 0o755)

    execution_identity = _execution_identity(
        profile=profile,
        resources=resource_map,
        executable=executable,
        setup=setup,
        petsc=petsc,
        script_path=paths.script,
        recorder_path=paths.recorder,
    )
    handoff_id = compute_handoff_id(
        profile=profile,
        route=route,
        problem=problem,
        source_revisions=identity,
        palace_identity=palace_identity,
        hashes=hashes,
        execution_identity=execution_identity,
        route_a_thin_film=thin_film,
    )
    # Fields every record repeats; each record adds its own.
    shared: dict[str, Any] = {
        "route": route,
        **({"route_a_thin_film": thin_film} if thin_film is not None else {}),
        "problem": problem,
        "handoff_id": handoff_id,
    }

    _write_handoff_metadata(
        paths,
        shared=shared,
        profile=profile,
        palace_identity=palace_identity,
        source_revisions=identity,
        execution_identity=execution_identity,
        executable=executable,
        resources=resource_map,
        setup=setup,
        petsc=petsc,
        config_name=config_path.name,
        mesh_name=mesh_result.mesh_path.name,
        hashes=hashes,
    )
    _write_run_metadata(
        paths,
        shared=shared,
        palace_identity=palace_identity,
        source_revisions=identity,
        execution_identity=execution_identity,
        hashes=hashes,
    )
    _write_archive(paths, hashes)
    _write_resource_record(
        paths,
        shared=shared,
        profile=profile,
        palace_identity=palace_identity,
        source_revisions=identity,
        execution_identity=execution_identity,
        resources=resource_map,
    )

    return HandoffPlan(
        profile,
        "prepared",
        "manual_handoff",
        handoff_id,
        run_dir,
        paths.script,
        paths.metadata,
        paths.run_metadata,
        paths.resource_record,
        paths.manifest,
        paths.archive,
        paths.returned_receipt,
        paths.recorder,
    )


def _script_name(profile: str) -> str:
    return "run_palace.sh" if profile == "direct-local" else "run_palace.sbatch"


def _make_run_directories(run_dir: Path, profile: str) -> None:
    for directory in (
        run_dir / "logs",
        run_dir / "results" / "palace",
        run_dir / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    # Only the script for this profile may remain from an earlier handoff.
    other_script = "run_palace.sbatch" if profile == "direct-local" else "run_palace.sh"
    (run_dir / other_script).unlink(missing_ok=True)


def _install_recorder(recorder_path: Path) -> None:
    """Copy the standalone receipt recorder into the run directory."""
    if recorder_path.is_file():
        recorder_path.unlink()
    shutil.copy2(Path(__file__).with_name("returned_receipt.py"), recorder_path)
    recorder_path.chmod(0o755)


# --------------------------------------------------------------- metadata records


def _write_handoff_metadata(
    paths: _HandoffPaths,
    *,
    shared: Mapping[str, Any],
    profile: str,
    palace_identity: _PalaceIdentity,
    source_revisions: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    executable: str,
    resources: Mapping[str, Any],
    setup: Sequence[str],
    petsc: Sequence[str],
    config_name: str,
    mesh_name: str,
    hashes: list[_HashRecord],
) -> None:
    """What SCGSim prepared. The key order is the on-disk order."""
    _write_json(
        paths.metadata,
        {
            "schema_version": 1,
            "status": "prepared",
            "kind": "manual_handoff",
            "manual_intent": True,
            "profile": profile,
            "route": shared["route"],
            **(
                {"route_a_thin_film": shared["route_a_thin_film"]}
                if "route_a_thin_film" in shared
                else {}
            ),
            "problem": shared["problem"],
            "handoff_id": shared["handoff_id"],
            "returned_receipt_path": paths.relative(paths.returned_receipt),
            "returned_receipt_recorder": paths.relative(paths.recorder),
            "palace_identity": palace_identity,
            "source_revisions": source_revisions,
            "execution_identity": execution_identity,
            "command": _redacted_command(profile, executable, resources),
            "setup_commands": list(setup),
            "petsc_options": list(petsc),
            "requested_resources": resources,
            "resolved_resources": resources,
            "script": paths.script.name,
            "config": config_name,
            "mesh": mesh_name,
            "hashes": hashes,
        },
    )


def _write_run_metadata(
    paths: _HandoffPaths,
    *,
    shared: Mapping[str, Any],
    palace_identity: _PalaceIdentity,
    source_revisions: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    hashes: list[_HashRecord],
) -> None:
    """The run's own record. The returned-receipt recorder later updates its status."""
    _write_json(
        paths.run_metadata,
        {
            "schema_version": 1,
            "status": "not_run",
            "kind": "manual_handoff",
            "route": shared["route"],
            **(
                {"route_a_thin_film": shared["route_a_thin_film"]}
                if "route_a_thin_film" in shared
                else {}
            ),
            "problem": shared["problem"],
            "handoff_id": shared["handoff_id"],
            "returned_receipt_path": paths.relative(paths.returned_receipt),
            "palace_identity": palace_identity,
            "source_revisions": source_revisions,
            "execution_identity": execution_identity,
            "hashes": hashes,
        },
    )


def _write_archive(paths: _HandoffPaths, hashes: list[_HashRecord]) -> None:
    """Write the archive manifest, then the tar it describes.

    The resource record is excluded from the archive because it is written last, with
    the archive's own hash in it. The manifest is excluded from its own listing for
    the same reason: it cannot contain its own hash.
    """
    paths.resource_record.unlink(missing_ok=True)
    paths.archive.unlink(missing_ok=True)
    (paths.run_dir / "palace_handoff.tar.gz").unlink(missing_ok=True)

    excluded = {paths.resource_record, paths.run_dir / "palace_handoff.tar.gz"}
    members = _members(paths.run_dir, exclude=excluded | {paths.manifest})
    _write_json(
        paths.manifest,
        {
            "schema_version": 1,
            "archive": paths.archive_relative,
            "input_hashes": hashes,
            "members": members,
            "archive_members_excluded_from_self_listing": [
                paths.relative(paths.manifest)
            ],
            "archive_members_excluded": [paths.relative(paths.resource_record)],
        },
    )
    with tarfile.open(paths.archive, "w:gz") as archive:
        for path in sorted(paths.run_dir.rglob("*")):
            if path in excluded:
                continue
            archive.add(
                path,
                arcname=run_archive_arcname(paths.run_dir, path),
                recursive=False,
            )


def _write_resource_record(
    paths: _HandoffPaths,
    *,
    shared: Mapping[str, Any],
    profile: str,
    palace_identity: _PalaceIdentity,
    source_revisions: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> None:
    """What the scheduler or local script reports; the recorder updates it later."""
    _write_json(
        paths.resource_record,
        {
            "schema_version": 1,
            "status": "not_submitted",
            "kind": "manual_handoff",
            "profile": profile,
            "route": shared["route"],
            **(
                {"route_a_thin_film": shared["route_a_thin_film"]}
                if "route_a_thin_film" in shared
                else {}
            ),
            "problem": shared["problem"],
            "handoff_id": shared["handoff_id"],
            "source_revisions": source_revisions,
            "palace_identity": palace_identity,
            "execution_identity": execution_identity,
            "requested_resources": resources,
            "resolved_resources": resources,
            "archive": {
                "path": paths.archive_relative,
                "sha256": sha256_file(paths.archive),
            },
        },
    )


# -------------------------------------------------------------------- resources


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
    # "binary" runs the Palace executable directly; "wrapper" is the launcher script
    # that takes -np and -nt. An absent command_style means "binary" everywhere.
    style = result.get("command_style", "binary")
    if style not in {"binary", "wrapper"}:
        raise ValueError("resources.command_style must be 'binary' or 'wrapper'.")
    if profile != "direct-local" and style == "wrapper":
        raise ValueError(
            "resources.command_style='wrapper' is supported only for direct-local; "
            "Slurm profiles require 'binary'."
        )
    return result


# ------------------------------------------------------------------ run script
#
# The script is a list of bash lines. Its run order is: check inputs, set up the
# environment, ask Palace to --dry-run the config (record a receipt and stop if it
# refuses), solve, record the receipt. Both receipts go through the recorder.


# -------------------------------------------------------------- identity records


def _redacted_command(
    profile: str, executable: str, resources: Mapping[str, Any]
) -> dict[str, Any]:
    """The command, with the executable reduced to its file name (no local paths)."""
    return {
        "manual_only": True,
        "script": _script_name(profile),
        "executable_identity": Path(executable).name,
        "launcher": resources.get(
            "launcher", "local" if profile == "direct-local" else "srun"
        ),
        "arguments": ["[executable]", "config.json"],
    }


def _identity() -> dict[str, Any]:
    """The software revisions this handoff was prepared with."""
    try:
        scgsim = version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot prepare handoff without installed scgsim package metadata."
        ) from exc
    return {
        "scgsim": scgsim,
        "candidate_state": "converging",
        "gsim_meshing": GSIM_SHA,
        "gsim_portable_handoff": PORTABLE_HANDOFF_SHA,
        "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
        "sgb_derivation": {
            "base": SGB_DERIVATION_BASE_SHA,
            "imported_development": SGB_DERIVATION_IMPORTED_SHA,
        },
    }


def _execution_identity(
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
        "script": {"path": script_path.name, "sha256": sha256_file(script_path)},
        "recorder": {
            "path": f"metadata/{recorder_path.name}",
            "sha256": sha256_file(recorder_path),
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


def _members(root: Path, *, exclude: set[Path]) -> list[_ArchiveMember]:
    members: list[_ArchiveMember] = []
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
                    "sha256": sha256_file(path),
                }
            )
    return members


def _hashes(root: Path, paths: Sequence[Path]) -> list[_HashRecord]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


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
