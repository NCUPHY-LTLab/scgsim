"""Check that the metadata files describe one and the same run.

A returned run carries three records written by different steps: the handoff metadata
(what SCGSim prepared), the run metadata, and the resource record (what the scheduler
or local script reported). Here they are checked against each other, and against the
files they hash. Nothing here trusts a copied ID: the handoff ID is recomputed from
its inputs.

Not here: the returned receipt (``_returned_receipt.py``) and the tar archive
(``_archive_manifest.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._files import sha256_file
from .._handoff_id import compute_handoff_id
from ..execution.archive import resolve_run_archive_path
from ._artifact_files import _confined_path
from ._expect import (
    _expect_int,
    _expect_list,
    _expect_mapping,
    _expect_scalar,
    _optional_dict,
)
from ._run_documents import HandoffMetadata, ResourceRecord, RunMetadata

_REQUIRED_HASH_ENTRIES = (
    "palace.msh",
    "config.json",
    "metadata/mesh_manifest.json",
    "metadata/palace_index_map.json",
)

# ------------------------------------------------------ metadata headers


def _validate_contract_schema(payload: Mapping[str, Any], label: str) -> None:
    if _expect_scalar(payload, "schema_version", int, fallback=0) != 1:
        raise ValueError(f"{label} schema_version must be integer 1.")


# ``palace_identity`` as ``execution/handoff.py`` writes it: Palace's own --dry-run
# accepts the sealed config before solving (``config_compatibility_probe``). The
# runtime version and commit are provenance only, so declaring them as an
# eligibility gate is refused (docs/specs/palace-electrostatic-sgb.qmd). A run from
# before the probe existed declared a ``runtime`` string instead; it is refused.
def _validate_palace_identity(payload: Mapping[str, Any]) -> None:
    if not payload:
        raise ValueError("palace_identity must be a non-empty mapping.")
    _expect_scalar(payload, "config_schema", str)
    probe = payload.get("config_compatibility_probe")
    if probe is None:
        raise ValueError(
            "palace_identity has no config_compatibility_probe; this run was written "
            "by an older SCGSim. Supported: 'native_dry_run'. Prepare the handoff again."
        )
    if probe != "native_dry_run":
        raise ValueError("palace_identity config compatibility probe is invalid.")
    if payload.get("config_compatibility_semantics") != "required_before_solve":
        raise ValueError(
            "palace_identity must require config compatibility before solving."
        )
    if payload.get("runtime_identity_semantics") != "observed_provenance_only":
        raise ValueError(
            "palace_identity must treat runtime identity as observed provenance only."
        )
    if (
        payload.get("runtime") is not None
        or payload.get("runtime_git_commit") is not None
    ):
        raise ValueError(
            "palace_identity must not declare a runtime version or Git eligibility Gate."
        )


# ------------------------------------ handoff, run and resource agree


def _validate_scalar_identity(
    handoff_metadata: HandoffMetadata, run_metadata: RunMetadata
) -> None:
    handoff_identity = handoff_metadata.get("source_revisions")
    run_identity = run_metadata.get("source_revisions")
    if not isinstance(handoff_identity, dict) or not handoff_identity:
        raise ValueError("handoff source_revisions must be a non-empty mapping.")
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError("run source_revisions must be a non-empty mapping.")

    _validate_palace_identity(
        _expect_mapping(
            handoff_metadata.get("palace_identity"), "handoff_metadata.palace_identity"
        )
    )
    _validate_palace_identity(
        _expect_mapping(
            run_metadata.get("palace_identity"), "run_metadata.palace_identity"
        )
    )

    if handoff_identity != run_identity:
        raise ValueError(
            "manual handoff identity mismatch between handoff and run metadata."
        )

    handoff_palace_identity = handoff_metadata.get("palace_identity")
    run_palace_identity = run_metadata.get("palace_identity")
    if handoff_palace_identity != run_palace_identity:
        raise ValueError("palace identity mismatch between handoff and run metadata.")

    handoff_id = handoff_metadata.get("handoff_id")
    run_id = run_metadata.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        raise ValueError("handoff metadata must include handoff_id.")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run metadata must include handoff_id.")
    if handoff_id != run_id:
        raise ValueError("handoff id mismatch between handoff and run metadata.")
    if handoff_metadata.get("route_a_thin_film") != run_metadata.get(
        "route_a_thin_film"
    ):
        raise ValueError(
            "Route-A thin-film identity mismatch between handoff and run metadata."
        )


def _validate_execution_identity(root: Path, handoff_metadata: HandoffMetadata) -> None:
    execution = _expect_mapping(
        handoff_metadata.get("execution_identity"), "handoff execution_identity"
    )
    if execution != {
        "profile": _expect_scalar(handoff_metadata, "profile", str),
        "resources": _expect_mapping(
            handoff_metadata.get("requested_resources"), "handoff requested_resources"
        ),
        "setup_commands": _expect_list(
            handoff_metadata.get("setup_commands"),
            "handoff setup_commands",
            allow_empty=True,
        ),
        "petsc_options": _expect_list(
            handoff_metadata.get("petsc_options"),
            "handoff petsc_options",
            allow_empty=True,
        ),
        "command": _expect_mapping(handoff_metadata.get("command"), "handoff command"),
        "script": execution.get("script"),
        "recorder": execution.get("recorder"),
    }:
        raise ValueError("handoff execution identity does not match handoff metadata.")
    for field, relative in (
        ("script", _expect_scalar(handoff_metadata, "script", str)),
        (
            "recorder",
            _expect_scalar(handoff_metadata, "returned_receipt_recorder", str),
        ),
    ):
        item = _expect_mapping(execution.get(field), f"execution identity {field}")
        if _expect_scalar(item, "path", str) != relative:
            raise ValueError(
                f"handoff execution identity {field} path is inconsistent."
            )
        path = _confined_path(root, relative)
        if not path.is_file() or _expect_scalar(item, "sha256", str) != sha256_file(
            path
        ):
            raise ValueError(
                f"handoff execution identity {field} hash is inconsistent."
            )


def _validate_handoff_id(
    handoff_metadata: HandoffMetadata, expected_handoff_id: str
) -> None:
    """Recompute the identity; colocated copied IDs are not evidence."""

    computed = compute_handoff_id(
        profile=_expect_scalar(handoff_metadata, "profile", str),
        route=_expect_scalar(handoff_metadata, "route", str),
        problem=_expect_scalar(handoff_metadata, "problem", str),
        source_revisions=_expect_mapping(
            handoff_metadata.get("source_revisions"), "handoff source_revisions"
        ),
        palace_identity=_expect_mapping(
            handoff_metadata.get("palace_identity"), "handoff palace_identity"
        ),
        hashes=_expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes"),
        execution_identity=_expect_mapping(
            handoff_metadata.get("execution_identity"), "handoff execution_identity"
        ),
        route_a_thin_film=(
            _expect_mapping(
                handoff_metadata.get("route_a_thin_film"),
                "handoff route_a_thin_film",
            )
            if handoff_metadata.get("route_a_thin_film") is not None
            else None
        ),
    )
    recorded = _expect_scalar(handoff_metadata, "handoff_id", str)
    if computed != recorded or computed != expected_handoff_id:
        raise ValueError(
            "recomputed handoff identity does not match expected handoff_id."
        )


# The ID is the SHA-256 of canonical JSON (sorted keys, no spaces). A copy of this
# computation lives in palace/execution/handoff.py; the two must stay identical.


def _validate_resource_identity(
    resource_record: ResourceRecord, handoff_metadata: HandoffMetadata
) -> None:
    for field in (
        "profile",
        "route",
        "problem",
        "handoff_id",
        "source_revisions",
        "palace_identity",
        "execution_identity",
        "requested_resources",
        "resolved_resources",
        "route_a_thin_film",
    ):
        if resource_record.get(field) != handoff_metadata.get(field):
            raise ValueError(
                f"resource record {field} does not match handoff metadata."
            )
    if _expect_scalar(resource_record, "status", str) != "completed":
        raise ValueError("resource record status must be completed for a returned run.")
    if _expect_int(resource_record.get("exit_code")) != 0:
        raise ValueError("resource record exit_code must be zero for a returned run.")


# ---------------------------------- hashes and archive record are real files


# Every listed file is re-read and re-hashed. The metadata must also cover the mesh,
# config, mesh manifest and index map.
def _validate_hashes(
    root: Path, metadata: HandoffMetadata | RunMetadata, label: str
) -> None:
    hashes = _expect_list(metadata.get("hashes"), f"{label} metadata hashes")
    if not hashes:
        raise ValueError(f"{label} metadata has no hash entries.")

    observed: dict[str, tuple[int, str]] = {}
    for entry in hashes:
        if not isinstance(entry, dict):
            raise TypeError(f"{label} metadata hashes must be mapping entries.")
        path_name = _expect_scalar(entry, "path", str)
        bytes_value = _expect_int(entry.get("bytes"))
        sha_value = _expect_scalar(entry, "sha256", str)
        file_path = _confined_path(root, path_name)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{label} metadata references missing artifact {path_name}."
            )

        observed[path_name] = (
            _expect_int(file_path.stat().st_size),
            sha256_file(file_path),
        )
        if bytes_value != observed[path_name][0]:
            raise ValueError(
                f"{label} hash byte size mismatch for {path_name}: "
                f"declared {bytes_value}, observed {observed[path_name][0]}"
            )
        if sha_value != observed[path_name][1]:
            raise ValueError(f"{label} hash mismatch for {path_name}.")

    missing = [name for name in _REQUIRED_HASH_ENTRIES if name not in observed]
    if missing:
        raise ValueError(
            f"{label} metadata missing hash entries for: {', '.join(missing)}"
        )


def _validate_archive_record(root: Path, resource_record: ResourceRecord) -> None:
    if resource_record.get("kind") != "manual_handoff":
        raise ValueError("resource record kind must be manual_handoff.")

    if _expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError("resource record status must be a non-empty string.")
    requested_resources = _optional_dict(resource_record.get("requested_resources"))
    resolved_resources = _optional_dict(resource_record.get("resolved_resources"))
    if not requested_resources:
        raise ValueError("resource record requested_resources must be a mapping.")
    if not resolved_resources:
        raise ValueError("resource record resolved_resources must be a mapping.")

    archive = _optional_dict(resource_record.get("archive"))
    if not archive:
        raise ValueError("resource record archive must be a non-empty mapping.")
    path_name = _expect_scalar(archive, "path", str)
    expected_sha = _expect_scalar(archive, "sha256", str)
    archive_path = resolve_run_archive_path(root, path_name)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"resource record references missing archive payload {archive_path}"
        )
    if sha256_file(archive_path) != expected_sha:
        raise ValueError("resource record archive hash mismatch.")
