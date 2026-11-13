"""Validate handoff identity, recorded hashes and exact archived input members."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import (
    HANDOFF_METADATA,
    RUN_METADATA,
    RUN_RECORD_SCHEMA_VERSION,
    SEALED_INPUTS,
    HandoffIdentity,
    confined_path,
    sha256,
)
from scgsim.palace.results.artifacts import (
    FileHash,
    expect_int,
    expect_list,
    expect_mapping,
    expect_scalar,
    extract_hash_map,
    optional_dict,
)
from scgsim.palace.results.runtime_contract import validate_palace_identity
from scgsim.palace.run.archive import (
    logical_tar_member_name,
    resolve_run_archive_path,
)
from scgsim.refusal import refusal


def validate_scalar_identity(
    handoff_metadata: dict[str, Any], run_metadata: dict[str, Any]
) -> None:
    """Match shared handoff and run fields before checking artifact identities."""
    handoff_identity = handoff_metadata.get("source_revisions")
    run_identity = run_metadata.get("source_revisions")
    if not isinstance(handoff_identity, dict) or not handoff_identity:
        raise ValueError(
            refusal(
                "handoff source_revisions must be a non-empty mapping",
                found=handoff_identity,
            )
        )
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError(
            refusal(
                "run source_revisions must be a non-empty mapping", found=run_identity
            )
        )

    validate_palace_identity(
        expect_mapping(
            handoff_metadata.get("palace_identity"), "handoff_metadata.palace_identity"
        )
    )
    validate_palace_identity(
        expect_mapping(
            run_metadata.get("palace_identity"), "run_metadata.palace_identity"
        )
    )

    if handoff_identity != run_identity:
        raise ValueError(
            refusal(
                "manual handoff identity mismatch between handoff and run metadata",
                found=run_identity,
                expected=handoff_identity,
            )
        )

    handoff_palace_identity = handoff_metadata.get("palace_identity")
    run_palace_identity = run_metadata.get("palace_identity")
    if handoff_palace_identity != run_palace_identity:
        raise ValueError(
            refusal(
                "palace identity mismatch between handoff and run metadata",
                found=run_palace_identity,
                expected=handoff_palace_identity,
            )
        )

    handoff_id = handoff_metadata.get("handoff_id")
    run_id = run_metadata.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        raise ValueError(
            refusal("handoff metadata must include handoff_id", found=handoff_id)
        )
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(refusal("run metadata must include handoff_id", found=run_id))
    if handoff_id != run_id:
        raise ValueError(
            refusal(
                "handoff id mismatch between handoff and run metadata",
                found=run_id,
                expected=handoff_id,
            )
        )
    if handoff_metadata.get("route_a_thin_film") != run_metadata.get(
        "route_a_thin_film"
    ):
        raise ValueError(
            refusal(
                "Route-A thin-film identity mismatch between handoff and run metadata",
                found=run_metadata.get("route_a_thin_film"),
                expected=handoff_metadata.get("route_a_thin_film"),
            )
        )


def validate_handoff_id(
    handoff_metadata: dict[str, Any], expected_handoff_id: str
) -> None:
    """Recompute the identity; colocated copied IDs are not evidence."""

    computed = HandoffIdentity(
        profile=expect_scalar(handoff_metadata, "profile", str),
        route=expect_scalar(handoff_metadata, "route", str),
        problem=expect_scalar(handoff_metadata, "problem", str),
        source_revisions=expect_mapping(
            handoff_metadata.get("source_revisions"), "handoff source_revisions"
        ),
        palace_identity=expect_mapping(
            handoff_metadata.get("palace_identity"), "handoff palace_identity"
        ),
        hashes=expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes"),
        execution_identity=expect_mapping(
            handoff_metadata.get("execution_identity"), "handoff execution_identity"
        ),
        route_a_thin_film=expect_mapping(
            handoff_metadata.get("route_a_thin_film"),
            "handoff route_a_thin_film",
        )
        if handoff_metadata.get("route_a_thin_film") is not None
        else None,
    ).digest()
    recorded = expect_scalar(handoff_metadata, "handoff_id", str)
    if computed != recorded or computed != expected_handoff_id:
        raise ValueError(
            refusal(
                "recomputed handoff identity does not match expected handoff_id; the sealed "
                "inputs or the recorded handoff_id changed",
                found={"recomputed": computed, "recorded": recorded},
                expected=expected_handoff_id,
            )
        )


def validate_execution_identity(root: Path, handoff_metadata: dict[str, Any]) -> None:
    """Validate the command identity and its sealed execution-side metadata."""
    execution = expect_mapping(
        handoff_metadata.get("execution_identity"), "handoff execution_identity"
    )
    recorded = {
        "profile": expect_scalar(handoff_metadata, "profile", str),
        "resources": expect_mapping(
            handoff_metadata.get("requested_resources"), "handoff requested_resources"
        ),
        "setup_commands": expect_list(
            handoff_metadata.get("setup_commands"),
            "handoff setup_commands",
            allow_empty=True,
        ),
        "petsc_options": expect_list(
            handoff_metadata.get("petsc_options"),
            "handoff petsc_options",
            allow_empty=True,
        ),
        "command": expect_mapping(handoff_metadata.get("command"), "handoff command"),
        "script": execution.get("script"),
        "recorder": execution.get("recorder"),
    }
    if execution != recorded:
        raise ValueError(
            refusal(
                "handoff execution identity does not match handoff metadata",
                found=execution,
                expected=recorded,
            )
        )
    for field, relative in (
        ("script", expect_scalar(handoff_metadata, "script", str)),
        (
            "recorder",
            expect_scalar(handoff_metadata, "returned_receipt_recorder", str),
        ),
    ):
        item = expect_mapping(execution.get(field), f"execution identity {field}")
        recorded_path = expect_scalar(item, "path", str)
        if recorded_path != relative:
            raise ValueError(
                refusal(
                    f"handoff execution identity {field} path is inconsistent",
                    found=recorded_path,
                    expected=relative,
                )
            )
        path = confined_path(root, relative)
        recorded_sha = expect_scalar(item, "sha256", str) if path.is_file() else None
        if not path.is_file() or recorded_sha != sha256(path):
            raise ValueError(
                refusal(
                    f"handoff execution identity {field} hash is inconsistent for {relative}",
                    found=sha256(path) if path.is_file() else "no file",
                    expected=item.get("sha256"),
                )
            )


def validate_resource_identity(
    resource_record: dict[str, Any], handoff_metadata: dict[str, Any]
) -> None:
    """Require completed resource metadata to describe the same handoff and route."""
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
                refusal(
                    f"resource record {field} does not match handoff metadata",
                    found=resource_record.get(field),
                    expected=handoff_metadata.get(field),
                )
            )
    status = expect_scalar(resource_record, "status", str)
    if status != "completed":
        raise ValueError(
            refusal(
                "resource record status must be completed for a returned run",
                found=status,
                expected="completed",
            )
        )
    exit_code = expect_int(
        resource_record.get("exit_code"), "resource record exit_code"
    )
    if exit_code != 0:
        raise ValueError(
            refusal(
                "resource record exit_code must be zero for a returned run",
                found=exit_code,
                expected=0,
            )
        )


def validate_hashes(root: Path, metadata: dict[str, Any], label: str) -> None:
    """Verify required input identities against the exact run-directory bytes."""
    hashes = expect_list(metadata.get("hashes"), f"{label} metadata hashes")
    if not hashes:
        raise ValueError(f"{label} metadata has no hash entries.")

    observed: dict[str, FileHash] = {}
    for entry in hashes:
        if not isinstance(entry, dict):
            raise TypeError(
                refusal(f"{label} metadata hashes must be mapping entries", found=entry)
            )
        path_name = expect_scalar(entry, "path", str)
        bytes_value = expect_int(
            entry.get("bytes"), f"{label} metadata hash entry {path_name!r} bytes"
        )
        sha_value = expect_scalar(entry, "sha256", str)
        file_path = confined_path(root, path_name)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{label} metadata references missing artifact {path_name}."
            )

        observed[path_name] = FileHash(
            expect_int(file_path.stat().st_size, f"size of {path_name}"),
            sha256(file_path),
        )
        if bytes_value != observed[path_name].bytes:
            raise ValueError(
                f"{label} hash byte size mismatch for {path_name}: "
                f"declared {bytes_value}, observed {observed[path_name].bytes}"
            )
        if sha_value != observed[path_name].sha256:
            raise ValueError(
                refusal(
                    f"{label} hash mismatch for {path_name}",
                    found=observed[path_name].sha256,
                    expected=sha_value,
                )
            )

    missing = [name for name in SEALED_INPUTS if name not in observed]
    if missing:
        raise ValueError(
            f"{label} metadata missing hash entries for: {', '.join(missing)}"
        )


def validate_archive_record(root: Path, resource_record: dict[str, Any]) -> None:
    """Verify the retained compressed handoff archive against its resource record."""
    if resource_record.get("kind") != "manual_handoff":
        raise ValueError(
            refusal(
                "resource record kind must be manual_handoff",
                found=resource_record.get("kind"),
                expected="manual_handoff",
            )
        )

    if expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError(
            refusal(
                "resource record status must be a non-empty string",
                found=resource_record.get("status"),
            )
        )
    requested_resources = optional_dict(resource_record.get("requested_resources"))
    resolved_resources = optional_dict(resource_record.get("resolved_resources"))
    if not requested_resources:
        raise ValueError(
            refusal(
                "resource record requested_resources must be a non-empty mapping",
                found=resource_record.get("requested_resources"),
            )
        )
    if not resolved_resources:
        raise ValueError(
            refusal(
                "resource record resolved_resources must be a non-empty mapping",
                found=resource_record.get("resolved_resources"),
            )
        )

    archive = optional_dict(resource_record.get("archive"))
    if not archive:
        raise ValueError(
            refusal(
                "resource record archive must be a non-empty mapping",
                found=resource_record.get("archive"),
            )
        )
    path_name = expect_scalar(archive, "path", str)
    expected_sha = expect_scalar(archive, "sha256", str)
    archive_path = resolve_run_archive_path(root, path_name)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"resource record references missing archive payload {archive_path}"
        )
    archive_sha = sha256(archive_path)
    if archive_sha != expected_sha:
        raise ValueError(
            refusal(
                f"resource record archive hash mismatch for {archive_path}",
                found=archive_sha,
                expected=expected_sha,
            )
        )


def validate_archive_manifest(
    root: Path,
    manifest: dict[str, Any],
    resource_record: dict[str, Any],
    handoff_metadata: dict[str, Any],
) -> None:
    """Validate the archive inventory before comparing its members and exact file bytes."""
    if (
        expect_scalar(manifest, "schema_version", int, fallback=0)
        != RUN_RECORD_SCHEMA_VERSION
    ):
        raise ValueError(
            "handoff archive manifest schema_version must be integer "
            f"{RUN_RECORD_SCHEMA_VERSION}."
        )
    archive = expect_mapping(resource_record.get("archive"), "resource_record.archive")
    manifest_hashes = extract_hash_map(
        expect_list(manifest.get("input_hashes"), "archive manifest input_hashes")
    )
    handoff_hashes = extract_hash_map(
        expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes")
    )
    if manifest_hashes != handoff_hashes:
        raise ValueError(
            refusal(
                "archive manifest input hashes do not match handoff metadata",
                found=manifest_hashes,
                expected=handoff_hashes,
            )
        )
    manifest_archive = expect_scalar(manifest, "archive", str)
    record_archive = expect_scalar(archive, "path", str)
    if manifest_archive != record_archive:
        raise ValueError(
            refusal(
                "handoff archive manifest archive path does not match the resource record",
                found=manifest_archive,
                expected=record_archive,
            )
        )
    file_members, member_paths, member_types = _manifest_members(manifest)
    required = {*SEALED_INPUTS, HANDOFF_METADATA, RUN_METADATA}
    if missing := required - member_paths:
        raise ValueError(
            "handoff archive manifest lacks required members: "
            + ", ".join(sorted(missing))
        )
    if non_files := required - set(file_members):
        raise ValueError(
            "handoff archive manifest requires regular-file members: "
            + ", ".join(sorted(non_files))
        )
    _validate_manifest_inputs(file_members, handoff_metadata, manifest)
    archive_path = resolve_run_archive_path(root, expect_scalar(archive, "path", str))
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundled: dict[str, tarfile.TarInfo] = {}
        for member in bundle.getmembers():
            logical = logical_tar_member_name(member.name, root.name)
            if logical is None:
                continue
            bundled[logical.rstrip("/")] = member
        allowed_extra = {
            path.rstrip("/")
            for path in expect_list(
                manifest.get("archive_members_excluded_from_self_listing"),
                "archive manifest excluded self members",
            )
        }
        manifest_member_paths = {path.rstrip("/") for path in member_paths}
        if set(bundled) != manifest_member_paths | allowed_extra:
            raise ValueError(
                refusal(
                    "archive members do not match the archive manifest",
                    found=sorted(bundled),
                    expected=sorted(manifest_member_paths | allowed_extra),
                )
            )
        _validate_bundled_types(bundle, bundled, member_types)
        _validate_bundled_hashes(bundle, bundled, file_members)


def _manifest_members(manifest: Any) -> Any:
    """Read unique archive-member paths, declared types and regular-file identities."""
    members = expect_list(manifest.get("members"), "handoff archive manifest members")
    member_paths: set[str] = set()
    member_types: dict[str, str] = {}
    file_members: dict[str, tuple[int, str]] = {}
    for entry in members:
        data = expect_mapping(entry, "handoff archive manifest member")
        path = expect_scalar(data, "path", str)
        if not path or path in member_paths:
            raise ValueError(
                refusal(
                    "handoff archive manifest members must have unique non-empty paths",
                    found=path,
                )
            )
        member_paths.add(path)
        member_type = expect_scalar(data, "type", str)
        if member_type not in {"file", "directory"}:
            raise ValueError(
                refusal(
                    f"handoff archive manifest member {path} type is invalid",
                    found=member_type,
                    expected="file or directory",
                )
            )
        member_types[path] = member_type
        if member_type == "file":
            bytes_value = expect_scalar(data, "bytes", int)
            sha_value = expect_scalar(data, "sha256", str)
            file_members[path] = (bytes_value, sha_value)
    return file_members, member_paths, member_types


def _validate_manifest_inputs(
    file_members: Any, handoff_metadata: Any, manifest: Any
) -> None:
    """Require each sealed input hash to identify an exact regular archive member."""
    manifest_input_hashes = extract_hash_map(
        expect_list(manifest.get("input_hashes"), "archive manifest input_hashes")
    )
    handoff_input_hashes = extract_hash_map(
        expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes")
    )
    for path, expected in handoff_input_hashes.items():
        if manifest_input_hashes.get(path) != expected:
            raise ValueError(
                refusal(
                    f"archive manifest input hash mismatch for {path}",
                    found=manifest_input_hashes.get(path),
                    expected=expected,
                )
            )
        if file_members.get(path) != expected:
            raise ValueError(
                f"archive manifest must declare input {path} as an exact regular file."
            )


def _validate_bundled_types(
    bundle: tarfile.TarFile, bundled: dict[str, tarfile.TarInfo], member_types: Any
) -> None:
    """Validate sealed archive member types."""
    for path, member_type in member_types.items():
        member = bundled.get(path.rstrip("/"))
        if member is None:
            raise ValueError(f"archive member is missing: {path}.")
        if member_type == "file" and not member.isfile():
            raise ValueError(f"archive member must be a regular file: {path}.")
        if member_type == "directory" and not member.isdir():
            raise ValueError(f"archive member must be a directory: {path}.")


def _validate_bundled_hashes(
    bundle: tarfile.TarFile, bundled: dict[str, tarfile.TarInfo], file_members: Any
) -> None:
    """Validate sealed archive member byte counts and hashes."""
    for path, (expected_bytes, expected_sha) in file_members.items():
        member = bundled.get(path)
        if member is None or not member.isfile() or member.size != expected_bytes:
            raise ValueError(
                refusal(
                    f"archive member size mismatch for {path}",
                    found=None
                    if member is None or not member.isfile()
                    else member.size,
                    expected=expected_bytes,
                )
            )
        handle = bundle.extractfile(member)
        if handle is None:
            raise ValueError(f"archive member {path} cannot be read.")
        digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected_sha:
            raise ValueError(
                refusal(
                    f"archive member hash mismatch for {path}",
                    found=digest,
                    expected=expected_sha,
                )
            )
