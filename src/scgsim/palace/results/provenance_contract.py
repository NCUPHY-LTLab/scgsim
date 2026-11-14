"""Validate handoff identity, recorded hashes and exact archived input members."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path
from typing import Any, NamedTuple

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
    RETURNED_RUN_FOLDER,
    logical_tar_member_name,
    resolve_run_archive_path,
)
from scgsim.refusal import Declared, found_text


def validate_scalar_identity(
    handoff_metadata: dict[str, Any], run_metadata: dict[str, Any]
) -> None:
    """Match shared handoff and run fields before checking artifact identities."""
    _validate_source_identity(handoff_metadata, run_metadata)
    _validate_shared_handoff(handoff_metadata, run_metadata)


def _validate_source_identity(
    handoff_metadata: dict[str, Any], run_metadata: dict[str, Any]
) -> None:
    """Both records name the same source revisions, and each declares a valid Palace identity."""
    handoff_identity = handoff_metadata.get("source_revisions")
    run_identity = run_metadata.get("source_revisions")
    if not isinstance(handoff_identity, dict) or not handoff_identity:
        raise ValueError(
            Declared(
                "handoff source_revisions must be a non-empty mapping",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(handoff_identity))
        )
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError(
            Declared(
                "run source_revisions must be a non-empty mapping", RETURNED_RUN_FOLDER
            ).refusal(found_text(run_identity))
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
            Declared(
                "manual handoff identity mismatch between handoff and run metadata",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(run_identity, handoff_identity))
        )


def _validate_shared_handoff(
    handoff_metadata: dict[str, Any], run_metadata: dict[str, Any]
) -> None:
    """Both records name the same Palace identity, handoff id and Route-A film."""
    handoff_palace_identity = handoff_metadata.get("palace_identity")
    run_palace_identity = run_metadata.get("palace_identity")
    if handoff_palace_identity != run_palace_identity:
        raise ValueError(
            Declared(
                "palace identity mismatch between handoff and run metadata",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(run_palace_identity, handoff_palace_identity))
        )

    handoff_id = handoff_metadata.get("handoff_id")
    run_id = run_metadata.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        raise ValueError(
            Declared(
                "handoff metadata must include handoff_id", RETURNED_RUN_FOLDER
            ).refusal(found_text(handoff_id))
        )
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(
            Declared(
                "run metadata must include handoff_id", RETURNED_RUN_FOLDER
            ).refusal(found_text(run_id))
        )
    if handoff_id != run_id:
        raise ValueError(
            Declared(
                "handoff id mismatch between handoff and run metadata",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(run_id, handoff_id))
        )
    if handoff_metadata.get("route_a_thin_film") != run_metadata.get(
        "route_a_thin_film"
    ):
        raise ValueError(
            Declared(
                "Route-A thin-film identity mismatch between handoff and run metadata",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    run_metadata.get("route_a_thin_film"),
                    handoff_metadata.get("route_a_thin_film"),
                )
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
            Declared(
                "recomputed handoff identity does not match expected handoff_id; the sealed "
                "inputs or the recorded handoff_id changed",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    {"recomputed": computed, "recorded": recorded}, expected_handoff_id
                )
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
        # The convergence runner is part of the execution only when the handoff declared convergence.
        **(
            {"runner": execution.get("runner")}
            if "convergence_runner" in handoff_metadata
            else {}
        ),
    }
    if execution != recorded:
        raise ValueError(
            Declared(
                "handoff execution identity does not match handoff metadata",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(execution, recorded))
        )
    files = [
        ("script", expect_scalar(handoff_metadata, "script", str)),
        ("recorder", expect_scalar(handoff_metadata, "returned_receipt_recorder", str)),
    ]
    if "convergence_runner" in handoff_metadata:
        files.append(
            ("runner", expect_scalar(handoff_metadata, "convergence_runner", str))
        )
    for field, relative in files:
        _validate_execution_file(root, execution, field, relative)


def _validate_execution_file(
    root: Path, execution: dict[str, Any], field: str, relative: str
) -> None:
    """One file of the execution identity sits at its recorded path with its recorded hash."""
    item = expect_mapping(execution.get(field), f"execution identity {field}")
    recorded_path = expect_scalar(item, "path", str)
    if recorded_path != relative:
        raise ValueError(
            Declared(
                f"handoff execution identity {field} path is inconsistent",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(recorded_path, relative))
        )
    path = confined_path(root, relative)
    recorded_sha = expect_scalar(item, "sha256", str) if path.is_file() else None
    if not path.is_file() or recorded_sha != sha256(path):
        raise ValueError(
            Declared(
                f"handoff execution identity {field} hash is inconsistent for {relative}",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    sha256(path) if path.is_file() else "no file",
                    item.get("sha256"),
                )
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
                Declared(
                    f"resource record {field} does not match handoff metadata",
                    RETURNED_RUN_FOLDER,
                ).refusal(
                    found_text(resource_record.get(field), handoff_metadata.get(field))
                )
            )
    status = expect_scalar(resource_record, "status", str)
    if status != "completed":
        raise ValueError(
            Declared(
                "resource record status must be completed for a returned run",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(status, "completed"))
        )
    # Which exit code a completed run has depends on how its solver ended: the receipt's check decides it,
    # and the resource record must agree with the receipt (``validate_returned_state_agreement``).
    expect_int(resource_record.get("exit_code"), "resource record exit_code")


def validate_hashes(root: Path, metadata: dict[str, Any], label: str) -> None:
    """Verify required input identities against the exact run-directory bytes."""
    hashes = expect_list(metadata.get("hashes"), f"{label} metadata hashes")
    if not hashes:
        raise ValueError(
            Declared(f"{label} metadata", RETURNED_RUN_FOLDER).refusal(
                "has no hash entries"
            )
        )

    observed: dict[str, FileHash] = {}
    for entry in hashes:
        if not isinstance(entry, dict):
            raise TypeError(
                Declared(
                    f"{label} metadata hashes must be mapping entries",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(entry))
            )
        path_name = expect_scalar(entry, "path", str)
        bytes_value = expect_int(
            entry.get("bytes"), f"{label} metadata hash entry {path_name!r} bytes"
        )
        sha_value = expect_scalar(entry, "sha256", str)
        file_path = confined_path(root, path_name)
        if not file_path.is_file():
            raise FileNotFoundError(
                Declared(f"{label} metadata", RETURNED_RUN_FOLDER).refusal(
                    f"references missing artifact {path_name}"
                )
            )

        observed[path_name] = FileHash(
            expect_int(file_path.stat().st_size, f"size of {path_name}"),
            sha256(file_path),
        )
        if bytes_value != observed[path_name].bytes:
            raise ValueError(
                Declared(f"{label} hash byte size", RETURNED_RUN_FOLDER).refusal(
                    f"mismatch for {path_name}: declared {bytes_value}, observed {observed[path_name].bytes}"
                )
            )
        if sha_value != observed[path_name].sha256:
            raise ValueError(
                Declared(
                    f"{label} hash mismatch for {path_name}", RETURNED_RUN_FOLDER
                ).refusal(found_text(observed[path_name].sha256, sha_value))
            )

    missing = [name for name in SEALED_INPUTS if name not in observed]
    if missing:
        raise ValueError(
            Declared(f"{label} metadata", RETURNED_RUN_FOLDER).refusal(
                f"missing hash entries for: {', '.join(missing)}"
            )
        )


def validate_archive_record(root: Path, resource_record: dict[str, Any]) -> None:
    """Verify the retained compressed handoff archive against its resource record."""
    if resource_record.get("kind") != "manual_handoff":
        raise ValueError(
            Declared(
                "resource record kind must be manual_handoff", RETURNED_RUN_FOLDER
            ).refusal(found_text(resource_record.get("kind"), "manual_handoff"))
        )

    if expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError(
            Declared(
                "resource record status must be a non-empty string", RETURNED_RUN_FOLDER
            ).refusal(found_text(resource_record.get("status")))
        )
    requested_resources = optional_dict(resource_record.get("requested_resources"))
    resolved_resources = optional_dict(resource_record.get("resolved_resources"))
    if not requested_resources:
        raise ValueError(
            Declared(
                "resource record requested_resources must be a non-empty mapping",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(resource_record.get("requested_resources")))
        )
    if not resolved_resources:
        raise ValueError(
            Declared(
                "resource record resolved_resources must be a non-empty mapping",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(resource_record.get("resolved_resources")))
        )

    archive = optional_dict(resource_record.get("archive"))
    if not archive:
        raise ValueError(
            Declared(
                "resource record archive must be a non-empty mapping",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(resource_record.get("archive")))
        )
    path_name = expect_scalar(archive, "path", str)
    expected_sha = expect_scalar(archive, "sha256", str)
    archive_path = resolve_run_archive_path(root, path_name)
    if not archive_path.is_file():
        raise FileNotFoundError(
            Declared("resource record", RETURNED_RUN_FOLDER).refusal(
                f"references missing archive payload {archive_path}"
            )
        )
    archive_sha = sha256(archive_path)
    if archive_sha != expected_sha:
        raise ValueError(
            Declared(
                f"resource record archive hash mismatch for {archive_path}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(archive_sha, expected_sha))
        )


def validate_archive_manifest(
    root: Path,
    manifest: dict[str, Any],
    resource_record: dict[str, Any],
    handoff_metadata: dict[str, Any],
) -> None:
    """Validate the archive inventory before comparing its members and exact file bytes."""
    archive = _validate_manifest_header(manifest, resource_record, handoff_metadata)
    members = _manifest_members(manifest)
    required = {*SEALED_INPUTS, HANDOFF_METADATA, RUN_METADATA}
    if missing := required - members.paths:
        raise ValueError(
            Declared("handoff archive manifest", RETURNED_RUN_FOLDER).refusal(
                "lacks required members: " + ", ".join(sorted(missing))
            )
        )
    if non_files := required - set(members.files):
        raise ValueError(
            Declared("handoff archive manifest", RETURNED_RUN_FOLDER).refusal(
                "requires regular-file members: " + ", ".join(sorted(non_files))
            )
        )
    _validate_manifest_inputs(members.files, handoff_metadata, manifest)
    archive_path = resolve_run_archive_path(root, expect_scalar(archive, "path", str))
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundled = _bundled_members(bundle, root)
        allowed_extra = {
            path.rstrip("/")
            for path in expect_list(
                manifest.get("archive_members_excluded_from_self_listing"),
                "archive manifest excluded self members",
            )
        }
        manifest_member_paths = {path.rstrip("/") for path in members.paths}
        if set(bundled) != manifest_member_paths | allowed_extra:
            raise ValueError(
                Declared(
                    "archive members do not match the archive manifest",
                    RETURNED_RUN_FOLDER,
                ).refusal(
                    found_text(
                        sorted(bundled), sorted(manifest_member_paths | allowed_extra)
                    )
                )
            )
        _validate_bundled_types(bundle, bundled, members.types)
        _validate_bundled_hashes(bundle, bundled, members.files)


def _validate_manifest_header(
    manifest: dict[str, Any],
    resource_record: dict[str, Any],
    handoff_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Check the manifest's schema, input hashes and archive path; return the resource record's archive."""
    schema_version = expect_scalar(manifest, "schema_version", int, fallback=0)
    if schema_version != RUN_RECORD_SCHEMA_VERSION:
        raise ValueError(
            Declared(
                "handoff archive manifest schema_version must be integer "
                f"{RUN_RECORD_SCHEMA_VERSION}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(schema_version))
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
            Declared(
                "archive manifest input hashes do not match handoff metadata",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(manifest_hashes, handoff_hashes))
        )
    manifest_archive = expect_scalar(manifest, "archive", str)
    record_archive = expect_scalar(archive, "path", str)
    if manifest_archive != record_archive:
        raise ValueError(
            Declared(
                "handoff archive manifest archive path does not match the resource record",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(manifest_archive, record_archive))
        )
    return archive


def _bundled_members(bundle: tarfile.TarFile, root: Path) -> dict[str, tarfile.TarInfo]:
    """The archive's members by their path inside the run folder; the folder's own entry is left out."""
    bundled: dict[str, tarfile.TarInfo] = {}
    for member in bundle.getmembers():
        logical = logical_tar_member_name(member.name, root.name)
        if logical is None:
            continue
        bundled[logical.rstrip("/")] = member
    return bundled


class _ManifestMembers(NamedTuple):
    """The archive manifest's members: each regular file's size and SHA-256, every path, and its type."""

    files: dict[str, tuple[int, str]]
    paths: set[str]
    types: dict[str, str]


def _manifest_members(manifest: Any) -> _ManifestMembers:
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
                Declared(
                    "handoff archive manifest members must have unique non-empty paths",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(path))
            )
        member_paths.add(path)
        member_type = expect_scalar(data, "type", str)
        if member_type not in {"file", "directory"}:
            raise ValueError(
                Declared(
                    f"handoff archive manifest member {path} type is invalid",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(member_type, "file or directory"))
            )
        member_types[path] = member_type
        if member_type == "file":
            bytes_value = expect_scalar(data, "bytes", int)
            sha_value = expect_scalar(data, "sha256", str)
            file_members[path] = (bytes_value, sha_value)
    return _ManifestMembers(file_members, member_paths, member_types)


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
                Declared(
                    f"archive manifest input hash mismatch for {path}",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(manifest_input_hashes.get(path), expected))
            )
        if file_members.get(path) != expected:
            raise ValueError(
                Declared("archive manifest", RETURNED_RUN_FOLDER).refusal(
                    f"must declare input {path} as an exact regular file"
                )
            )


def _validate_bundled_types(
    bundle: tarfile.TarFile, bundled: dict[str, tarfile.TarInfo], member_types: Any
) -> None:
    """Validate sealed archive member types."""
    for path, member_type in member_types.items():
        member = bundled.get(path.rstrip("/"))
        if member is None:
            raise ValueError(
                Declared("archive member", RETURNED_RUN_FOLDER).refusal(
                    f"is missing: {path}"
                )
            )
        if member_type == "file" and not member.isfile():
            raise ValueError(
                Declared("archive member", RETURNED_RUN_FOLDER).refusal(
                    f"must be a regular file: {path}"
                )
            )
        if member_type == "directory" and not member.isdir():
            raise ValueError(
                Declared("archive member", RETURNED_RUN_FOLDER).refusal(
                    f"must be a directory: {path}"
                )
            )


def _validate_bundled_hashes(
    bundle: tarfile.TarFile, bundled: dict[str, tarfile.TarInfo], file_members: Any
) -> None:
    """Validate sealed archive member byte counts and hashes."""
    for path, (expected_bytes, expected_sha) in file_members.items():
        member = bundled.get(path)
        if member is None or not member.isfile() or member.size != expected_bytes:
            raise ValueError(
                Declared(
                    f"archive member size mismatch for {path}", RETURNED_RUN_FOLDER
                ).refusal(
                    found_text(
                        None if member is None or not member.isfile() else member.size,
                        expected_bytes,
                    )
                )
            )
        handle = bundle.extractfile(member)
        if handle is None:
            raise ValueError(
                Declared(f"archive member {path}", RETURNED_RUN_FOLDER).refusal(
                    "cannot be read"
                )
            )
        digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected_sha:
            raise ValueError(
                Declared(
                    f"archive member hash mismatch for {path}", RETURNED_RUN_FOLDER
                ).refusal(found_text(digest, expected_sha))
            )
