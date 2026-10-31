"""Check the handoff tar archive against its manifest.

The archive is what was sent to the machine that ran Palace. The manifest lists every
member with size and SHA-256. A run is only trusted if the archive on disk is
exactly what the manifest says and the manifest's inputs equal the handoff
metadata's.
"""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..execution.archive import logical_tar_member_name, resolve_run_archive_path
from ._artifact_files import _extract_hash_map
from ._expect import _expect_list, _expect_mapping, _expect_scalar
from ._run_documents import ArchiveManifest, HandoffMetadata, ResourceRecord


@dataclass(frozen=True)
class _ManifestMembers:
    """Manifest member listing, indexed the three ways the checks need it."""

    paths: set[str]
    types: dict[str, str]  # path -> "file" | "directory"
    files: dict[str, tuple[int, str]]  # path -> (bytes, sha256), files only


def _validate_archive_manifest(
    root: Path,
    manifest: ArchiveManifest,
    resource_record: ResourceRecord,
    handoff_metadata: HandoffMetadata,
) -> None:
    archive = _check_manifest_header(manifest, resource_record, handoff_metadata)
    members = _read_manifest_members(manifest)
    _check_manifest_inputs(manifest, handoff_metadata, members)
    _check_archive_matches_manifest(root, manifest, archive, members)


def _check_manifest_header(
    manifest: ArchiveManifest,
    resource_record: ResourceRecord,
    handoff_metadata: HandoffMetadata,
) -> dict[str, Any]:
    """Return the resource record's archive entry after the manifest agrees with it."""
    if _expect_scalar(manifest, "schema_version", int, fallback=0) != 1:
        raise ValueError("handoff archive manifest schema_version must be integer 1.")
    archive = _expect_mapping(resource_record.get("archive"), "resource_record.archive")
    if _extract_hash_map(
        _expect_list(manifest.get("input_hashes"), "archive manifest input_hashes")
    ) != _extract_hash_map(
        _expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes")
    ):
        raise ValueError("archive manifest input hashes do not match handoff metadata.")
    if _expect_scalar(manifest, "archive", str) != _expect_scalar(archive, "path", str):
        raise ValueError(
            "handoff archive manifest archive path does not match receipt."
        )
    return archive


def _read_manifest_members(manifest: ArchiveManifest) -> _ManifestMembers:
    members = _expect_list(manifest.get("members"), "handoff archive manifest members")
    member_paths: set[str] = set()
    member_types: dict[str, str] = {}
    file_members: dict[str, tuple[int, str]] = {}
    for entry in members:
        data = _expect_mapping(entry, "handoff archive manifest member")
        path = _expect_scalar(data, "path", str)
        if not path or path in member_paths:
            raise ValueError("handoff archive manifest members must have unique paths.")
        member_paths.add(path)
        member_type = _expect_scalar(data, "type", str)
        if member_type not in {"file", "directory"}:
            raise ValueError("handoff archive manifest member type is invalid.")
        member_types[path] = member_type
        if member_type == "file":
            bytes_value = _expect_scalar(data, "bytes", int)
            sha_value = _expect_scalar(data, "sha256", str)
            file_members[path] = (bytes_value, sha_value)
    return _ManifestMembers(paths=member_paths, types=member_types, files=file_members)


def _check_manifest_inputs(
    manifest: ArchiveManifest,
    handoff_metadata: HandoffMetadata,
    members: _ManifestMembers,
) -> None:
    """The files needed to rerun are listed as regular files with the hashed bytes."""
    member_paths = members.paths
    file_members = members.files
    required = {
        "config.json",
        "palace.msh",
        "metadata/mesh_manifest.json",
        "metadata/palace_index_map.json",
        "metadata/palace_handoff_metadata.json",
        "metadata/palace_run_metadata.json",
    }
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
    manifest_input_hashes = _extract_hash_map(
        _expect_list(manifest.get("input_hashes"), "archive manifest input_hashes")
    )
    handoff_input_hashes = _extract_hash_map(
        _expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes")
    )
    for path, expected in handoff_input_hashes.items():
        if manifest_input_hashes.get(path) != expected:
            raise ValueError(f"archive manifest input hash mismatch for {path}.")
        if file_members.get(path) != expected:
            raise ValueError(
                f"archive manifest must declare input {path} as an exact regular file."
            )


def _check_archive_matches_manifest(
    root: Path,
    manifest: ArchiveManifest,
    archive: dict[str, Any],
    members: _ManifestMembers,
) -> None:
    """Open the tar and compare member names, kinds, sizes and hashes."""
    member_paths = members.paths
    member_types = members.types
    file_members = members.files
    archive_path = resolve_run_archive_path(root, _expect_scalar(archive, "path", str))
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundled: dict[str, tarfile.TarInfo] = {}
        for member in bundle.getmembers():
            logical = logical_tar_member_name(member.name, root.name)
            if logical is None:
                continue
            bundled[logical.rstrip("/")] = member
        allowed_extra = {
            path.rstrip("/")
            for path in _expect_list(
                manifest.get("archive_members_excluded_from_self_listing"),
                "archive manifest excluded self members",
            )
        }
        manifest_member_paths = {path.rstrip("/") for path in member_paths}
        if set(bundled) != manifest_member_paths | allowed_extra:
            raise ValueError("archive members do not match the archive manifest.")
        for path, member_type in member_types.items():
            member = bundled.get(path.rstrip("/"))
            if member is None:
                raise ValueError(f"archive member is missing: {path}.")
            if member_type == "file" and not member.isfile():
                raise ValueError(f"archive member must be a regular file: {path}.")
            if member_type == "directory" and not member.isdir():
                raise ValueError(f"archive member must be a directory: {path}.")
        for path, (expected_bytes, expected_sha) in file_members.items():
            member = bundled.get(path)
            if member is None or not member.isfile() or member.size != expected_bytes:
                raise ValueError(f"archive member size mismatch for {path}.")
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError(f"archive member {path} cannot be read.")
            digest = hashlib.sha256(handle.read()).hexdigest()
            if digest != expected_sha:
                raise ValueError(f"archive member hash mismatch for {path}.")
