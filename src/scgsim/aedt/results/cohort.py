"""Check that a completed receipt belongs to the preparation it claims.

Owns: the preparation cohort (the handoff metadata, the handoff manifest and every
file the manifest lists) and the hash chain tying them to the completed receipt.
It does not read solver evidence from the receipt; the ``*_readback.py`` modules do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import (
    canonical_handoff_paths,
    canonical_member_paths,
    validate_hfss_import_source,
)
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    initial_receipt_payload,
    initial_receipt_sha256,
    validate_runtime_source,
)
from scgsim.aedt.results.readback_common import (
    contained_in_handoff_root,
    is_sha256_hex,
)
from scgsim.aedt.specs.hfss_prepared import HfssDrivenGeometrySpec, HfssEprSpec

AEDT_MODES = {"terminal", "modal", "eigenmode", "q3d", "q2d"}


def validate_completion_cohort(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    """Verify the immutable preparation cohort and complete execution provenance."""
    _check_receipt_schema_and_mode(receipt)
    metadata_path = contained_in_handoff_root(
        root, "metadata/aedt_handoff_metadata.json"
    )
    manifest_path = contained_in_handoff_root(
        root, "metadata/aedt_handoff_manifest.json"
    )
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("completed v3 receipt lacks preparation cohort files")
    metadata = read_json(metadata_path)
    manifest = read_json(manifest_path)
    _check_preparation_metadata(metadata, manifest, receipt["mode"], spec)
    prepared_receipt_sha = _check_hash_bindings(
        receipt, metadata, manifest, manifest_path
    )
    _check_manifest_members(root, manifest, spec, prepared_receipt_sha)
    validate_hfss_import_source(root, read_json(root / "aedt_spec.json"), spec)


def _check_receipt_schema_and_mode(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_V3:
        raise RuntimeError("completed receipt cohort schema is invalid")
    if receipt.get("expected_receipt_schema") != RECEIPT_V3:
        raise RuntimeError("completed v3 receipt expectation marker is invalid")
    if receipt.get("mode") not in AEDT_MODES:
        raise RuntimeError("AEDT receipt mode is invalid")


def _check_preparation_metadata(
    metadata: Any, manifest: Any, mode: str, spec: Any
) -> None:
    """Check the handoff metadata and manifest headers and the file map."""
    # Body-first (EPR and driven-terminal) preparations use the v2 handoff schemas.
    body_first = isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec))
    expected_manifest_schema = (
        "scgsim.aedt.handoff-manifest.v2"
        if body_first
        else "scgsim.aedt.handoff-manifest.v1"
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version")
        != ("scgsim.aedt.handoff.v2" if body_first else "scgsim.aedt.handoff.v1")
        or metadata.get("status") != "prepared"
        or metadata.get("mode") != mode
        or not isinstance(manifest, dict)
        or manifest.get("schema_version") != expected_manifest_schema
    ):
        raise RuntimeError("completed v3 preparation cohort is invalid")

    expected_files = {
        "spec": "aedt_spec.json",
        "receipt": "metadata/aedt_run_receipt.json",
    }
    if mode not in {"q2d", "q3d"} and not body_first:
        expected_files["gds"] = "geometry/design.gds"
    if (
        isinstance(spec, HfssDrivenGeometrySpec)
        and metadata.get("workflow") != "body_first_driven_terminal"
    ):
        raise RuntimeError("body-first Driven Terminal preparation workflow is invalid")
    if isinstance(spec, HfssEprSpec) and metadata.get("workflow") not in {
        "body_first_eigenmode",
        "epr",
    }:
        raise RuntimeError("body-first Eigenmode preparation workflow is invalid")
    if metadata.get("files") != expected_files:
        raise RuntimeError("completed v3 preparation file map is not canonical")


def _check_hash_bindings(
    receipt: dict[str, Any],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> str:
    """Check markers and receipt-to-manifest hashes; return the initial receipt hash."""
    if receipt.get("preparation_cohort") != "prepared_v3":
        raise RuntimeError("completed v3 preparation cohort is invalid")
    if (
        metadata.get("expected_receipt_schema") != RECEIPT_V3
        or manifest.get("expected_receipt_schema") != RECEIPT_V3
    ):
        raise RuntimeError("completed v3 preparation markers are inconsistent")
    validate_runtime_source(receipt.get("prepared_runtime_source"), stage="prepared")

    validate_runtime_source(receipt.get("runtime_source"), stage="actual")
    prepared_receipt_sha = receipt.get("prepared_receipt_sha256")
    verified_manifest_sha = receipt.get("verified_prepared_manifest_sha256")
    if (
        not isinstance(prepared_receipt_sha, str)
        or not is_sha256_hex(prepared_receipt_sha)
        or not isinstance(verified_manifest_sha, str)
        or not is_sha256_hex(verified_manifest_sha)
        or file_sha256(manifest_path) != verified_manifest_sha
        or _recomputed_initial_receipt_sha256(receipt) != prepared_receipt_sha
    ):
        raise RuntimeError("completed v3 preparation hash bindings are invalid")
    return prepared_receipt_sha


def _recomputed_initial_receipt_sha256(receipt: dict[str, Any]) -> str:
    """Hash the receipt as it was at preparation time, before any output existed.

    The completed receipt has grown since; rebuilding the initial payload from the
    fields that existed then must reproduce the hash the manifest recorded.
    """
    initial = initial_receipt_payload(
        schema_version=RECEIPT_V3,
        mode=receipt.get("mode"),
        requested=receipt.get("requested"),
        pdk_materials=receipt.get("pdk_materials"),
        vacuum_material_id=receipt.get("vacuum_material_id"),
        source=receipt.get("source"),
        prepared_runtime_source_value=receipt.get("prepared_runtime_source"),
        outputs={},
        prepared_at_utc=receipt.get("prepared_at_utc"),
    )
    return initial_receipt_sha256(initial)


def _check_manifest_members(
    root: Path, manifest: dict[str, Any], spec: Any, prepared_receipt_sha: str
) -> None:
    """Every manifest member must exist on disk with the recorded size and hash."""
    expected_paths = canonical_handoff_paths(spec)
    if manifest.get("allowed_paths") != expected_paths:
        raise RuntimeError("completed v3 manifest allowed paths are not canonical")
    members = manifest.get("members")
    # The manifest cannot list its own hash, so it is allowed but not a member.
    if (
        not isinstance(members, list)
        or len(members) != len(expected_paths) - 1
        or [item.get("path") if isinstance(item, dict) else None for item in members]
        != canonical_member_paths(spec)
    ):
        raise RuntimeError("completed v3 manifest members are not canonical")
    for member in members:
        _check_member_record(member)
        relative = member["path"]
        if relative == "metadata/aedt_run_receipt.json":
            # The run receipt changes during the run, so only its initial hash is bound.
            if member["sha256"] != prepared_receipt_sha:
                raise RuntimeError("completed v3 initial receipt binding is invalid")
            continue
        path = contained_in_handoff_root(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != member["bytes"]
            or file_sha256(path) != member["sha256"]
        ):
            raise RuntimeError(f"completed v3 manifest member mismatch: {relative}")


def _check_member_record(member: Any) -> None:
    if (
        not isinstance(member, dict)
        or set(member) != {"path", "bytes", "sha256"}
        or not isinstance(member.get("bytes"), int)
        or isinstance(member.get("bytes"), bool)
        or member["bytes"] < 0
        or not isinstance(member.get("sha256"), str)
        or not is_sha256_hex(member["sha256"])
    ):
        raise RuntimeError("completed v3 manifest member is invalid")
