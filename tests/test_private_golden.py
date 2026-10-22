"""Owner-authorized AEDT returned-data checks; never construct or run a model.

Supply --private-golden-manifest PATH. The external JSON manifest contains
run_dir, operation_kind (aedt_one_shot, aedt_batch_collection or aedt_saved_epr),
files (the complete run-relative file-to-SHA256 mapping), and expected_selected
(an owner-selected exact JSON subset of the reader output below). Optional
expected_epr_record and expected_physics_csv select hash-bound raw files for
field/value preservation, independently read before the public reader. For saved EPR,
include epr_result_path. Optional expected_handoff_metadata_sha256 binds the real
canonical metadata file; no synthetic handoff ID is introduced.

The selected output has operation_kind, receipt, mode, physics, epr and
handoff_metadata_sha256. Dictionary keys select a subset; lists retain their
exact length/order. Numeric comparisons are exact, with no agent-chosen tolerance.
The manifest is an input, not a grant or a security sandbox. Data ownership and
read permission must be established separately before invoking this test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest


pytestmark = pytest.mark.private_golden


def _inside(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or part.as_posix() != relative:
        raise ValueError("private fixture member path is not canonical and run-relative")
    path = (root / part).resolve()
    path.relative_to(root)
    return path


def _file_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest, stat.st_size, stat.st_mtime_ns


def _inventory(root: Path) -> dict[str, tuple[str, int, int]]:
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("private fixture must provide regular copied files, not symlinks")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = _file_identity(path)
    return files



def _verify_declared_files(root: Path, declared: dict) -> dict[str, tuple[str, int, int]]:
    present = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if set(declared) - present:
        raise AssertionError("private fixture required file is missing")
    if present - set(declared):
        raise AssertionError("private fixture contains undeclared files")
    before = {}
    for name, digest in declared.items():
        path = _inside(root, name)
        if path.is_symlink() or (root / name).is_symlink():
            raise ValueError("private fixture must provide regular copied files, not symlinks")
        identity = _file_identity(path)
        if identity[0] != digest:
            raise AssertionError("private fixture content SHA256 differs")
        before[name] = identity
    return before


def _verify_metadata_identity(before: dict, expected: str) -> None:
    identity = before.get("metadata/aedt_handoff_metadata.json")
    if identity is None or identity[0] != expected:
        raise AssertionError("private fixture exact handoff metadata identity differs")


def _assert_epr_preserved(actual, expected) -> None:
    _assert_subset(actual, expected, "raw EPR record", raw_record=True)
    identities = [(row["mode"], row.get("native_pass")) for row in actual["rows"]]
    if len(set(identities)) != len(identities):
        raise AssertionError("returned EPR mode/pass identities repeat")
    for row in actual["rows"]:
        groups = [item["group_id"] for item in row.get("surface_contributions", []) if "group_id" in item]
        if len(set(groups)) != len(groups):
            raise AssertionError("returned EPR group identities repeat within mode/pass")


def _assert_subset(actual, selected, location="reader", *, raw_record=False) -> None:
    if isinstance(selected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"owner-selected mapping is absent at {location}")
        for key, value in selected.items():
            if key not in actual:
                raise AssertionError(f"owner-selected member is absent at {location}")
            _assert_subset(actual[key], value, f"{location}.{key}", raw_record=raw_record)
    elif isinstance(selected, list):
        if not isinstance(actual, list) or len(actual) != len(selected):
            raise AssertionError(f"owner-selected list differs at {location}")
        for index, (observed, expected) in enumerate(zip(actual, selected, strict=True)):
            _assert_subset(observed, expected, f"{location}[{index}]", raw_record=raw_record)
    elif raw_record and isinstance(selected, float) and math.isnan(selected):
        # Native history may retain NaN padding as an unavailable observation.
        if type(actual) is not type(selected) or not math.isnan(actual):
            raise AssertionError(f"raw missing-value marker differs at {location}")
    elif type(actual) is not type(selected) or actual != selected:
        raise AssertionError(f"owner-selected value differs at {location}")


def test_private_returned_run(private_golden_manifest: Path) -> None:
    """Use public strict readers and owner-provided expectations on immutable files."""
    manifest_before = _file_identity(private_golden_manifest)
    manifest = json.loads(private_golden_manifest.read_text(encoding="utf-8"))
    root = Path(manifest["run_dir"]).expanduser()
    if not root.is_absolute():
        root = private_golden_manifest.parent / root
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("private fixture run_dir is not a directory")
    # The manifest lives outside the run so its own hash is not self-referential.
    if private_golden_manifest.is_relative_to(root):
        raise ValueError("private fixture manifest must be outside the retained run")
    declared = manifest["files"]
    if not isinstance(declared, dict) or not declared:
        raise ValueError("private fixture requires a complete nonempty file/SHA256 mapping")
    for relative, digest in declared.items():
        _inside(root, relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("private fixture SHA256 must be owner-provided hexadecimal text")
        bytes.fromhex(digest)
    before = _verify_declared_files(root, declared)
    selected = manifest["expected_selected"]
    if not isinstance(selected, dict) or not selected:
        raise ValueError("private fixture requires a nonempty owner-selected result subset")
    kind = manifest["operation_kind"]
    if kind not in {"aedt_one_shot", "aedt_batch_collection", "aedt_saved_epr"}:
        raise ValueError("private fixture operation_kind is unsupported")
    metadata_relative = "metadata/aedt_handoff_metadata.json"
    metadata_sha = before.get(metadata_relative, (None, 0, 0))[0]
    if "expected_handoff_metadata_sha256" in manifest:
        _verify_metadata_identity(before, manifest["expected_handoff_metadata_sha256"])
    raw_epr = None
    raw_physics = None
    if "expected_epr_record" in manifest:
        relative = manifest["expected_epr_record"]
        if relative not in declared:
            raise ValueError("expected EPR record must be in the bound fixture inventory")
        raw_epr = json.loads(_inside(root, relative).read_text(encoding="utf-8"))
    if "expected_physics_csv" in manifest:
        relative = manifest["expected_physics_csv"]
        if relative not in declared:
            raise ValueError("expected physics record must be in the bound fixture inventory")
        with _inside(root, relative).open(encoding="utf-8", newline="") as stream:
            raw_physics = list(csv.DictReader(stream))
    try:
        receipt = json.loads(_inside(root, "metadata/aedt_run_receipt.json").read_text(encoding="utf-8"))
        # Import only when a fixture has been explicitly selected and bound.
        from scgsim.aedt import resolve_epr_result, resolve_results

        if kind == "aedt_saved_epr":
            if receipt.get("status") != "epr_analysis_completed" or "native_batch" in receipt:
                raise AssertionError("private fixture is not a saved-EPR analysis operation")
            result_path = _inside(root, manifest["epr_result_path"])
            epr = resolve_epr_result(result_path).to_payload()
            mode = receipt.get("mode")
            physics = None
        else:
            is_batch = "native_batch" in receipt
            if is_batch != (kind == "aedt_batch_collection"):
                raise AssertionError("private fixture one-shot/batch lineage kind differs")
            # The existing public resolver validates completed status, native
            # batch lineage when present, receipt hashes and requested EPR.
            result = resolve_results(root)
            mode = result.mode
            physics = list(result.physics_results())
            epr_result = result.epr_result()
            epr = None if epr_result is None else epr_result.to_payload()
        if raw_epr is not None:
            _assert_epr_preserved(epr, raw_epr)
        if raw_physics is not None:
            _assert_subset(physics, raw_physics, "raw physics CSV")
        _assert_subset({
            "operation_kind": kind, "receipt": receipt, "mode": mode,
            "physics": physics, "epr": epr,
            "handoff_metadata_sha256": metadata_sha,
        }, selected)
    finally:
        if _inventory(root) != before or _file_identity(private_golden_manifest) != manifest_before:
            raise AssertionError("private fixture files changed during strict returned-data reading")
