"""Check the returned run receipt that the generated run script writes.

The receipt is the run script's own account of what happened: exit codes, the job it
ran under, the hashes of its inputs and outputs, and the Palace log. This module
checks it against the metadata and against the files on disk, so a receipt copied
from another run, or one that claims files that are not there, is refused.

Every run preflights the config with Palace's ``--dry-run`` before it solves
(``config_compatibility_probe``), so every receipt carries the execution stage, the
solver invocation and a runtime-compatibility artifact
(docs/specs/palace-electrostatic-sgb.qmd, "native --dry-run").
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._files import sha256_file
from ._artifact_files import (
    HashEntry,
    compute_hash_entry,
    compute_input_hash_entry,
    confined_path,
    extract_hash_map,
    read_json,
)
from ._expect import (
    expect_int,
    expect_list,
    expect_mapping,
    expect_scalar,
)
from ._index_map import IndexCounts, count_index_entries, surface_mask_entries
from ._result_tables import output_families
from ._result_types import PalaceReturnedReceipt
from ._run_documents import (
    HandoffMetadata,
    ResourceRecord,
    ReturnedReceiptPayload,
    RunMetadata,
)

_RETURNED_RECEIPT_SCHEMA = "palace-returned-run-receipt.v1"

RETURNED_RECEIPT_FILE = "metadata/palace_returned_run_receipt.json"

_RUNTIME_COMPATIBILITY_PATH = "metadata/palace_runtime_compatibility.json"

_RUNTIME_COMPATIBILITY_SCHEMA = "palace-runtime-compatibility.v1"

# ------------------------------------------------------- the receipt checks


def validate_receipt_payload(  # noqa: PLR0913 - keyword-only; each argument is a distinct run document
    *,
    receipt_payload: ReturnedReceiptPayload,
    expected_handoff_id: str,
    handoff_metadata: HandoffMetadata,
    run_metadata: RunMetadata,
    problem: str,
    route: str,
    root: Path,
) -> None:
    """Fail unless the receipt proves a completed run of exactly this handoff."""

    _check_receipt_claims(receipt_payload, handoff_metadata, problem, route)
    _check_handoff_id_chain(
        receipt_payload, expected_handoff_id, handoff_metadata, run_metadata
    )
    _check_receipt_exit_codes(receipt_payload)
    _check_receipt_job(receipt_payload)
    _check_receipt_inputs(receipt_payload, handoff_metadata, run_metadata, route, root)
    output_map = _check_receipt_outputs(receipt_payload, problem, root)
    _check_runtime_compatibility_probe(
        receipt_payload, handoff_metadata, output_map, root
    )
    _check_receipt_log(receipt_payload, root)
    _check_receipt_solver_identity(receipt_payload, root)


def _check_receipt_claims(
    receipt_payload: ReturnedReceiptPayload,
    handoff_metadata: HandoffMetadata,
    problem: str,
    route: str,
) -> None:
    """The receipt says: completed, this route and problem, this thin-film identity."""
    if expect_scalar(receipt_payload, "schema", str) != _RETURNED_RECEIPT_SCHEMA:
        raise ValueError("returned run receipt schema is invalid.")
    if expect_int(receipt_payload.get("schema_version"), optional=False) != 1:
        raise ValueError("returned run receipt schema_version must be 1.")

    status = expect_scalar(receipt_payload, "status", str)
    if status != "completed":
        raise ValueError("returned run receipt status is not completed.")

    if expect_scalar(receipt_payload, "route", str) != route:
        raise ValueError("returned run receipt route mismatch.")
    if expect_scalar(receipt_payload, "problem", str) != problem:
        raise ValueError("returned run receipt problem mismatch.")
    if receipt_payload.get("route_a_thin_film") != handoff_metadata.get(
        "route_a_thin_film"
    ):
        raise ValueError("returned run receipt Route-A thin-film identity mismatch.")


def _check_handoff_id_chain(
    receipt_payload: ReturnedReceiptPayload,
    expected_handoff_id: str,
    handoff_metadata: HandoffMetadata,
    run_metadata: RunMetadata,
) -> None:
    """Run metadata, handoff metadata and receipt must all carry the expected ID."""
    if (
        expect_scalar(run_metadata, "handoff_id", str, fallback="")
        != expected_handoff_id
    ):
        raise ValueError("run metadata handoff_id does not match expected_handoff_id.")
    if (
        expect_scalar(handoff_metadata, "handoff_id", str, fallback="")
        != expected_handoff_id
    ):
        raise ValueError(
            "handoff metadata handoff_id does not match expected_handoff_id."
        )
    if handoff_metadata.get("returned_receipt_path") != RETURNED_RECEIPT_FILE:
        raise ValueError("handoff metadata returned_receipt_path is incorrect.")
    if run_metadata.get("returned_receipt_path") != RETURNED_RECEIPT_FILE:
        raise ValueError("run metadata returned_receipt_path is incorrect.")

    handoff_id = expect_scalar(receipt_payload, "handoff_id", str)
    if handoff_id != expected_handoff_id:
        raise ValueError(
            "returned run receipt handoff_id does not match expected_handoff_id."
        )
    if expect_scalar(receipt_payload, "identity_verified", bool) is not True:
        raise ValueError("returned run receipt identity is not verified.")


def _check_receipt_exit_codes(receipt_payload: ReturnedReceiptPayload) -> None:
    """A completed run exited zero in Palace, in the solver and in output capture."""
    exit_code = expect_int(receipt_payload.get("exit_code"))
    if exit_code != 0:
        raise ValueError(
            "returned run receipt exit_code must be zero for completed runs."
        )
    if expect_int(receipt_payload.get("solver_exit_code")) != 0:
        raise ValueError("returned run receipt solver_exit_code must be zero.")
    if expect_int(receipt_payload.get("tee_exit_code")) != 0:
        raise ValueError("returned run receipt tee_exit_code must be zero.")


def _check_receipt_job(receipt_payload: ReturnedReceiptPayload) -> None:
    """The receipt has a UTC timestamp and a named job."""
    _validate_timestamp(
        expect_scalar(receipt_payload, "timestamp_utc", str), "timestamp_utc"
    )
    job_identity = expect_mapping(
        receipt_payload.get("job_identity"), "returned receipt job_identity"
    )
    if expect_scalar(job_identity, "job_id", str, fallback="") == "":
        raise ValueError("returned run receipt job_id must be a non-empty string.")
    if expect_scalar(job_identity, "job_name", str, fallback="") == "":
        raise ValueError("returned run receipt job_name must be a non-empty string.")


def _check_receipt_inputs(
    receipt_payload: ReturnedReceiptPayload,
    handoff_metadata: HandoffMetadata,
    run_metadata: RunMetadata,
    route: str,
    root: Path,
) -> None:
    """The inputs the run read hash to what the handoff sealed."""
    expected_input_map = extract_hash_map(
        expect_list(receipt_payload.get("input_hashes"), "input_hashes")
    )
    configured_input_map = extract_hash_map(
        expect_list(handoff_metadata.get("hashes"), "metadata hashes")
    )
    if expected_input_map != configured_input_map:
        raise ValueError(
            "returned run receipt input hashes do not match metadata hashes."
        )
    for name, expected in expected_input_map.items():
        observed = compute_input_hash_entry(root, name)
        if expected != observed:
            raise ValueError(f"returned run receipt input hash mismatch for {name}.")

    if expect_scalar(handoff_metadata, "route", str) != route:
        raise ValueError("handoff metadata route mismatch.")
    if expect_scalar(run_metadata, "route", str) != route:
        raise ValueError("run metadata route mismatch.")


def _check_receipt_outputs(
    receipt_payload: ReturnedReceiptPayload, problem: str, root: Path
) -> dict[str, HashEntry]:
    """Every required output is listed, present and hashes as listed. Returns the map."""
    output_files = expect_list(receipt_payload.get("output_files"), "output_files")
    output_map = extract_hash_map(output_files)
    index_map = read_json(root / "metadata" / "palace_index_map.json")
    required_outputs = _required_output_records(
        problem,
        index_counts=count_index_entries(index_map),
        masks_configured=bool(surface_mask_entries(index_map)),
    )
    for expected in required_outputs:
        rel = expected["path"]
        if rel not in output_map:
            raise ValueError(
                f"returned run receipt misses required output entry: {rel}"
            )

    for entry in output_files:
        data = expect_mapping(entry, "returned receipt output entry")
        if expect_scalar(data, "present", bool) is not True:
            raise ValueError("returned run receipt lists a missing required output.")

    for rel, expected in output_map.items():
        observed = compute_hash_entry(confined_path(root, rel))
        if observed[0] != expected[0] or observed[1] != expected[1]:
            raise ValueError(f"returned run receipt output hash mismatch for {rel}")
    return output_map


def _check_runtime_compatibility_probe(
    receipt_payload: ReturnedReceiptPayload,
    handoff_metadata: HandoffMetadata,
    output_map: dict[str, HashEntry],
    root: Path,
) -> None:
    """The run must prove Palace accepted the sealed config before it solved.

    Every handoff declares the ``native_dry_run`` probe; ``_validate_palace_identity``
    has already refused any other identity.
    """
    declared_palace_identity = expect_mapping(
        handoff_metadata.get("palace_identity"), "handoff palace_identity"
    )
    if receipt_payload.get("execution_stage") != "solver":
        raise ValueError("completed Palace receipt execution_stage must be solver.")
    if receipt_payload.get("solver_invoked") is not True:
        raise ValueError("completed Palace receipt must record solver invocation.")
    if expect_int(receipt_payload.get("preflight_exit_code")) != 0:
        raise ValueError(
            "completed Palace receipt config compatibility preflight must pass."
        )
    if _RUNTIME_COMPATIBILITY_PATH not in output_map:
        raise ValueError(
            "returned run receipt misses the Palace runtime compatibility artifact."
        )
    runtime_compatibility = expect_mapping(
        receipt_payload.get("runtime_compatibility"),
        "receipt.runtime_compatibility",
    )
    runtime_artifact = read_json(confined_path(root, _RUNTIME_COMPATIBILITY_PATH))
    if runtime_compatibility != runtime_artifact:
        raise ValueError(
            "returned run receipt Palace runtime compatibility does not match "
            "its artifact."
        )
    _validate_runtime_compatibility(
        runtime_compatibility, declared_palace_identity, root=root
    )
    runtime_command = expect_mapping(
        runtime_compatibility.get("command"),
        "receipt.runtime_compatibility.command",
    )
    execution_identity = expect_mapping(
        handoff_metadata.get("execution_identity"), "handoff execution_identity"
    )
    expected_command = expect_mapping(
        execution_identity.get("command"), "handoff execution command"
    )
    if runtime_command.get("executable_identity") != expected_command.get(
        "executable_identity"
    ):
        raise ValueError(
            "Palace compatibility probe executable does not match the handoff command."
        )
    resources = expect_mapping(
        handoff_metadata.get("requested_resources"), "handoff requested_resources"
    )
    # The caller may leave command_style out of the resources: the handoff then
    # writes a binary command (execution/handoff.py ``_validate_resources``).
    if runtime_command.get("command_style") != resources.get("command_style", "binary"):
        raise ValueError(
            "Palace compatibility probe command style does not match the handoff."
        )


def _check_receipt_log(receipt_payload: ReturnedReceiptPayload, root: Path) -> None:
    """The Palace log named in the receipt exists and hashes as recorded."""
    log = expect_mapping(receipt_payload.get("log"), "return_receipt.log")
    if not log:
        raise ValueError("returned run receipt log entry is missing.")
    log_path = expect_scalar(log, "path", str)
    log_expected = expect_scalar(log, "bytes", int), expect_scalar(log, "sha256", str)
    log_observed = compute_hash_entry(confined_path(root, log_path))
    if log_expected != log_observed:
        raise ValueError(f"returned run receipt log hash mismatch for {log_path}.")
    if log_path == "":
        raise ValueError("returned run receipt log path must be non-empty.")
    if expect_scalar(log, "present", bool) is not True:
        raise ValueError("returned run receipt log is not present.")


# palace.json is written by Palace itself, so it is the reference for the solver
# identity the run script copied into the receipt.
def _check_receipt_solver_identity(
    receipt_payload: ReturnedReceiptPayload, root: Path
) -> None:
    """The solver identity in the receipt is the one Palace wrote in palace.json."""
    receipt_identity = expect_mapping(
        receipt_payload.get("solver_identity"), "receipt.solver_identity"
    )
    palace_identity = read_json(root / "results/palace/palace.json")
    expected_identity = {
        key: palace_identity[key]
        for key in ("GitTag", "PeakMemoryMegabytes", "PeakNodeMemoryMegabytes")
        if key in palace_identity
    }
    if not expected_identity or receipt_identity != expected_identity:
        raise ValueError(
            "returned run receipt solver identity does not match palace.json."
        )


# ------------------------------------- receipt versus the other records


# The run metadata and resource record are written separately from the receipt,
# so they must repeat its status, exit code and job identity exactly.
def validate_returned_state_agreement(
    *,
    receipt_payload: ReturnedReceiptPayload,
    run_metadata: RunMetadata,
    resource_record: ResourceRecord,
) -> None:
    receipt_status = expect_scalar(receipt_payload, "status", str)
    receipt_exit = expect_int(receipt_payload.get("exit_code"))
    receipt_job = expect_mapping(receipt_payload.get("job_identity"), "receipt job")
    for label, payload in (
        ("run metadata", run_metadata),
        ("resource record", resource_record),
    ):
        if expect_scalar(payload, "status", str) != receipt_status:
            raise ValueError(f"{label} status does not match returned receipt.")
        if expect_int(payload.get("exit_code")) != receipt_exit:
            raise ValueError(f"{label} exit_code does not match returned receipt.")
        if (
            expect_mapping(payload.get("job_identity"), f"{label} job identity")
            != receipt_job
        ):
            raise ValueError(f"{label} job identity does not match returned receipt.")


def _required_output_records(
    problem: str, *, index_counts: IndexCounts, masks_configured: bool = False
) -> tuple[dict[str, Any], ...]:
    families = output_families(problem, index_counts)
    records: list[dict[str, Any]] = []
    for family in families:
        rel = f"results/palace/{family}.csv"
        records.append({"path": rel, "bytes": 0, "sha256": ""})
    if masks_configured:
        for name in ("surface-mask-Q", "surface-mask-energy"):
            records.append(
                {"path": f"results/palace/{name}.csv", "bytes": 0, "sha256": ""}
            )
    records.append({"path": "results/palace/palace.json", "bytes": 0, "sha256": ""})
    return tuple(records)


def _validate_runtime_compatibility(
    payload: Mapping[str, Any], palace_identity: Mapping[str, Any], *, root: Path
) -> None:
    if payload.get("schema") != _RUNTIME_COMPATIBILITY_SCHEMA:
        raise ValueError("Palace runtime compatibility schema is invalid.")
    config = expect_mapping(payload.get("config"), "runtime compatibility config")
    expected_config = {
        "path": "config.json",
        "bytes": (root / "config.json").stat().st_size,
        "sha256": sha256_file(root / "config.json"),
        "declared_schema": expect_scalar(palace_identity, "config_schema", str),
    }
    if config != expected_config:
        raise ValueError(
            "Palace runtime compatibility probe does not bind the sealed config."
        )
    compatibility = expect_mapping(
        payload.get("compatibility"), "runtime compatibility result"
    )
    if compatibility != {
        "method": "native_dry_run",
        "probe_exit_code": 0,
        "verified": True,
    }:
        raise ValueError("Palace did not accept the sealed config during dry-run.")
    observed = expect_mapping(
        payload.get("observed_identity"), "runtime observed identity"
    )
    native_git_tag = observed.get("native_git_tag")
    if native_git_tag is not None and (
        not isinstance(native_git_tag, str) or not native_git_tag
    ):
        raise ValueError("Palace observed native GitTag must be a non-empty string.")
    version_probe_exit_code = observed.get("version_probe_exit_code")
    if version_probe_exit_code is not None and (
        isinstance(version_probe_exit_code, bool)
        or not isinstance(version_probe_exit_code, int)
    ):
        raise ValueError("Palace version probe exit code must be an integer or null.")
    runtime_binary_identity = observed.get("runtime_binary_identity")
    runtime_binary_sha256 = observed.get("runtime_binary_sha256")
    if (runtime_binary_identity is None) != (runtime_binary_sha256 is None):
        raise ValueError("Palace runtime binary provenance must be complete or absent.")
    if runtime_binary_identity is not None:
        if (
            not isinstance(runtime_binary_identity, str)
            or not runtime_binary_identity
            or Path(runtime_binary_identity).name != runtime_binary_identity
        ):
            raise ValueError("Palace runtime binary identity must be a basename.")
        if (
            not isinstance(runtime_binary_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", runtime_binary_sha256) is None
        ):
            raise ValueError("Palace runtime binary SHA-256 is invalid.")
    command = expect_mapping(payload.get("command"), "runtime compatibility command")
    if command.get("command_style") not in {"binary", "wrapper"}:
        raise ValueError("Palace runtime compatibility command style is invalid.")
    executable_identity = expect_scalar(command, "executable_identity", str)
    if Path(executable_identity).name != executable_identity:
        raise ValueError("Palace runtime executable must be a basename.")
    if runtime_binary_identity is not None and command.get("command_style") != "binary":
        raise ValueError(
            "Palace runtime binary SHA may describe only the configured binary."
        )


def _validate_timestamp(value: str, label: str) -> None:
    try:
        parsed = value
        if value.endswith("Z"):
            parsed = value.replace("Z", "+00:00")
        __import__("datetime").datetime.fromisoformat(parsed)
    except Exception as exc:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp.") from exc


# ------------------------------------------------- receipt as a value object


def build_returned_receipt(payload: ReturnedReceiptPayload) -> PalaceReturnedReceipt:
    return PalaceReturnedReceipt(
        schema=expect_scalar(payload, "schema", str),
        schema_version=expect_int(payload.get("schema_version")),
        handoff_id=expect_scalar(payload, "handoff_id", str),
        status=expect_scalar(payload, "status", str),
        exit_code=expect_int(payload.get("exit_code")),
        solver_exit_code=expect_int(payload.get("solver_exit_code")),
        tee_exit_code=expect_int(payload.get("tee_exit_code")),
        route=expect_scalar(payload, "route", str),
        problem=expect_scalar(payload, "problem", str),
        identity_verified=expect_scalar(payload, "identity_verified", bool),
        timestamp_utc=expect_scalar(payload, "timestamp_utc", str),
        job_identity=expect_mapping(
            payload.get("job_identity"), "receipt.job_identity"
        ),
        input_hashes=tuple(
            expect_mapping(entry, "returned_receipt.input_hash")
            for entry in expect_list(
                payload.get("input_hashes"), "returned receipt input_hashes"
            )
        ),
        output_files=tuple(
            expect_mapping(entry, "returned_receipt.output_file")
            for entry in expect_list(
                payload.get("output_files"), "returned receipt output_files"
            )
        ),
        log=expect_mapping(payload.get("log"), "receipt.log"),
        solver_identity=expect_mapping(
            payload.get("solver_identity"), "receipt.solver_identity"
        ),
        execution_stage=expect_scalar(payload, "execution_stage", str),
        solver_invoked=expect_scalar(payload, "solver_invoked", bool),
        preflight_exit_code=expect_int(payload.get("preflight_exit_code")),
    )
