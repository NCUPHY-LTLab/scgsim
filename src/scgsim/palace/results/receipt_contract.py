"""Validate completed-run receipts and native config-compatibility observations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import (
    CONFIG,
    INDEX_MAP,
    NATIVE_METADATA,
    RETURNED_RECEIPT,
    RETURNED_RECEIPT_SCHEMA,
    RUNTIME_COMPATIBILITY,
    RUNTIME_COMPATIBILITY_SCHEMA,
    collects_adapted_meshes,
    output_table,
)
from scgsim.palace.results.artifacts import (
    compute_hash_entry,
    compute_input_hash_entry,
    confined_path,
    expect_int,
    expect_list,
    expect_mapping,
    expect_scalar,
    extract_hash_map,
    read_json,
    sha256,
    validate_timestamp,
)
from scgsim.palace.results.models import PalaceReturnedReceipt
from scgsim.palace.results.tables import count_index_entries, output_families
from scgsim.palace.run.returned_receipt import adapted_mesh_output_paths


@dataclass(frozen=True)
class ReceiptValidation:
    """Sealed receipt, input metadata and run location validated as one returned run."""

    receipt_payload: dict[str, Any]
    expected_handoff_id: str
    handoff_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    problem: str
    route: str
    root: Path

    def validate(self) -> None:
        """Verify receipt binding, success, artifacts, compatibility and logs in their original order."""
        if (
            expect_scalar(self.receipt_payload, "schema", str)
            != RETURNED_RECEIPT_SCHEMA
        ):
            raise ValueError("returned run receipt schema is invalid.")
        if expect_int(self.receipt_payload.get("schema_version"), optional=False) != 1:
            raise ValueError("returned run receipt schema_version must be 1.")

        status = expect_scalar(self.receipt_payload, "status", str)
        if status != "completed":
            raise ValueError("returned run receipt status is not completed.")

        if expect_scalar(self.receipt_payload, "route", str) != self.route:
            raise ValueError("returned run receipt route mismatch.")
        if expect_scalar(self.receipt_payload, "problem", str) != self.problem:
            raise ValueError("returned run receipt problem mismatch.")
        if self.receipt_payload.get("route_a_thin_film") != self.handoff_metadata.get(
            "route_a_thin_film"
        ):
            raise ValueError(
                "returned run receipt Route-A thin-film identity mismatch."
            )

        _validate_receipt_binding(
            self.expected_handoff_id,
            self.handoff_metadata,
            self.receipt_payload,
            self.run_metadata,
        )

        _validate_success_and_job(self.receipt_payload)

        _validate_returned_inputs(
            self.handoff_metadata,
            self.receipt_payload,
            self.root,
            self.route,
            self.run_metadata,
        )

        output_map = _validate_returned_outputs(
            self.handoff_metadata, self.problem, self.receipt_payload, self.root
        )

        _validate_completed_runtime(
            self.handoff_metadata, output_map, self.receipt_payload, self.root
        )

        _validate_log_and_solver_identity(self.receipt_payload, self.root)


def build_returned_receipt(payload: dict[str, Any]) -> PalaceReturnedReceipt:
    """Detach validated receipt fields while retaining optional execution-stage observations."""
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
        log=expect_mapping(payload.get("log"), "receipt.log")
        if payload.get("log") is not None
        else None,
        solver_identity=expect_mapping(
            payload.get("solver_identity"), "receipt.solver_identity"
        ),
        execution_stage=expect_scalar(payload, "execution_stage", str, fallback=None),
        solver_invoked=expect_scalar(payload, "solver_invoked", bool, fallback=None),
        preflight_exit_code=expect_int(
            payload.get("preflight_exit_code"), optional=True
        ),
    )


def validate_returned_state_agreement(
    *,
    receipt_payload: dict[str, Any],
    run_metadata: dict[str, Any],
    resource_record: dict[str, Any],
) -> None:
    """Require receipt, run metadata and resource record to report the same completion."""
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
    problem: str, *, index_counts: dict[str, int]
) -> tuple[dict[str, Any], ...]:
    """List native output families required by the written problem and configured selectors."""
    paths = [
        *(output_table(family) for family in output_families(problem, index_counts)),
        NATIVE_METADATA,
    ]
    return tuple({"path": path, "bytes": 0, "sha256": ""} for path in paths)


def validate_palace_identity(payload: Mapping[str, Any]) -> None:
    """Validate solver provenance without introducing version-based eligibility policy."""
    if not payload:
        raise ValueError("palace_identity must be a non-empty mapping.")
    expect_scalar(payload, "config_schema", str)
    probe = payload.get("config_compatibility_probe")
    if probe is None:
        expect_scalar(payload, "runtime", str)
        return
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


def validate_runtime_compatibility(
    payload: Mapping[str, Any], palace_identity: Mapping[str, Any], *, root: Path
) -> None:
    """Bind successful native dry-run observations to the exact sealed config bytes."""
    if payload.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA:
        raise ValueError("Palace runtime compatibility schema is invalid.")
    config = expect_mapping(payload.get("config"), "runtime compatibility config")
    expected_config = {
        "path": CONFIG,
        "bytes": (root / CONFIG).stat().st_size,
        "sha256": sha256(root / CONFIG),
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
    _validate_observed_version(observed)
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


def _validate_log_and_solver_identity(receipt_payload: Any, root: Path) -> None:
    """Verify returned log bytes and solver identity against the native Palace JSON."""
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

    receipt_identity = expect_mapping(
        receipt_payload.get("solver_identity"), "receipt.solver_identity"
    )
    palace_identity = read_json(root / NATIVE_METADATA)
    expected_identity = {
        key: palace_identity[key]
        for key in ("GitTag", "PeakMemoryMegabytes", "PeakNodeMemoryMegabytes")
        if key in palace_identity
    }
    if not expected_identity or receipt_identity != expected_identity:
        raise ValueError(
            f"returned run receipt solver identity does not match {Path(NATIVE_METADATA).name}."
        )


def _validate_completed_runtime(
    handoff_metadata: Any, output_map: Any, receipt_payload: Any, root: Path
) -> None:
    """Require the completed solve to retain its successful config-probe artifact."""
    declared_palace_identity = expect_mapping(
        handoff_metadata.get("palace_identity"), "handoff palace_identity"
    )
    if declared_palace_identity.get("config_compatibility_probe") is not None:
        if receipt_payload.get("execution_stage") != "solver":
            raise ValueError("completed Palace receipt execution_stage must be solver.")
        if receipt_payload.get("solver_invoked") is not True:
            raise ValueError("completed Palace receipt must record solver invocation.")
        if expect_int(receipt_payload.get("preflight_exit_code")) != 0:
            raise ValueError(
                "completed Palace receipt config compatibility preflight must pass."
            )
        if RUNTIME_COMPATIBILITY not in output_map:
            raise ValueError(
                "returned run receipt misses the Palace runtime compatibility artifact."
            )
        runtime_compatibility = expect_mapping(
            receipt_payload.get("runtime_compatibility"),
            "receipt.runtime_compatibility",
        )
        runtime_artifact = read_json(confined_path(root, RUNTIME_COMPATIBILITY))
        if runtime_compatibility != runtime_artifact:
            raise ValueError(
                "returned run receipt Palace runtime compatibility does not match "
                "its artifact."
            )
        validate_runtime_compatibility(
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
                "Palace compatibility probe executable does not match the handoff "
                "command."
            )
        resources = expect_mapping(
            handoff_metadata.get("requested_resources"), "handoff requested_resources"
        )
        if runtime_command.get("command_style") != resources.get(
            "command_style", "binary"
        ):
            raise ValueError(
                "Palace compatibility probe command style does not match the handoff."
            )


def _validate_returned_outputs(
    handoff_metadata: Any, problem: str, receipt_payload: Any, root: Path
) -> Any:
    """Verify required solver artifacts and every returned output size and hash."""
    output_files = expect_list(receipt_payload.get("output_files"), "output_files")
    output_map = extract_hash_map(output_files)
    index_map = read_json(root / INDEX_MAP)
    required_outputs = _required_output_records(
        problem,
        index_counts=count_index_entries(index_map),
    )
    config = read_json(root / CONFIG)
    if collects_adapted_meshes(handoff_metadata.get("palace_identity")):
        required_outputs += tuple(
            {"path": relative} for relative in adapted_mesh_output_paths(root, config)
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


def _validate_returned_inputs(
    handoff_metadata: Any,
    receipt_payload: Any,
    root: Path,
    route: str,
    run_metadata: Any,
) -> None:
    """Bind receipt input hashes and route to the original run and handoff metadata."""
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


def _validate_success_and_job(receipt_payload: Any) -> None:
    """Require successful pipeline exit codes and a nonempty recorded job identity."""
    exit_code = expect_int(receipt_payload.get("exit_code"))
    if exit_code != 0:
        raise ValueError(
            "returned run receipt exit_code must be zero for completed runs."
        )
    if expect_int(receipt_payload.get("solver_exit_code")) != 0:
        raise ValueError("returned run receipt solver_exit_code must be zero.")
    if expect_int(receipt_payload.get("tee_exit_code")) != 0:
        raise ValueError("returned run receipt tee_exit_code must be zero.")

    validate_timestamp(
        expect_scalar(receipt_payload, "timestamp_utc", str), "timestamp_utc"
    )
    job_identity = expect_mapping(
        receipt_payload.get("job_identity"), "returned receipt job_identity"
    )
    if expect_scalar(job_identity, "job_id", str, fallback="") == "":
        raise ValueError("returned run receipt job_id must be a non-empty string.")
    if expect_scalar(job_identity, "job_name", str, fallback="") == "":
        raise ValueError("returned run receipt job_name must be a non-empty string.")


def _validate_receipt_binding(
    expected_handoff_id: str,
    handoff_metadata: Any,
    receipt_payload: Any,
    run_metadata: Any,
) -> None:
    """Require the returned receipt to identify the expected sealed handoff."""
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
    if handoff_metadata.get("returned_receipt_path") != RETURNED_RECEIPT:
        raise ValueError("handoff metadata returned_receipt_path is incorrect.")
    if run_metadata.get("returned_receipt_path") != RETURNED_RECEIPT:
        raise ValueError("run metadata returned_receipt_path is incorrect.")

    handoff_id = expect_scalar(receipt_payload, "handoff_id", str)
    if handoff_id != expected_handoff_id:
        raise ValueError(
            "returned run receipt handoff_id does not match expected_handoff_id."
        )
    if expect_scalar(receipt_payload, "identity_verified", bool) is not True:
        raise ValueError("returned run receipt identity is not verified.")


def _validate_observed_version(observed: Any) -> None:
    """Validate optional native version observations without making them an execution Gate."""
    native_git_tag = observed.get("native_git_tag")
    if native_git_tag is not None and (
        not isinstance(native_git_tag, str) or not native_git_tag
    ):
        raise ValueError("Palace observed native GitTag must be a non-empty string.")
    native_schema_version = observed.get("native_schema_version")
    if native_schema_version is not None and (
        not isinstance(native_schema_version, str) or not native_schema_version
    ):
        raise ValueError(
            "Palace observed native schema version must be non-empty text or null"
        )
    version_probe_exit_code = observed.get("version_probe_exit_code")
    if version_probe_exit_code is not None and (
        isinstance(version_probe_exit_code, bool)
        or not isinstance(version_probe_exit_code, int)
    ):
        raise ValueError("Palace version probe exit code must be an integer or null.")
