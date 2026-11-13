"""Validate completed-run receipts and native config-compatibility observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import (
    CONFIG,
    HANDOFF_METADATA,
    INDEX_MAP,
    NATIVE_METADATA,
    RETURNED_RECEIPT,
    RETURNED_RECEIPT_SCHEMA,
    RUNTIME_COMPATIBILITY,
    collects_adapted_meshes,
    confined_path,
    output_table,
    read_json,
)
from scgsim.geometry.models.refusals import Declared
from scgsim.palace.results.artifacts import (
    RETURNED_RUN_FOLDER,
    FileHash,
    compute_hash_entry,
    compute_input_hash_entry,
    expect_int,
    expect_list,
    expect_mapping,
    expect_scalar,
    extract_hash_map,
    validate_timestamp,
)
from scgsim.palace.results.resolved_records import PalaceReturnedReceipt
from scgsim.palace.results.runtime_contract import (
    validate_runtime_compatibility,
)
from scgsim.palace.results.tables import count_index_entries, output_families
from scgsim.palace.run.returned_receipt import adapted_mesh_output_paths
from scgsim.refusal import found_text


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
        self._validate_header()
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

    def _validate_header(self) -> None:
        """The receipt has the current schema, says completed, and names the route and problem of the run."""
        schema = expect_scalar(self.receipt_payload, "schema", str)
        if schema != RETURNED_RECEIPT_SCHEMA:
            raise ValueError(
                Declared(
                    "returned run receipt schema is invalid", RETURNED_RUN_FOLDER
                ).refusal(found_text(schema, RETURNED_RECEIPT_SCHEMA))
            )
        schema_version = expect_int(
            self.receipt_payload.get("schema_version"),
            "returned receipt schema_version",
            optional=False,
        )
        if schema_version != 1:
            raise ValueError(
                Declared(
                    "returned run receipt schema_version must be 1", RETURNED_RUN_FOLDER
                ).refusal(found_text(schema_version))
            )

        status = expect_scalar(self.receipt_payload, "status", str)
        if status != "completed":
            raise ValueError(
                Declared(
                    "returned run receipt status is not completed", RETURNED_RUN_FOLDER
                ).refusal(found_text(status, "completed"))
            )

        route = expect_scalar(self.receipt_payload, "route", str)
        if route != self.route:
            raise ValueError(
                Declared(
                    "returned run receipt route mismatch", RETURNED_RUN_FOLDER
                ).refusal(found_text(route, self.route))
            )
        problem = expect_scalar(self.receipt_payload, "problem", str)
        if problem != self.problem:
            raise ValueError(
                Declared(
                    "returned run receipt problem mismatch", RETURNED_RUN_FOLDER
                ).refusal(found_text(problem, self.problem))
            )
        thin_film = self.receipt_payload.get("route_a_thin_film")
        if thin_film != self.handoff_metadata.get("route_a_thin_film"):
            raise ValueError(
                Declared(
                    "returned run receipt Route-A thin-film identity mismatch",
                    RETURNED_RUN_FOLDER,
                ).refusal(
                    found_text(
                        thin_film, self.handoff_metadata.get("route_a_thin_film")
                    )
                )
            )


def build_returned_receipt(payload: dict[str, Any]) -> PalaceReturnedReceipt:
    """Detach validated receipt fields while retaining optional execution-stage observations."""
    return PalaceReturnedReceipt(
        schema=expect_scalar(payload, "schema", str),
        schema_version=expect_int(
            payload.get("schema_version"), "returned receipt schema_version"
        ),
        handoff_id=expect_scalar(payload, "handoff_id", str),
        status=expect_scalar(payload, "status", str),
        exit_code=expect_int(payload.get("exit_code"), "returned receipt exit_code"),
        solver_exit_code=expect_int(
            payload.get("solver_exit_code"), "returned receipt solver_exit_code"
        ),
        tee_exit_code=expect_int(
            payload.get("tee_exit_code"), "returned receipt tee_exit_code"
        ),
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
            payload.get("preflight_exit_code"),
            "returned receipt preflight_exit_code",
            optional=True,
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
    receipt_exit = expect_int(
        receipt_payload.get("exit_code"), "returned receipt exit_code"
    )
    receipt_job = expect_mapping(receipt_payload.get("job_identity"), "receipt job")
    for label, payload in (
        ("run metadata", run_metadata),
        ("resource record", resource_record),
    ):
        status = expect_scalar(payload, "status", str)
        if status != receipt_status:
            raise ValueError(
                Declared(
                    f"{label} status does not match returned receipt",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(status, receipt_status))
            )
        exit_code = expect_int(payload.get("exit_code"), f"{label} exit_code")
        if exit_code != receipt_exit:
            raise ValueError(
                Declared(
                    f"{label} exit_code does not match returned receipt",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(exit_code, receipt_exit))
            )
        job = expect_mapping(payload.get("job_identity"), f"{label} job identity")
        if job != receipt_job:
            raise ValueError(
                Declared(
                    f"{label} job identity does not match returned receipt",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(job, receipt_job))
            )


def _required_output_records(
    problem: str, *, index_counts: dict[str, int]
) -> tuple[dict[str, Any], ...]:
    """List native output families required by the written problem and configured selectors."""
    paths = [
        *(output_table(family) for family in output_families(problem, index_counts)),
        NATIVE_METADATA,
    ]
    return tuple({"path": path, "bytes": 0, "sha256": ""} for path in paths)


def _validate_log_and_solver_identity(receipt_payload: Any, root: Path) -> None:
    """Verify returned log bytes and solver identity against the native Palace JSON."""
    log = expect_mapping(receipt_payload.get("log"), "return_receipt.log")
    if not log:
        raise ValueError(
            Declared(
                f"returned run receipt log entry in {RETURNED_RECEIPT} is missing",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(log))
        )
    log_path = expect_scalar(log, "path", str)
    log_expected = FileHash(
        expect_scalar(log, "bytes", int), expect_scalar(log, "sha256", str)
    )
    log_observed = compute_hash_entry(confined_path(root, log_path))
    if log_expected != log_observed:
        raise ValueError(
            Declared(
                f"returned run receipt log hash mismatch for {log_path}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(log_observed, log_expected))
        )
    if log_path == "":
        raise ValueError(
            Declared(
                "returned run receipt log path must be non-empty", RETURNED_RUN_FOLDER
            ).refusal(found_text(log_path))
        )
    if expect_scalar(log, "present", bool) is not True:
        raise ValueError(
            Declared(
                f"returned run receipt log {log_path} is not present",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(log.get("present"), True))
        )

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
            Declared(
                f"returned run receipt solver identity does not match {Path(NATIVE_METADATA).name}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(receipt_identity, expected_identity))
        )


def _validate_completed_runtime(
    handoff_metadata: Any, output_map: Any, receipt_payload: Any, root: Path
) -> None:
    """Require the completed solve to retain its successful config-probe artifact."""
    declared_palace_identity = expect_mapping(
        handoff_metadata.get("palace_identity"), "handoff palace_identity"
    )
    if declared_palace_identity.get("config_compatibility_probe") is None:
        return
    _validate_solver_stage(receipt_payload, output_map)
    runtime_compatibility = _recorded_runtime_compatibility(receipt_payload, root)
    validate_runtime_compatibility(
        runtime_compatibility, declared_palace_identity, root=root
    )
    _validate_probe_command(runtime_compatibility, handoff_metadata)


def _validate_solver_stage(receipt_payload: Any, output_map: Any) -> None:
    """A completed probed run reached the solver after a passing preflight, and kept the probe's record."""
    if receipt_payload.get("execution_stage") != "solver":
        raise ValueError(
            Declared(
                "completed Palace receipt execution_stage must be solver",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(receipt_payload.get("execution_stage"), "solver"))
        )
    if receipt_payload.get("solver_invoked") is not True:
        raise ValueError(
            Declared(
                "completed Palace receipt must record solver invocation",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(receipt_payload.get("solver_invoked"), True))
        )
    if (
        expect_int(
            receipt_payload.get("preflight_exit_code"),
            "returned receipt preflight_exit_code",
        )
        != 0
    ):
        raise ValueError(
            Declared(
                "completed Palace receipt config compatibility preflight must pass",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(receipt_payload.get("preflight_exit_code"), 0))
        )
    if RUNTIME_COMPATIBILITY not in output_map:
        raise ValueError(
            Declared(
                "returned run receipt misses the Palace runtime compatibility artifact",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(sorted(output_map), f"an entry for {RUNTIME_COMPATIBILITY}")
            )
        )


def _recorded_runtime_compatibility(receipt_payload: Any, root: Path) -> dict[str, Any]:
    """The receipt's runtime-compatibility record, which must equal the artifact in the run folder."""
    runtime_compatibility = expect_mapping(
        receipt_payload.get("runtime_compatibility"),
        "receipt.runtime_compatibility",
    )
    runtime_artifact = read_json(confined_path(root, RUNTIME_COMPATIBILITY))
    if runtime_compatibility != runtime_artifact:
        raise ValueError(
            Declared(
                "returned run receipt Palace runtime compatibility does not match "
                f"its artifact {RUNTIME_COMPATIBILITY}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(runtime_compatibility, runtime_artifact))
        )
    return runtime_compatibility


def _validate_probe_command(
    runtime_compatibility: dict[str, Any], handoff_metadata: Any
) -> None:
    """The dry-run probed the executable and command style the handoff runs."""
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
            Declared(
                "Palace compatibility probe executable does not match the handoff command",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    runtime_command.get("executable_identity"),
                    expected_command.get("executable_identity"),
                )
            )
        )
    resources = expect_mapping(
        handoff_metadata.get("requested_resources"), "handoff requested_resources"
    )
    if runtime_command.get("command_style") != resources.get("command_style", "binary"):
        raise ValueError(
            Declared(
                "Palace compatibility probe command style does not match the handoff",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    runtime_command.get("command_style"),
                    resources.get("command_style", "binary"),
                )
            )
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
            raise ValueError(
                Declared(
                    "returned run receipt lists a missing required output",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(data.get("path")))
            )

    for rel, expected in output_map.items():
        observed = compute_hash_entry(confined_path(root, rel))
        if observed != expected:
            raise ValueError(
                Declared(
                    f"returned run receipt output hash mismatch for {rel}",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(observed, expected))
            )
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
            Declared(
                f"returned run receipt input hashes do not match the hashes of {HANDOFF_METADATA}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(expected_input_map, configured_input_map))
        )
    for name, expected in expected_input_map.items():
        observed = compute_input_hash_entry(root, name)
        if expected != observed:
            raise ValueError(
                Declared(
                    f"returned run receipt input hash mismatch for {name}",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(observed, expected))
            )

    handoff_route = expect_scalar(handoff_metadata, "route", str)
    if handoff_route != route:
        raise ValueError(
            Declared("handoff metadata route mismatch", RETURNED_RUN_FOLDER).refusal(
                found_text(handoff_route, route)
            )
        )
    run_route = expect_scalar(run_metadata, "route", str)
    if run_route != route:
        raise ValueError(
            Declared("run metadata route mismatch", RETURNED_RUN_FOLDER).refusal(
                found_text(run_route, route)
            )
        )


def _validate_success_and_job(receipt_payload: Any) -> None:
    """Require successful pipeline exit codes and a nonempty recorded job identity."""
    exit_code = expect_int(
        receipt_payload.get("exit_code"), "returned receipt exit_code"
    )
    if exit_code != 0:
        raise ValueError(
            Declared(
                "returned run receipt exit_code must be zero for completed runs",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(exit_code, 0))
        )
    solver_exit_code = expect_int(
        receipt_payload.get("solver_exit_code"), "returned receipt solver_exit_code"
    )
    if solver_exit_code != 0:
        raise ValueError(
            Declared(
                "returned run receipt solver_exit_code must be zero",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(solver_exit_code, 0))
        )
    tee_exit_code = expect_int(
        receipt_payload.get("tee_exit_code"), "returned receipt tee_exit_code"
    )
    if tee_exit_code != 0:
        raise ValueError(
            Declared(
                "returned run receipt tee_exit_code must be zero", RETURNED_RUN_FOLDER
            ).refusal(found_text(tee_exit_code, 0))
        )

    validate_timestamp(
        expect_scalar(receipt_payload, "timestamp_utc", str), "timestamp_utc"
    )
    job_identity = expect_mapping(
        receipt_payload.get("job_identity"), "returned receipt job_identity"
    )
    if expect_scalar(job_identity, "job_id", str, fallback="") == "":
        raise ValueError(
            Declared(
                "returned run receipt job_id must be a non-empty string",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(job_identity.get("job_id")))
        )
    if expect_scalar(job_identity, "job_name", str, fallback="") == "":
        raise ValueError(
            Declared(
                "returned run receipt job_name must be a non-empty string",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(job_identity.get("job_name")))
        )


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
        raise ValueError(
            Declared(
                "run metadata handoff_id does not match expected_handoff_id",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(run_metadata.get("handoff_id"), expected_handoff_id))
        )
    if (
        expect_scalar(handoff_metadata, "handoff_id", str, fallback="")
        != expected_handoff_id
    ):
        raise ValueError(
            Declared(
                "handoff metadata handoff_id does not match expected_handoff_id",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(handoff_metadata.get("handoff_id"), expected_handoff_id)
            )
        )
    if handoff_metadata.get("returned_receipt_path") != RETURNED_RECEIPT:
        raise ValueError(
            Declared(
                "handoff metadata returned_receipt_path is incorrect",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    handoff_metadata.get("returned_receipt_path"), RETURNED_RECEIPT
                )
            )
        )
    if run_metadata.get("returned_receipt_path") != RETURNED_RECEIPT:
        raise ValueError(
            Declared(
                "run metadata returned_receipt_path is incorrect", RETURNED_RUN_FOLDER
            ).refusal(
                found_text(run_metadata.get("returned_receipt_path"), RETURNED_RECEIPT)
            )
        )

    handoff_id = expect_scalar(receipt_payload, "handoff_id", str)
    if handoff_id != expected_handoff_id:
        raise ValueError(
            Declared(
                "returned run receipt handoff_id does not match expected_handoff_id",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(handoff_id, expected_handoff_id))
        )
    if expect_scalar(receipt_payload, "identity_verified", bool) is not True:
        raise ValueError(
            Declared(
                "returned run receipt identity is not verified", RETURNED_RUN_FOLDER
            ).refusal(found_text(receipt_payload.get("identity_verified"), True))
        )
