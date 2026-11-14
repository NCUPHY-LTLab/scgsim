# long-file: the receipt, the ending it states and the dry run travel to the solver host as one standalone script.

"""The receipt recorder of a manual Palace run: it runs in the run folder with only a Python interpreter.

The run script calls it twice: ``verify-config`` before the solve, which records Palace's dry run of
``config.json``, and then with the solver's exit codes, which writes the returned-run receipt.  For a run with
a convergence plan the receipt states how the solver ended from the runner's decision record, which it binds by
hash.  The paths, problems and helpers it imports are the run-folder and convergence contracts; the copy
``prepare_handoff`` puts in the run folder carries their text in place of the imports, so the copy imports only
the standard library.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.contracts.palace_convergence import (
    ConvergenceDecision,
    completed_stop_problem,
)
from scgsim.contracts.palace_run_folder import (
    CONFIG,
    CONVERGENCE_DECISION,
    CONVERGENCE_PLAN,
    HANDOFF_METADATA,
    NATIVE_METADATA,
    NATIVE_METADATA_NAME,
    OUTPUT_DIR,
    RESOURCE_RECORD,
    RETURNED_RECEIPT,
    RETURNED_RECEIPT_SCHEMA,
    RETURNED_RECEIPT_SCHEMA_VERSION,
    RUN_METADATA,
    RUNTIME_COMPATIBILITY,
    RUNTIME_COMPATIBILITY_SCHEMA,
    WHERE_RUN_FOLDER,
    FileRecord,
    JobIdentity,
    ObservedIdentity,
    ReceiptStatus,
    ReturnedReceipt,
    RunEnding,
    canonical_sha256,
    collects_adapted_meshes,
    confined_path,
    iteration_number,
    output_table,
    palace_problem,
    read_json,
    record_for_path,
    refused,
    sha256,
    table_name,
    utc_timestamp,
    write_json_atomic,
)

_WHERE_RUN_SCRIPT = "the run script that calls the recorder: prepare the handoff again"
"""Where a refused recorder call is set right: ``prepare_handoff`` writes the script and its arguments."""


def _script_main() -> None:
    """Run the step the run script asks for: ``verify-config`` before the solve, the receipt after it."""
    if sys.argv[1:2] == ["verify-config"]:
        raise SystemExit(_verify_config(sys.argv[2:]))
    raise SystemExit(_record_return(_return_request(sys.argv[1:])))


# ---- The returned-run receipt -------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReturnRequest:
    """What the run script tells the recorder about one execution stage, from its arguments.

    ``exit_code`` is the stage's pipeline exit code; ``solver_exit_code`` is given only when the solver ran.
    """

    root: Path
    exit_code: int
    log_path: str
    job: JobIdentity
    execution_stage: Literal["config_compatibility", "solver"]
    preflight_exit_code: int
    solver_exit_code: int | None
    tee_exit_code: int | None
    diagnostic_after_the_fact: bool
    receipt_path: str

    @property
    def solver_invoked(self) -> bool:
        """Whether the solver ran: the preflight passed and the script reached the solve."""
        return self.execution_stage == "solver"


@dataclass(frozen=True)
class _Handoff:
    """The handoff metadata the receipt is written against, with the fields the receipt reads checked."""

    metadata: dict[str, Any]
    route: str
    problem: str
    input_hashes: list[dict[str, Any]]
    handoff_id: str


def _record_return(request: _ReturnRequest) -> int:
    """Write the returned-run receipt of one execution stage, and the status into the two run records.

    The exit code is 0 for a completed or diagnostic receipt, else the stage's exit code (3 when the stage
    exited 0 but the receipt failed).
    """
    handoff = _read_handoff(request)
    receipt = _receipt(request, handoff)
    for path in (RESOURCE_RECORD, RUN_METADATA):
        _record_status(request, handoff, receipt, path)
    write_json_atomic(confined_path(request.root, request.receipt_path), receipt)
    if receipt["status"] == "completed" or request.diagnostic_after_the_fact:
        return 0
    return request.exit_code if request.exit_code != 0 else 3


def _return_request(argv: list[str]) -> _ReturnRequest:
    """The recorder call of one execution stage; exit codes that contradict the stage are refused."""
    parser = argparse.ArgumentParser(
        description="Write a returned Palace run receipt for manual handoff.",
    )
    parser.add_argument("run_dir", type=Path, help="The run folder of the handoff.")
    parser.add_argument("exit_code", type=int, help="Palace process exit code.")
    parser.add_argument("--log-path", dest="log_path", required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument(
        "--execution-stage",
        choices=("config_compatibility", "solver"),
        required=True,
    )
    parser.add_argument("--preflight-exit-code", type=int, required=True)
    parser.add_argument("--solver-exit-code", type=int, default=None)
    parser.add_argument("--tee-exit-code", type=int, default=None)
    parser.add_argument("--diagnostic-after-the-fact", action="store_true")
    parser.add_argument("--receipt-path", default=RETURNED_RECEIPT)
    args = parser.parse_args(argv)
    _check_stage_exit_codes(args)
    return _ReturnRequest(
        root=args.run_dir.expanduser().resolve(),
        exit_code=args.exit_code,
        log_path=args.log_path,
        job={"job_id": args.job_id, "job_name": args.job_name},
        execution_stage=args.execution_stage,
        preflight_exit_code=args.preflight_exit_code,
        solver_exit_code=args.solver_exit_code,
        tee_exit_code=args.tee_exit_code,
        diagnostic_after_the_fact=args.diagnostic_after_the_fact,
        receipt_path=args.receipt_path,
    )


def _check_stage_exit_codes(args: argparse.Namespace) -> None:
    """Refuse a solver stage without a passed preflight and its exit code, or a preflight stage with one."""
    stage = f"recorder call --execution-stage {args.execution_stage}"
    if args.execution_stage == "solver":
        if args.preflight_exit_code != 0:
            raise ValueError(
                refused(
                    stage,
                    f"needs a passed preflight, got --preflight-exit-code {args.preflight_exit_code}",
                    _WHERE_RUN_SCRIPT,
                )
            )
        if args.solver_exit_code is None:
            raise ValueError(
                refused(
                    stage, "needs the solver's --solver-exit-code", _WHERE_RUN_SCRIPT
                )
            )
    elif args.solver_exit_code is not None:
        raise ValueError(
            refused(
                stage,
                "must not pass --solver-exit-code: the solver did not run",
                _WHERE_RUN_SCRIPT,
            )
        )


def _read_handoff(request: _ReturnRequest) -> _Handoff:
    """The handoff metadata of the run folder; a diagnostic receipt may stand in for a missing handoff id."""
    metadata = read_json(request.root / HANDOFF_METADATA)
    route = metadata.get("route")
    problem = metadata.get("problem")
    if not isinstance(route, str) or route not in {"A", "B"}:
        raise ValueError(
            refused(
                f"{HANDOFF_METADATA} route",
                f"must be 'A' or 'B', found {route!r}",
                WHERE_RUN_FOLDER,
            )
        )
    if not isinstance(problem, str) or not problem:
        raise ValueError(
            refused(
                f"{HANDOFF_METADATA} problem",
                f"must be a non-empty name, found {problem!r}",
                WHERE_RUN_FOLDER,
            )
        )
    input_hashes = metadata.get("hashes")
    if not isinstance(input_hashes, list):
        raise TypeError(
            refused(f"{HANDOFF_METADATA} hashes", "must be a list", WHERE_RUN_FOLDER)
        )
    handoff_id = metadata.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        if not request.diagnostic_after_the_fact:
            raise ValueError(
                refused(
                    f"{HANDOFF_METADATA} handoff_id",
                    "is missing; only a --diagnostic-after-the-fact receipt is written without it",
                    WHERE_RUN_FOLDER,
                )
            )
        handoff_id = _diagnostic_handoff_id(
            route=route, problem=problem, input_hashes=input_hashes
        )
    return _Handoff(
        metadata=metadata,
        route=route,
        problem=problem,
        input_hashes=input_hashes,
        handoff_id=handoff_id,
    )


def _diagnostic_handoff_id(
    *, route: str, problem: str, input_hashes: list[dict[str, Any]]
) -> str:
    """The id of a handoff that recorded none, for a receipt written after the fact.

    Its ``diagnostic-`` start is a label for a person reading the receipt; no reader tests it, the
    receipt's ``diagnostic_unverified`` status is what keeps ``resolve`` from accepting it.
    """
    return "diagnostic-" + canonical_sha256(
        {"route": route, "problem": problem, "input_hashes": input_hashes}
    )


def _receipt(request: _ReturnRequest, handoff: _Handoff) -> ReturnedReceipt:
    """Observe the run's outputs and inputs, judge its status, and assemble its receipt."""
    root = request.root
    solver_identity = _solver_identity(root)
    compatibility_schema = _compatibility_schema(handoff.metadata)
    runtime_compatibility = _runtime_compatibility(root, compatibility_schema)
    config = read_json(root / CONFIG)
    ended = _how_it_ended(request, handoff)
    output_files = [
        record_for_path(root, path)
        for path in _output_paths(request, handoff, config, compatibility_schema, ended)
    ]
    log = record_for_path(root, request.log_path)
    status = _status(
        request,
        ended,
        outputs_present=all(record["present"] is True for record in output_files),
        inputs_match=_input_hashes_match(root, handoff.input_hashes),
        runtime_compatible=_runtime_compatibility_verified(
            root, compatibility_schema, runtime_compatibility
        ),
    )
    receipt: ReturnedReceipt = {
        "schema": RETURNED_RECEIPT_SCHEMA,
        "schema_version": RETURNED_RECEIPT_SCHEMA_VERSION,
        "handoff_id": handoff.handoff_id,
        "route": handoff.route,
        "problem": handoff.problem,
        "status": status,
        "exit_code": request.exit_code,
        "execution_stage": request.execution_stage,
        "solver_invoked": request.solver_invoked,
        "preflight_exit_code": request.preflight_exit_code,
        "solver_exit_code": request.solver_exit_code,
        "tee_exit_code": request.tee_exit_code,
        "ending": ended.ending,
        "identity_verified": status == "completed",
        "timestamp_utc": utc_timestamp(),
        "job_identity": request.job,
        "input_hashes": handoff.input_hashes,
        "output_files": output_files,
        "log": log,
        "convergence_decision": record_for_path(root, CONVERGENCE_DECISION)
        if ended.declared
        else None,
        "solver_identity": solver_identity,
    }
    if compatibility_schema is not None:
        receipt["runtime_compatibility"] = runtime_compatibility
    if handoff.metadata.get("route_a_thin_film") is not None:
        receipt["route_a_thin_film"] = handoff.metadata["route_a_thin_film"]
    return receipt


@dataclass(frozen=True)
class _Ended:
    """How the solver ended, as the recorder reads it, and why that cannot complete the run when it cannot.

    ``declared`` runs have a sealed convergence plan: their solver ran through the convergence runner, and the
    runner's ``decision`` says how it ended.  ``problem`` is set when the decision is missing, unreadable, for
    another plan, contradicts the exit code the script passed, or does not record a completed stop.
    """

    ending: RunEnding
    declared: bool
    decision: ConvergenceDecision | None = None
    problem: str | None = None

    @property
    def stopped(self) -> bool:
        """Whether the runner ended the solver during its adaptive loop, so the top level is no solve of its own."""
        return self.ending in (
            "stopped_at_convergence",
            "stop_escalated",
            "runner_terminated",
        )

    def required_result_folder(self) -> str | None:
        """The pass folder whose tables are the result of a run stopped at convergence, None otherwise."""
        if self.ending != "stopped_at_convergence" or self.decision is None:
            return None
        return self.decision.result_folder


def _how_it_ended(request: _ReturnRequest, handoff: _Handoff) -> _Ended:
    """The solver's ending: from the stage for an undeclared run, from the runner's decision for a declared one."""
    sealed = next(
        (
            entry
            for entry in handoff.input_hashes
            if isinstance(entry, dict) and entry.get("path") == CONVERGENCE_PLAN
        ),
        None,
    )
    if not request.solver_invoked:
        return _Ended("not_started", declared=sealed is not None)
    if sealed is None:
        return _Ended("solver_ended", declared=False)
    try:
        decision = ConvergenceDecision.from_payload(
            read_json(request.root / CONVERGENCE_DECISION)
        )
    except (OSError, ValueError, TypeError) as error:
        return _Ended(
            "not_recorded", declared=True, problem=f"{CONVERGENCE_DECISION}: {error}"
        )
    ending = decision.ending
    if ending is None or decision.solver is None:
        return _Ended(
            "not_recorded",
            declared=True,
            decision=decision,
            problem=f"{CONVERGENCE_DECISION} was written before the stop and never completed",
        )
    return _Ended(
        ending,
        declared=True,
        decision=decision,
        problem=_decision_problem(request, decision, sealed),
    )


def _decision_problem(
    request: _ReturnRequest, decision: ConvergenceDecision, sealed: dict[str, Any]
) -> str | None:
    """Why the runner's decision cannot complete the run, None when it can."""
    if decision.plan_sha256 != sealed.get("sha256"):
        return f"{CONVERGENCE_DECISION} is for another plan than the one the handoff sealed"
    if (
        decision.solver is not None
        and decision.solver.exit_code != request.solver_exit_code
    ):
        return (
            f"{CONVERGENCE_DECISION} records solver exit code {decision.solver.exit_code}, and the run "
            f"script passed {request.solver_exit_code}"
        )
    if decision.ending == "stopped_at_convergence":
        return completed_stop_problem(decision)
    return None


def _solver_identity(root: Path) -> dict[str, Any]:
    """The release and peak memory Palace recorded in ``palace.json``; empty when Palace wrote none."""
    palace_json_path = root / NATIVE_METADATA
    if not palace_json_path.is_file():
        return {}
    payload = read_json(palace_json_path)
    return {
        k: payload[k]
        for k in ("GitTag", "PeakMemoryMegabytes", "PeakNodeMemoryMegabytes")
        if k in payload
    }


def _compatibility_schema(handoff_metadata: dict[str, Any]) -> str | None:
    """The config schema the run had to be dry-run against before the solve; None when no probe was asked."""
    subject = f"{HANDOFF_METADATA} palace_identity"
    palace_identity = handoff_metadata.get("palace_identity")
    if not isinstance(palace_identity, dict):
        raise TypeError(refused(subject, "must be a JSON object", WHERE_RUN_FOLDER))
    probe = palace_identity.get("config_compatibility_probe")
    if probe is None:
        return None
    expected = {
        "config_compatibility_probe": "native_dry_run",
        "config_compatibility_semantics": "required_before_solve",
        "runtime_identity_semantics": "observed_provenance_only",
    }
    for field, value in expected.items():
        if palace_identity.get(field) != value:
            raise ValueError(
                refused(
                    f"{subject} {field}",
                    f"must be {value!r}, found {palace_identity.get(field)!r}",
                    WHERE_RUN_FOLDER,
                )
            )
    config_schema = palace_identity.get("config_schema")
    if not isinstance(config_schema, str) or not config_schema:
        raise ValueError(
            refused(
                f"{subject} config_schema", "must be a non-empty name", WHERE_RUN_FOLDER
            )
        )
    return config_schema


def _runtime_compatibility(
    root: Path, compatibility_schema: str | None
) -> dict[str, Any]:
    """The dry-run record ``verify-config`` wrote, as it is; empty when none was asked or none was written."""
    if compatibility_schema is None:
        return {}
    path = confined_path(root, RUNTIME_COMPATIBILITY)
    if not path.is_file():
        return {}
    return read_json(path)


def _output_paths(
    request: _ReturnRequest,
    handoff: _Handoff,
    config: dict[str, Any],
    compatibility_schema: str | None,
    ended: _Ended,
) -> list[str]:
    """Every output the receipt records: the tables, ``palace.json``, the log, the dry run, adapted meshes.

    A run's result is its top-level tables, each required.  A solver the runner ended during its adaptive loop
    left no solve of its own at the top level (links into its last pass, or a solve under way): those tables
    are recorded when present, and for a stop at convergence the tables of the last pass recorded are
    required instead.
    """
    families = _written_families(handoff.problem, config)
    result_folder = ended.required_result_folder()
    paths = [
        *(
            output_table(family)
            for family in families
            if not ended.stopped
            or confined_path(request.root, output_table(family)).is_file()
        ),
        NATIVE_METADATA,
        *([request.log_path] if request.log_path else []),
        *iteration_output_paths(request.root, handoff.problem, config=config),
        *(
            [
                *(output_table(family, result_folder) for family in families),
                f"{result_folder}/{NATIVE_METADATA_NAME}",
            ]
            if result_folder is not None
            else []
        ),
        *([RUNTIME_COMPATIBILITY] if compatibility_schema is not None else []),
    ]
    if collects_adapted_meshes(handoff.metadata.get("palace_identity")):
        paths.extend(adapted_mesh_output_paths(request.root, config))
    return list(dict.fromkeys(paths))


def _written_families(problem: str, config: dict[str, Any]) -> tuple[str, ...]:
    """The output families Palace writes for ``problem`` from this ``config``: its ports and Surface-EPR rows."""
    record = palace_problem(problem)
    boundaries = config.get("Boundaries")
    if not isinstance(boundaries, dict):
        raise TypeError(
            refused(f"{CONFIG} Boundaries", "must be a JSON object", WHERE_RUN_FOLDER)
        )
    ports = boundaries.get(record.excitation, []) if record.port_families else []
    postprocessing = boundaries.get("Postprocessing", {})
    if not isinstance(ports, list) or not isinstance(postprocessing, dict):
        raise TypeError(
            refused(
                f"{CONFIG} Boundaries",
                f"must hold {record.excitation} as a list and Postprocessing as a JSON object",
                WHERE_RUN_FOLDER,
            )
        )
    surfaces = postprocessing.get("Dielectric", [])
    if not isinstance(surfaces, list):
        raise TypeError(
            refused(
                f"{CONFIG} Boundaries.Postprocessing.Dielectric",
                "must be a list",
                WHERE_RUN_FOLDER,
            )
        )
    return record.written_families(lumped_ports=bool(ports), surfaces=bool(surfaces))


def iteration_output_paths(
    root: Path, problem: str, *, config: dict[str, Any]
) -> list[str]:
    """The output tables and ``palace.json`` present in the ``iteration<N>`` folders, by pass number."""
    names = [
        *(table_name(family) for family in _written_families(problem, config)),
        NATIVE_METADATA_NAME,
    ]
    return [
        (directory / name).relative_to(root).as_posix()
        for directory in _iteration_directories(root)
        for name in names
        if (directory / name).is_file()
    ]


def _iteration_directories(root: Path) -> list[Path]:
    """The ``iteration<N>`` folders of the output folder, by pass number."""
    results = root / OUTPUT_DIR
    if not confined_path(root, OUTPUT_DIR).is_dir():
        return []
    numbered: list[tuple[int, Path]] = []
    for path in results.iterdir():
        number = iteration_number(path.name)
        if (
            number is not None
            and confined_path(root, path.relative_to(root).as_posix()).is_dir()
        ):
            numbered.append((number, path))
    return [path for _, path in sorted(numbered, key=lambda item: item[0])]


def adapted_mesh_output_paths(root: Path, config: dict[str, Any]) -> list[str]:
    """The adapted meshes Palace declares it saved, each after the ``palace.json`` that declares it.

    Palace 0.18.1 writes the ``Model.Mesh`` stem plus ``.meshgz`` (zlib) or ``.mesh`` next to
    ``palace.json``, in the output folder and in each ``iteration<N>`` folder (``SaveAdaptIterations`` can
    leave confined links there).  Without a ``SaveAdaptMesh`` request no adapted mesh is expected, and the
    sealed input mesh is never among the paths.
    """
    if config.get("Model", {}).get("Refinement", {}).get("SaveAdaptMesh") is not True:
        return []
    mesh_name = Path(config["Model"]["Mesh"]).stem
    if not confined_path(root, OUTPUT_DIR).is_dir():
        return []
    outputs: list[str] = []
    for directory in [root / OUTPUT_DIR, *_iteration_directories(root)]:
        outputs.extend(_adapted_mesh_paths_in(root, directory, mesh_name))
    return outputs


def _adapted_mesh_paths_in(root: Path, directory: Path, mesh_name: str) -> list[str]:
    """The ``palace.json`` of one output folder and the adapted mesh it declares; none when it declares none."""
    metadata_relative = (directory / NATIVE_METADATA_NAME).relative_to(root).as_posix()
    metadata_path = confined_path(root, metadata_relative)
    if not metadata_path.is_file():
        return []  # The receipt records the missing palace.json among the required outputs.
    metadata = read_json(metadata_path)
    if "SavedAdaptedMesh" not in metadata:
        return []
    if not isinstance(metadata["SavedAdaptedMesh"], dict):
        raise TypeError(
            refused(
                f"{metadata_relative} SavedAdaptedMesh",
                "must be a JSON object",
                WHERE_RUN_FOLDER,
            )
        )
    alternatives = [
        (directory / f"{mesh_name}{suffix}").relative_to(root).as_posix()
        for suffix in (".meshgz", ".mesh")
    ]
    present = [
        relative for relative in alternatives if confined_path(root, relative).is_file()
    ]
    if len(present) > 1:
        raise ValueError(
            refused(
                f"adapted mesh {mesh_name} declared in {metadata_relative}",
                f"has ambiguous compression variants: both {present[0]} and {present[1]} are present; remove the stale one",
                WHERE_RUN_FOLDER,
            )
        )
    # A missing mesh is recorded under both names as not present, rather than guessing which compression
    # the Palace build was configured with.
    return [metadata_relative, *(present if present else alternatives)]


def _status(
    request: _ReturnRequest,
    ended: _Ended,
    *,
    outputs_present: bool,
    inputs_match: bool,
    runtime_compatible: bool,
) -> ReceiptStatus:
    """Completed when the inputs match, the outputs are present, the dry run passed and the solver ended well.

    The solver ended well in one of two ways.  ``solver_ended``: every exit code is 0, as for every run.
    ``stopped_at_convergence``: the runner's decision records a completed stop and the script's exit code is
    the solver's own, whatever the stop gave.  The output capture must have exited 0 in both.  Everything else
    is ``failed``.
    """
    if request.diagnostic_after_the_fact:
        return "diagnostic_unverified"
    if not (
        request.solver_invoked
        and request.preflight_exit_code == 0
        and request.tee_exit_code == 0
        and inputs_match
        and outputs_present
        and runtime_compatible
        and ended.problem is None
    ):
        return "failed"
    if ended.ending == "solver_ended":
        ended_well = request.exit_code == 0 and request.solver_exit_code == 0
    else:
        ended_well = (
            ended.ending == "stopped_at_convergence"
            and request.exit_code == request.solver_exit_code
        )
    return "completed" if ended_well else "failed"


def _input_hashes_match(root: Path, input_hashes: list[dict[str, Any]]) -> bool:
    """Whether every sealed input the handoff hashed is still present with the same size and SHA-256."""
    for entry in input_hashes:
        path_name = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")
        if not isinstance(path_name, str) or not isinstance(expected_bytes, int):
            return False
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            return False
        path = confined_path(root, path_name)
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or sha256(path) != expected_sha
        ):
            return False
    return bool(input_hashes)


def _runtime_compatibility_verified(
    root: Path, compatibility_schema: str | None, runtime_compatibility: dict[str, Any]
) -> bool:
    """Whether the dry run passed on the ``config.json`` the run folder holds now, for the declared schema."""
    if compatibility_schema is None:
        return True
    if runtime_compatibility.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA:
        return False
    config = runtime_compatibility.get("config")
    compatibility = runtime_compatibility.get("compatibility")
    if not isinstance(config, dict) or not isinstance(compatibility, dict):
        return False
    if (
        config.get("path") != CONFIG
        or config.get("declared_schema") != compatibility_schema
    ):
        return False
    current = record_for_path(root, CONFIG)
    if (
        current["present"] is not True
        or config.get("bytes") != current["bytes"]
        or config.get("sha256") != current["sha256"]
    ):
        return False
    return (
        compatibility.get("method") == "native_dry_run"
        and compatibility.get("probe_exit_code") == 0
        and compatibility.get("verified") is True
    )


def _record_status(
    request: _ReturnRequest,
    handoff: _Handoff,
    receipt: ReturnedReceipt,
    relative: str,
) -> None:
    """Write the receipt's status, exit code and job into run record ``relative`` of the same handoff."""
    path = request.root / relative
    record = read_json(path)
    if record.get("handoff_id") != handoff.metadata.get("handoff_id"):
        raise ValueError(
            refused(
                f"{relative} handoff_id",
                f"{record.get('handoff_id')!r} does not match {HANDOFF_METADATA} ({handoff.metadata.get('handoff_id')!r})",
                WHERE_RUN_FOLDER,
            )
        )
    record.update(
        {
            "status": receipt["status"],
            "exit_code": request.exit_code,
            "job_identity": receipt["job_identity"],
            "returned_receipt_path": RETURNED_RECEIPT,
        }
    )
    write_json_atomic(path, record)


# ---- The dry run before the solve ---------------------------------------------------------------------

_SUPPORTED_PALACE_RELEASE = "0.18.1"
_SUPPORTED_PALACE_COMMIT = "0dc74cdf8c36c58b69b21c4a06e816048ec0b83f"
_SUPPORTED_PALACE_TAGS = frozenset(
    {
        _SUPPORTED_PALACE_RELEASE,
        "v" + _SUPPORTED_PALACE_RELEASE,
        # A build from a git checkout prints its commit, abbreviated to at least git's 7 hex digits.
        *(_SUPPORTED_PALACE_COMMIT[:length] for length in range(7, 41)),
    }
)
"""The ``Palace version`` lines of the supported upstream release; recognising one attests no build."""


@dataclass(frozen=True)
class _ProbeRequest:
    """The ``verify-config`` call: the Palace command, the config to dry-run and where to record it."""

    executable: str
    command_style: Literal["binary", "wrapper"]
    output_path: str
    config_path: str
    config_schema: str

    def command(self, *arguments: str) -> list[str]:
        """The Palace command with ``arguments``; the wrapper runs one process with ``-serial``."""
        serial = ["-serial"] if self.command_style == "wrapper" else []
        return [self.executable, *serial, *arguments]


@dataclass(frozen=True)
class _ProbeOutput:
    """What one probe of Palace returned: its exit code (None when it could not start) and its output."""

    exit_code: int | None
    text: str


@dataclass(frozen=True)
class _DryRun:
    """Palace's dry run of the config: its exit code, the config's record, and whether it passed unchanged."""

    exit_code: int | None
    config: FileRecord
    verified: bool


def _verify_config(argv: list[str]) -> int:
    """Dry-run Palace on the sealed config and record what it met; exit 78 when the config is not accepted."""
    probe = _probe_request(argv)
    observed = _observe_palace(probe)
    root = Path.cwd().resolve()
    dry_run = _dry_run(probe, root)
    write_json_atomic(
        confined_path(root, probe.output_path),
        _compatibility_record(probe, observed, dry_run),
    )
    tag = observed["native_git_tag"]
    observed_label = tag if tag is not None else "unavailable"
    print(
        "SCGSim Palace config compatibility: "
        f"schema={probe.config_schema} dry_run_exit={dry_run.exit_code} "
        f"compatible={str(dry_run.verified).lower()} native_git_tag={observed_label} "
        f"binary_sha_recorded={str(observed['runtime_binary_sha256'] is not None).lower()}"
    )
    return 0 if dry_run.verified else 78


def _probe_request(argv: list[str]) -> _ProbeRequest:
    """The ``verify-config`` call the run script makes."""
    parser = argparse.ArgumentParser(
        description="Verify that Palace accepts the sealed config before solving."
    )
    parser.add_argument("executable")
    parser.add_argument("command_style", choices=("binary", "wrapper"))
    parser.add_argument("output_path")
    parser.add_argument("config_path")
    parser.add_argument("--config-schema", required=True)
    args = parser.parse_args(argv)
    return _ProbeRequest(
        executable=args.executable,
        command_style=args.command_style,
        output_path=args.output_path,
        config_path=args.config_path,
        config_schema=args.config_schema,
    )


def _observe_palace(probe: _ProbeRequest) -> ObservedIdentity:
    """The release and schema Palace prints for ``--version``, and the binary it runs from."""
    version = _run_probe(probe.command("--version"))
    native_git_tag = _single_line_value("Palace version", version.text)
    binary = (
        _resolved_executable(probe.executable)
        if probe.command_style == "binary"
        else None
    )
    supported = version.exit_code == 0 and native_git_tag in _SUPPORTED_PALACE_TAGS
    return {
        "native_git_tag": native_git_tag,
        "supported_upstream_release": _SUPPORTED_PALACE_RELEASE if supported else None,
        "native_schema_version": _single_line_value("Schema version", version.text),
        "version_probe_exit_code": version.exit_code,
        "runtime_binary_identity": binary.name if binary is not None else None,
        "runtime_binary_sha256": sha256(binary) if binary is not None else None,
    }


def _single_line_value(label: str, text: str) -> str | None:
    """The value of the one ``<label>: <value>`` line of ``text``; None when there is none or more than one."""
    matches = re.findall(rf"(?m)^{re.escape(label)}: ([^\s]+)\s*$", text)
    return matches[0] if len(matches) == 1 else None


def _dry_run(probe: _ProbeRequest, root: Path) -> _DryRun:
    """Palace's ``--dry-run`` of the config; it passes only when Palace accepts it and leaves it unchanged."""
    before = record_for_path(root, probe.config_path)
    exit_code = _run_probe(probe.command("--dry-run", probe.config_path)).exit_code
    after = record_for_path(root, probe.config_path)
    verified = exit_code == 0 and after["present"] is True and after == before
    return _DryRun(exit_code=exit_code, config=after, verified=verified)


def _compatibility_record(
    probe: _ProbeRequest, observed: ObservedIdentity, dry_run: _DryRun
) -> dict[str, Any]:
    """The runtime-compatibility record, in the order it is written."""
    return {
        "schema": RUNTIME_COMPATIBILITY_SCHEMA,
        "config": {
            "path": dry_run.config["path"],
            "bytes": dry_run.config["bytes"],
            "sha256": dry_run.config["sha256"],
            "declared_schema": probe.config_schema,
        },
        "compatibility": {
            "method": "native_dry_run",
            "probe_exit_code": dry_run.exit_code,
            "verified": dry_run.verified,
        },
        "observed_identity": observed,
        "command": {
            "command_style": probe.command_style,
            "executable_identity": Path(probe.executable).name,
        },
    }


def _run_probe(command: list[str]) -> _ProbeOutput:
    """Run one Palace probe and keep its exit code and combined output; a command that cannot start gives None."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return _ProbeOutput(exit_code=None, text="")
    return _ProbeOutput(exit_code=completed.returncode, text=completed.stdout)


def _resolved_executable(executable: str) -> Path | None:
    """The file ``executable`` runs from, found on PATH when it names no folder; None when there is none."""
    candidate = (
        Path(executable)
        if Path(executable).parent != Path(".")
        else Path(shutil.which(executable) or "")
    )
    if not candidate:
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_file() else None


if __name__ == "__main__":
    _script_main()
