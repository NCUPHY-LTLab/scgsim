"""One solver-only process and CLI; installed or transported with its protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast


def _check_standalone_protocol() -> None:
    # Verify the exact helper member before importing code from the package.
    root = Path(__file__).resolve().parent
    raw: object = json.loads(
        (root / "native-batch-package.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw, dict):
        raise TypeError("standalone package manifest requires an object")
    entries = cast(dict[str, object], raw).get("members")
    if not isinstance(entries, list):
        raise TypeError("standalone package manifest requires members")
    matches: list[dict[str, object]] = []
    for value in cast(list[object], entries):
        if isinstance(value, dict):
            item = cast(dict[str, object], value)
            if item.get("path") == "batch_protocol.py":
                matches.append(item)
    if len(matches) != 1:
        raise RuntimeError("standalone protocol member is missing or repeated")
    helper = root / "batch_protocol.py"
    if helper.is_symlink():
        raise ValueError("standalone protocol must not be a symlink")
    payload = helper.read_bytes()
    if len(payload) != matches[0].get("bytes") or hashlib.sha256(
        payload
    ).hexdigest() != matches[0].get("sha256"):
        raise RuntimeError("standalone protocol member identity differs")


# Installed execution uses the package protocol; transported execution must
# verify the copied helper bytes before Python imports that executable member.
if __package__:
    from . import batch_protocol as protocol
else:
    _check_standalone_protocol()
    # Transported members are immutable inputs to strict inventory verification.
    sys.dont_write_bytecode = True
    import batch_protocol as protocol


@dataclass
class AttemptExecution:
    """Attempt ledger plus the child process, retained through settlement."""

    attempt: protocol.BatchAttempt
    process: subprocess.Popen[bytes] | None = None


def _initial_attempt(
    package_root: Path, output: Path, package: protocol.PackageManifest
) -> protocol.BatchAttempt:
    """Record ownership and resource semantics before staging child files."""
    return protocol.BatchAttempt(
        attempt_id=str(uuid.uuid4()),
        package_sha256=protocol.sha256(package_root / protocol.MANIFEST),
        package_id=package.package_id,
        status="not_started",
        requested_aedt_version=package.requested_aedt_version,
        actual_aedt_version=None,
        version_observation="not_queried; raw solver stdout/stderr retained",
        execution_location=protocol.object_record(
            {"hostname": socket.gethostname(), "directory": str(output)}
        ),
        resources_requested=package.resources,
        resource_semantics="native MachineList per-host solver limits; not an OS hard cap",
        started_at_unix=time.time(),
    )


def _prepare_work(package_root: Path, output: Path) -> Path:
    """Copy the verified native input inventory into a fresh attempt directory."""
    work = output / "work"
    native = package_root / "native"
    protocol.copy_members(native, work, protocol.inventory(native))
    return work


def _batch_command(package: protocol.PackageManifest, work: Path) -> tuple[str, ...]:
    """Resolve the executable and construct the exact single-host AEDT argv."""
    resource = package.resources
    machine = (
        f"list={socket.gethostname()}:-1:{resource.cores}:{resource.ram_limit_percent}%"
    )
    executable = shutil.which(package.recipe.executable)
    if executable is None:
        raise FileNotFoundError("bound AEDT batch executable is unavailable")
    return (
        executable,
        "-ng",
        "-UseElectronicsPPE",
        "-distributed",
        "-auto",
        "-machinelist",
        machine,
        "-BatchSolve",
        package.recipe.target,
        str(protocol.inside(work, package.project)),
    )


def _run_child_process(
    execution: AttemptExecution,
    output: Path,
    work: Path,
    argv: tuple[str, ...],
) -> int:
    """Launch once, record its PID, and wait while retaining raw streams."""
    with (
        (output / "solver.stdout").open("xb") as stdout,
        (output / "solver.stderr").open("xb") as stderr,
    ):
        execution.process = subprocess.Popen(
            argv, cwd=work, stdout=stdout, stderr=stderr
        )
        execution.attempt = replace(
            execution.attempt,
            status="running",
            child_pid=execution.process.pid,
        )
        protocol.write_json(
            output / "batch-attempt.json", execution.attempt.to_payload()
        )
        return execution.process.wait()


def _terminal_evidence(output: Path) -> tuple[protocol.Member, ...]:
    """Hash both raw solver streams after the child reaches a terminal state."""
    return tuple(
        protocol.Member(
            name,
            (output / name).stat().st_size,
            protocol.sha256(output / name),
        )
        for name in ("solver.stdout", "solver.stderr")
    )


def record_terminal_attempt(
    execution: AttemptExecution, output: Path, work: Path, returncode: int
) -> None:
    """Persist settled process status with the full returned work inventory."""
    raw_evidence = _terminal_evidence(output)
    execution.attempt = replace(
        execution.attempt,
        status="external_process_terminal",
        returncode=returncode,
        finished_at_unix=time.time(),
        work_members=protocol.inventory(work),
        raw_evidence=raw_evidence,
    )
    protocol.write_json(output / "batch-attempt.json", execution.attempt.to_payload())


def _record_failed_attempt(
    execution: AttemptExecution, output: Path, exc: BaseException
) -> None:
    """Retain unsettled ownership only while the child has no terminal status."""
    process = execution.process
    state = (
        "unsettled" if process is not None and process.returncode is None else "failed"
    )
    execution.attempt = replace(
        execution.attempt,
        status=state,
        error=f"{type(exc).__name__}: {exc}",
    )
    protocol.write_json(output / "batch-attempt.json", execution.attempt.to_payload())


def run_native_batch(package_dir: str | Path, *, output_dir: str | Path) -> Path:
    """Run once without Desktop/PyAEDT; preserve interrupted ownership as unsettled."""
    package_root = Path(package_dir).expanduser().resolve()
    package = protocol.read_package(package_root)
    output = protocol.fresh(output_dir, package_root)
    execution = AttemptExecution(_initial_attempt(package_root, output, package))
    protocol.write_json(output / "batch-attempt.json", execution.attempt.to_payload())
    try:
        work = _prepare_work(package_root, output)
        argv = _batch_command(package, work)
        execution.attempt = replace(
            execution.attempt,
            argv=argv,
            executable_sha256=protocol.sha256(Path(argv[0])),
            status="starting",
        )
        protocol.write_json(
            output / "batch-attempt.json", execution.attempt.to_payload()
        )
        returncode = _run_child_process(execution, output, work, argv)
        record_terminal_attempt(execution, output, work, returncode)
        return output / "batch-attempt.json"
    except BaseException as exc:
        _record_failed_attempt(execution, output, exc)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the standalone solver-only attempt command."""
    parser = argparse.ArgumentParser(
        description="Prepared-native HFSS EPR solver-only attempt"
    )
    commands = parser.add_subparsers(dest="stage", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--package", required=True)
    solve.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_native_batch(
        cast(str, args.package), output_dir=cast(str, args.output)
    )
    attempt = protocol.BatchAttempt.from_payload(protocol.read_json(result))
    print(result)
    return 0 if attempt.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
