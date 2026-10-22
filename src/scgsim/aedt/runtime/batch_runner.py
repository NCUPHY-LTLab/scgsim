"""One solver-only process and CLI; installed or transported with its protocol."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import cast
import uuid


def _check_standalone_protocol() -> None:
    # Verify the exact helper member before importing code from the package.
    root = Path(__file__).resolve().parent
    raw: object = json.loads((root / "native-batch-package.json").read_text(encoding="utf-8"))
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
    if len(payload) != matches[0].get("bytes") or hashlib.sha256(payload).hexdigest() != matches[0].get("sha256"):
        raise RuntimeError("standalone protocol member identity differs")


if __package__:
    from . import batch_protocol as protocol
else:
    _check_standalone_protocol()
    # Transported members are immutable inputs to strict inventory verification.
    sys.dont_write_bytecode = True
    import batch_protocol as protocol


def run_native_batch(package_dir: str | Path, *, output_dir: str | Path) -> Path:
    """Run once without Desktop/PyAEDT; preserve interrupted ownership as unsettled."""
    package_root = Path(package_dir).expanduser().resolve()
    package = protocol.read_package(package_root)
    output = protocol.fresh(output_dir, package_root)
    attempt = protocol.BatchAttempt(
        attempt_id=str(uuid.uuid4()), package_sha256=protocol.sha256(package_root / protocol.MANIFEST),
        package_id=package.package_id, status="not_started",
        requested_aedt_version=package.requested_aedt_version, actual_aedt_version=None,
        version_observation="not_queried; raw solver stdout/stderr retained",
        execution_location=protocol.object_record({"hostname": socket.gethostname(), "directory": str(output)}),
        resources_requested=package.resources,
        resource_semantics="native MachineList per-host solver limits; not an OS hard cap",
        started_at_unix=time.time(),
    )
    protocol.write_json(output / "batch-attempt.json", attempt.to_payload())
    process: subprocess.Popen[bytes] | None = None
    try:
        work = output / "work"
        native = package_root / "native"
        protocol.copy_members(native, work, protocol.inventory(native))
        resource = package.resources
        machine = f"list={socket.gethostname()}:-1:{resource.cores}:{resource.ram_limit_percent}%"
        executable = shutil.which(package.recipe.executable)
        if executable is None:
            raise FileNotFoundError("bound AEDT batch executable is unavailable")
        argv = (executable, "-ng", "-UseElectronicsPPE", "-distributed", "-auto",
                "-machinelist", machine, "-BatchSolve", package.recipe.target,
                str(protocol.inside(work, package.project)))
        attempt = replace(attempt, argv=argv, executable_sha256=protocol.sha256(Path(executable)), status="starting")
        protocol.write_json(output / "batch-attempt.json", attempt.to_payload())
        with (output / "solver.stdout").open("xb") as stdout, (output / "solver.stderr").open("xb") as stderr:
            process = subprocess.Popen(argv, cwd=work, stdout=stdout, stderr=stderr)
            attempt = replace(attempt, status="running", child_pid=process.pid)
            protocol.write_json(output / "batch-attempt.json", attempt.to_payload())
            returncode = process.wait()
        raw_evidence = tuple(protocol.Member(name, (output / name).stat().st_size, protocol.sha256(output / name))
                             for name in ("solver.stdout", "solver.stderr"))
        attempt = replace(attempt, status="external_process_terminal", returncode=returncode,
                          finished_at_unix=time.time(), work_members=protocol.inventory(work), raw_evidence=raw_evidence)
        protocol.write_json(output / "batch-attempt.json", attempt.to_payload())
        return output / "batch-attempt.json"
    except BaseException as exc:
        attempt = replace(attempt, status="unsettled" if process is not None and process.returncode is None else "failed",
                          error=f"{type(exc).__name__}: {exc}")
        protocol.write_json(output / "batch-attempt.json", attempt.to_payload())
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepared-native HFSS EPR solver-only attempt")
    commands = parser.add_subparsers(dest="stage", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--package", required=True)
    solve.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_native_batch(cast(str, args.package), output_dir=cast(str, args.output))
    attempt = protocol.BatchAttempt.from_payload(protocol.read_json(result))
    print(result)
    return 0 if attempt.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
