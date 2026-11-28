"""Package a prepared HFSS EPR model, run its solver-only attempt and collect the result.

``prepare_native_batch`` packages a released prepare-only model, ``run_native_batch``
solves it with a standard-library runner and ``collect_native_batch`` collects the
result on an owned copy. The installed and transported solver runners share one
standard-library protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .batch_stages import (
    NativeBatchPackage,  # noqa: F401 -- preserve the existing batch module result type.
    collect_native_batch,
    prepare_native_batch,
)
from .runtime import batch_protocol as protocol
from .runtime.batch_runner import run_native_batch as _run_native_batch
from .specs.common import AedtResources


def run_native_batch(package_dir: str | Path, *, output_dir: str | Path) -> Path:
    """Delegate one standard-library solver-only attempt to its runner."""
    return _run_native_batch(package_dir, output_dir=output_dir)


def main(argv: list[str] | None = None) -> int:
    """Run one requested package, solve, or collection command."""
    parser = argparse.ArgumentParser(
        description="Prepared-native HFSS EPR batch stages"
    )
    commands = parser.add_subparsers(dest="stage", required=True)
    package = commands.add_parser("package")
    package.add_argument("--prepared-run", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--solver-executable", required=True)
    package.add_argument("--cores", required=True, type=int)
    package.add_argument("--ram-limit-percent", required=True, type=int)
    package.add_argument("--dependency", action="append", default=[])
    solve = commands.add_parser("solve")
    solve.add_argument("--package", required=True)
    solve.add_argument("--output", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--package", required=True)
    collect.add_argument("--attempt", required=True)
    collect.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.stage == "solve":
        result = run_native_batch(args.package, output_dir=args.output)
        print(result)
        payload = protocol.object_payload(protocol.read_json(result))
        return 0 if payload["returncode"] == 0 else 1
    if args.stage == "package":
        result = prepare_native_batch(
            args.prepared_run,
            output_dir=args.output,
            solver_executable=args.solver_executable,
            resources=AedtResources(args.cores, args.ram_limit_percent),
            dependencies=args.dependency,
        )
        print(result.manifest_path)
    else:
        result = collect_native_batch(
            args.package, args.attempt, output_dir=args.output
        )
        print(result.receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
