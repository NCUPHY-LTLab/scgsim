"""Prepared-native HFSS EPR packaging, solver-only attempts and owned collection.

Native model preparation and collection remain transaction-owned operations.
The installed and transported solver runners share one standard-library protocol.
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

_PACKAGE = protocol.PACKAGE_SCHEMA
_ATTEMPT = protocol.ATTEMPT_SCHEMA
_MANIFEST = protocol.MANIFEST


# Compatibility helpers delegate to the one internal file protocol codec.
def _sha(path: Path) -> str:
    return protocol.sha256(path)


def _read(path: Path) -> dict[str, object]:
    return protocol.object_payload(protocol.read_json(path))


def _write(path: Path, value: dict[str, object]) -> None:
    protocol.write_json(path, value)


def _inside(root: Path, relative: str) -> Path:
    return protocol.inside(root, relative)


def _fresh(path: str | Path, *sources: Path) -> Path:
    return protocol.fresh(path, *sources)


def _inventory(root: Path) -> list[dict[str, object]]:
    return [item.to_payload() for item in protocol.inventory(root)]


def _verify(root: Path, records: list[dict[str, object]]) -> None:
    protocol.verify_members(root, protocol.members(records))


def _copy_members(source: Path, target: Path, records: list[dict[str, object]]) -> None:
    protocol.copy_members(source, target, protocol.members(records))


def _resources(value: dict[str, object]) -> dict[str, object]:
    return protocol.Resources.from_payload(value).to_payload()


def _package(root: Path) -> protocol.PackageManifest:
    return protocol.read_package(root)


def run_native_batch(package_dir: str | Path, *, output_dir: str | Path) -> Path:
    """Delegate one standard-library solver-only attempt to its runner."""
    return _run_native_batch(package_dir, output_dir=output_dir)


def _attempt(
    package_root: Path, attempt_root: Path, package: protocol.PackageManifest
) -> protocol.BatchAttempt:
    return protocol.read_attempt(package_root, attempt_root, package)


def _collection_source(
    run_dir: Path, context: dict[str, object]
) -> tuple[protocol.PackageManifest, protocol.BatchAttempt, protocol.JsonObject]:
    return protocol.collection_source(
        run_dir, protocol.CollectionContext.from_payload(context)
    )


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
        return 0 if _read(result)["returncode"] == 0 else 1
    if args.stage == "package":
        from scgsim.aedt.specs.common import AedtResources  # noqa: PLC0415 -- construct resources only for the package CLI stage.

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
