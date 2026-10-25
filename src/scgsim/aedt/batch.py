"""Prepared-native HFSS EPR packaging, solver-only attempts and owned collection.

Native model preparation and collection remain transaction-owned operations.
The installed and transported solver runners share one standard-library protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tarfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .runtime import batch_protocol as protocol
from .runtime.batch_runner import run_native_batch as _run_native_batch

if TYPE_CHECKING:
    from scgsim.aedt.specs.common import AedtResources
    from scgsim.aedt.results.resolve import ResolvedRun

_PACKAGE = protocol.PACKAGE_SCHEMA
_ATTEMPT = protocol.ATTEMPT_SCHEMA
_MANIFEST = protocol.MANIFEST


# Compatibility helpers delegate to the one internal file protocol codec.
def _sha(path: Path) -> str:
    return protocol.sha256(path)


def _read(path: Path) -> dict:
    return protocol.object_payload(protocol.read_json(path))


def _write(path: Path, value: dict) -> None:
    protocol.write_json(path, value)


def _inside(root: Path, relative: str) -> Path:
    return protocol.inside(root, relative)


def _fresh(path: str | Path, *sources: Path) -> Path:
    return protocol.fresh(path, *sources)


def _inventory(root: Path) -> list[dict]:
    return [item.to_payload() for item in protocol.inventory(root)]


def _verify(root: Path, records: list[dict]) -> None:
    protocol.verify_members(root, protocol.members(records))


def _copy_members(source: Path, target: Path, records: list[dict]) -> None:
    protocol.copy_members(source, target, protocol.members(records))


def _resources(value: dict) -> dict:
    return protocol.Resources.from_payload(value).to_payload()


@dataclass(frozen=True)
class NativeBatchPackage:
    """Paths to a source-bound prepared-native package and its portable archive."""
    root: Path
    manifest_path: Path
    archive_path: Path


def prepare_native_batch(
    prepared_run: str | Path, *, output_dir: str | Path,
    solver_executable: str, resources: AedtResources,
    dependencies: Sequence[str] = (),
) -> NativeBatchPackage:
    """Package a released prepare-only EPR model; never launch a solver.

    dependencies explicitly lists additional run-relative native dependency
    files/directories. The saved .aedtresults tree, if present, travels in full.
    The package does not include an AEDT installation, license or scheduler.
    """
    from scgsim.aedt.results.prepared import resolve_prepared_model
    from scgsim.aedt.specs.common import AedtResources
    from scgsim.aedt.specs.hfss import HfssEprSpec
    from scgsim.aedt.specs.parse import parse_aedt_spec
    from scgsim.aedt.results.provenance import initial_receipt_payload, encode_initial_receipt
    from scgsim.aedt.preparation.cohort import canonical_handoff_paths

    root = Path(prepared_run).expanduser().resolve()
    verified = resolve_prepared_model(root)
    receipt = _read(verified.receipt_path)
    spec = parse_aedt_spec(_read(root / "aedt_spec.json"), base_dir=root)
    if not isinstance(spec, HfssEprSpec) or spec.epr_request is None or spec.run_control is None:
        raise TypeError("native batch requires a complete body-first EPR preparation")
    if not isinstance(resources, AedtResources):
        raise TypeError("resources must be AedtResources")
    resource_payload = _resources(resources.to_payload())
    if not isinstance(solver_executable, str) or not solver_executable or "\0" in solver_executable:
        raise ValueError("solver_executable must be an explicit executable path or command")
    if isinstance(dependencies, (str, bytes)):
        raise TypeError("dependencies must be a sequence of run-relative paths")
    from scgsim.aedt.epr.evidence import PreparedEprEvidence

    PreparedEprEvidence.from_payload(
        receipt, evidence_root=root / "metadata/epr_expressions",
    )
    output = _fresh(output_dir, root)
    try:
        inputs = output / "inputs"
        inputs.mkdir()
        initial = initial_receipt_payload(
            schema_version=receipt["schema_version"], mode=receipt["mode"],
            requested=receipt["requested"], pdk_materials=receipt["pdk_materials"],
            vacuum_material_id=receipt["vacuum_material_id"], source=receipt["source"],
            prepared_runtime_source_value=receipt["prepared_runtime_source"], outputs={},
            prepared_at_utc=receipt["prepared_at_utc"],
        )
        for relative in canonical_handoff_paths(spec):
            target = _inside(inputs, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == "metadata/aedt_run_receipt.json":
                target.write_bytes(encode_initial_receipt(initial))
            else:
                shutil.copy2(_inside(root, relative), target)
        original_manifest = _read(inputs / "metadata/aedt_handoff_manifest.json")
        _verify(inputs, original_manifest["members"])
        _write(output / "native-preparation-receipt.json", receipt)
        # Retain the complete authored calculator evidence without regenerating it.
        evidence = root / "metadata/epr_expressions"
        if not evidence.is_dir():
            raise RuntimeError("native preparation calculator evidence is missing")
        _copy_members(evidence, output / "epr-expressions", _inventory(evidence))
        native = output / "native"
        native.mkdir()
        project_relative = verified.project_path.relative_to(root).as_posix()
        required = [project_relative, *dependencies]
        results_relative = f"{spec.project_name}.aedtresults"
        if (root / results_relative).exists():
            required.append(results_relative)
        copied = set()
        for relative in required:
            source = _inside(root, relative)
            if source.is_dir():
                records = _inventory(source)
                for item in records:
                    child = (Path(relative) / item["path"]).as_posix()
                    if child in copied:
                        continue
                    target = _inside(native, child)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(_inside(root, child), target)
                    copied.add(child)
            elif source.is_file():
                if relative not in copied:
                    target = _inside(native, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied.add(relative)
            else:
                raise FileNotFoundError(f"declared native dependency unavailable: {relative}")
        runtime_dir = Path(protocol.__file__).parent
        shutil.copy2(runtime_dir / "batch_runner.py", output / "run_batch.py")
        shutil.copy2(runtime_dir / "batch_protocol.py", output / "batch_protocol.py")
        manifest = {
            "schema_version": _PACKAGE, "status": "prepared_native",
            "package_id": str(uuid.uuid4()), "created_at_unix": time.time(),
            "source_location": str(root), "requested_aedt_version": spec.aedt_version,
            "prepared_connected": receipt["connected"],
            "spec_sha256": _sha(inputs / "aedt_spec.json"),
            "model_sha256": spec.geometry.model_sha256,
            "source_sha256": spec.geometry.source_sha256,
            "project": project_relative, "results": results_relative,
            "design": spec.design_name, "setup": spec.run_control.setup_name,
            "resources": resource_payload,
            "recipe": {"executable": solver_executable,
                       "mode": "prepared_project_batchsolve_ppe",
                       "allocation": "one_host_native_auto",
                       "target": f"{spec.design_name}:Nominal:{spec.run_control.setup_name}"},
            "declared_dependencies": list(dependencies),
            "members": _inventory(output),
        }
        typed_manifest = protocol.PackageManifest.from_payload(manifest)
        _write(output / _MANIFEST, typed_manifest.to_payload())
        archive = output / "native_batch.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for item in [*manifest["members"], {"path": _MANIFEST}]:
                stream.add(_inside(output, item["path"]), arcname=item["path"], recursive=False)
        return NativeBatchPackage(output, output / _MANIFEST, archive)
    except BaseException as exc:
        _write(output / "package-failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _package(root: Path) -> protocol.PackageManifest:
    return protocol.read_package(root)


def run_native_batch(package_dir: str | Path, *, output_dir: str | Path) -> Path:
    """Delegate one standard-library solver-only attempt to its runner."""
    return _run_native_batch(package_dir, output_dir=output_dir)


def _attempt(package_root: Path, attempt_root: Path, package: protocol.PackageManifest) -> protocol.BatchAttempt:
    return protocol.read_attempt(package_root, attempt_root, package)


def collect_native_batch(
    package_dir: str | Path, attempt_dir: str | Path, *, output_dir: str | Path,
) -> ResolvedRun:
    """Collect full adaptive EPR on an owned copy; never Analyze or rebuild.

    A zero process exit permits native inspection, not completion. The existing
    transaction must observe complete native history, release and seal first.
    """
    from scgsim.aedt.runtime.transaction import _execute
    from scgsim.aedt.results.resolve import resolve_results

    package_root = Path(package_dir).expanduser().resolve()
    attempt_root = Path(attempt_dir).expanduser().resolve()
    package = _package(package_root)
    attempt = _attempt(package_root, attempt_root, package)
    output = _fresh(output_dir, package_root, attempt_root)
    try:
        _copy_members(package_root / "inputs", output, _inventory(package_root / "inputs"))
        parent = output / "batch-source"
        _copy_members(package_root, parent / "package", [*[item.to_payload() for item in package.members],
                      {"path": _MANIFEST, "bytes": (package_root / _MANIFEST).stat().st_size,
                       "sha256": _sha(package_root / _MANIFEST)}])
        _copy_members(attempt_root / "work", parent / "attempt/work", _inventory(attempt_root / "work"))
        # read_attempt established these inventories before creating the child stage.
        assert attempt.raw_evidence is not None and attempt.work_members is not None
        evidence = [*[item.to_payload() for item in attempt.raw_evidence], {"path": "batch-attempt.json",
                    "bytes": (attempt_root / "batch-attempt.json").stat().st_size,
                    "sha256": _sha(attempt_root / "batch-attempt.json")}]
        _copy_members(attempt_root, parent / "attempt", evidence)
        protocol.copy_members(attempt_root / "work", output, attempt.work_members)
        context = protocol.CollectionContext(
            package_sha256=_sha(package_root / _MANIFEST),
            attempt_sha256=_sha(attempt_root / "batch-attempt.json"),
        )
        _write(output / "metadata/batch-collection.json", context.to_payload())
        rc = _execute(output / "metadata/aedt_handoff_metadata.json", batch_collection=context)
        if rc:
            raise RuntimeError("native batch collection failed; inspect its retained run receipt")
        return resolve_results(output)
    except BaseException as exc:
        _write(output / "metadata/batch-collection-failure.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _collection_source(run_dir: Path, context: dict):
    return protocol.collection_source(run_dir, protocol.CollectionContext.from_payload(context))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepared-native HFSS EPR batch stages")
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
        from scgsim.aedt.specs.common import AedtResources
        result = prepare_native_batch(args.prepared_run, output_dir=args.output,
                    solver_executable=args.solver_executable,
                    resources=AedtResources(args.cores, args.ram_limit_percent),
                    dependencies=args.dependency)
        print(result.manifest_path)
    else:
        result = collect_native_batch(args.package, args.attempt, output_dir=args.output)
        print(result.receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
