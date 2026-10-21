"""Prepared-native HFSS EPR packaging, solver-only attempts and owned collection.

This file is also copied verbatim into each package as run_batch.py. Its solve
path uses only the standard library and never imports the AEDT Python backend.
Native model preparation and collection remain transaction-owned operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scgsim.aedt.specs.common import AedtResources
    from scgsim.aedt.results.resolve import ResolvedRun

_PACKAGE = "scgsim.aedt.native-batch-package.v1"
_ATTEMPT = "scgsim.aedt.native-batch-attempt.v1"
_MANIFEST = "native-batch-package.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _inside(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("batch member must be a contained relative path")
    path = root / rel
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("batch member escapes its root or is a symlink")
    return path


def _fresh(path: str | Path, *sources: Path) -> Path:
    result = Path(path).expanduser().resolve()
    for source in sources:
        if result.is_relative_to(source) or source.is_relative_to(result):
            raise ValueError("batch output and input roots must not overlap")
    result.mkdir(parents=True, exist_ok=False)
    return result


def _inventory(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("batch inventory contains a symlink or escaping member")
        if path.is_file():
            records.append({"path": path.relative_to(root).as_posix(),
                            "bytes": path.stat().st_size, "sha256": _sha(path)})
    return records


def _verify(root: Path, records: list[dict]) -> None:
    if not isinstance(records, list):
        raise TypeError("batch inventory must be a list")
    names = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("batch inventory member is not canonical")
        path = _inside(root, item["path"])
        if item["path"] in names:
            raise ValueError("batch inventory repeats a member")
        names.add(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise RuntimeError(f"batch member identity differs: {item['path']}")


def _copy_members(source: Path, target: Path, records: list[dict]) -> None:
    _verify(source, records)
    for item in records:
        destination = _inside(target, item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_inside(source, item["path"]), destination)
    _verify(target, records)


def _resources(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"cores", "ram_limit_percent"}:
        raise ValueError("batch resources require cores and ram_limit_percent")
    if type(value["cores"]) is not int or value["cores"] < 1:
        raise ValueError("batch cores must be a positive integer")
    # v232 MachineList explicitly represents integer percentages 1..99.
    if type(value["ram_limit_percent"]) is not int or not 1 <= value["ram_limit_percent"] <= 99:
        raise ValueError("native MachineList RAM percentage must be in 1..99")
    return dict(value)


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
    setup = receipt.get("setup")
    cache = receipt.get("cache")
    expressions = receipt.get("expressions")
    if not setup or not isinstance(cache, dict) or not cache.get("items") or not expressions:
        raise RuntimeError("native preparation lacks setup/cache/expression readback")
    if not cache.get("serialized_readback", {}).get("saved_configuration_source"):
        raise RuntimeError("native preparation lacks fresh saved cache readback")
    from scgsim.aedt.epr.fields import _validated_expression_readback

    _validated_expression_readback(
        expressions, cache, evidence_root=root / "metadata/epr_expressions",
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
        shutil.copy2(Path(__file__), output / "run_batch.py")
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
        _write(output / _MANIFEST, manifest)
        archive = output / "native_batch.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for item in [*manifest["members"], {"path": _MANIFEST}]:
                stream.add(_inside(output, item["path"]), arcname=item["path"], recursive=False)
        return NativeBatchPackage(output, output / _MANIFEST, archive)
    except BaseException as exc:
        _write(output / "package-failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _package(root: Path) -> dict:
    value = _read(root / _MANIFEST)
    if value.get("schema_version") != _PACKAGE or value.get("status") != "prepared_native":
        raise ValueError("native batch package is not prepared")
    _verify(root, value["members"])
    observed = [item for item in _inventory(root)
                if item["path"] not in {_MANIFEST, "native_batch.tar.gz"}]
    if observed != value["members"]:
        raise RuntimeError("batch package contains undeclared or changed files")
    _resources(value["resources"])
    recipe = value["recipe"]
    if recipe.get("mode") != "prepared_project_batchsolve_ppe" or recipe.get("allocation") != "one_host_native_auto":
        raise ValueError("unsupported native batch execution recipe")
    if recipe.get("target") != f"{value['design']}:Nominal:{value['setup']}":
        raise ValueError("batch target differs from prepared design/setup")
    if _sha(root / "inputs/aedt_spec.json") != value["spec_sha256"]:
        raise RuntimeError("batch spec identity differs")
    # The manifest selects execution, but the preserved scientific request and
    # released native preparation own the design/setup/project identity.
    spec = _read(root / "inputs/aedt_spec.json")
    prepared = _read(root / "native-preparation-receipt.json")
    expected = {
        "design": spec["project"]["design"],
        "setup": spec["run_control"]["setup_name"],
        "project": f"{spec['project']['name']}.aedt",
        "requested_aedt_version": spec["aedt"]["requested_version"],
    }
    if any(value[key] != item for key, item in expected.items()):
        raise RuntimeError("batch execution identity differs from packaged request")
    if (prepared.get("status") != "native_preparation_only"
            or prepared.get("release") != {"ok": True}
            or prepared.get("save", {}).get("ok") is not True
            or prepared.get("solver_invoked") is not False
            or prepared.get("source", {}).get("spec_sha256") != value["spec_sha256"]
            or prepared.get("project") != expected["project"]
            or prepared.get("setup", {}).get("name") != expected["setup"]
            or prepared.get("requested", {}).get("aedt_version") != expected["requested_aedt_version"]
            or prepared.get("connected") != value["prepared_connected"]
            or _sha(_inside(root / "native", expected["project"])) != prepared["save"]["project_sha256"]):
        raise RuntimeError("batch execution identity differs from released native preparation")
    return value


def run_native_batch(package_dir: str | Path, *, output_dir: str | Path) -> Path:
    """Run exactly one solver-only process in a fresh attempt, without PyAEDT.

    No shell, scheduler, retry or arbitrary argument passthrough is used. The
    RAM setting is the native per-host solver budget, not an OS process cap.
    """
    package_root = Path(package_dir).expanduser().resolve()
    package = _package(package_root)
    output = _fresh(output_dir, package_root)
    attempt = {
        "schema_version": _ATTEMPT, "attempt_id": str(uuid.uuid4()),
        "package_sha256": _sha(package_root / _MANIFEST),
        "package_id": package["package_id"], "status": "not_started",
        "requested_aedt_version": package["requested_aedt_version"],
        "actual_aedt_version": None,
        "version_observation": "not_queried; raw solver stdout/stderr retained",
        "execution_location": {"hostname": socket.gethostname(), "directory": str(output)},
        "resources_requested": package["resources"],
        "resource_semantics": "native MachineList per-host solver limits; not an OS hard cap",
        "started_at_unix": time.time(),
    }
    _write(output / "batch-attempt.json", attempt)
    process = None
    try:
        work = output / "work"
        native = package_root / "native"
        _copy_members(native, work, _inventory(native))
        resource = package["resources"]
        machine = f"list={socket.gethostname()}:-1:{resource['cores']}:{resource['ram_limit_percent']}%"
        executable = shutil.which(package["recipe"]["executable"])
        if executable is None:
            raise FileNotFoundError("bound AEDT batch executable is unavailable")
        argv = [executable, "-ng", "-UseElectronicsPPE", "-distributed", "-auto",
                "-machinelist", machine, "-BatchSolve", package["recipe"]["target"],
                str(_inside(work, package["project"]))]
        attempt.update(argv=argv, executable_sha256=_sha(Path(executable)), status="starting")
        _write(output / "batch-attempt.json", attempt)
        with (output / "solver.stdout").open("xb") as stdout, (output / "solver.stderr").open("xb") as stderr:
            process = subprocess.Popen(argv, cwd=work, stdout=stdout, stderr=stderr)
            attempt.update(status="running", child_pid=process.pid)
            _write(output / "batch-attempt.json", attempt)
            returncode = process.wait()
        attempt.update(status="external_process_terminal", returncode=returncode,
                       finished_at_unix=time.time(), work_members=_inventory(work),
                       raw_evidence=[{"path": name, "bytes": (output / name).stat().st_size,
                                      "sha256": _sha(output / name)}
                                     for name in ("solver.stdout", "solver.stderr")])
        _write(output / "batch-attempt.json", attempt)
        return output / "batch-attempt.json"
    except BaseException as exc:
        # Interruption is not a terminal solver observation and never kills or
        # relaunches the child. Preserve ownership information for the operator.
        attempt.update(status="unsettled" if process is not None and process.returncode is None else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        _write(output / "batch-attempt.json", attempt)
        raise


def _attempt(package_root: Path, attempt_root: Path, package: dict) -> dict:
    value = _read(attempt_root / "batch-attempt.json")
    if (value.get("schema_version") != _ATTEMPT
            or value.get("status") != "external_process_terminal"
            or value.get("returncode") != 0
            or value.get("package_sha256") != _sha(package_root / _MANIFEST)
            or value.get("package_id") != package["package_id"]):
        raise RuntimeError("batch attempt is not a matching settled zero-exit process")
    _verify(attempt_root / "work", value["work_members"])
    if _inventory(attempt_root / "work") != value["work_members"]:
        raise RuntimeError("returned batch inventory changed after process settlement")
    _verify(attempt_root, value["raw_evidence"])
    project = _inside(attempt_root / "work", package["project"])
    results = _inside(attempt_root / "work", package["results"])
    if not project.is_file() or not results.is_dir():
        raise RuntimeError("batch return lacks the saved project or complete results directory")
    return value


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
    _attempt(package_root, attempt_root, package)
    output = _fresh(output_dir, package_root, attempt_root)
    try:
        _copy_members(package_root / "inputs", output, _inventory(package_root / "inputs"))
        parent = output / "batch-source"
        _copy_members(package_root, parent / "package", [*package["members"],
                      {"path": _MANIFEST, "bytes": (package_root / _MANIFEST).stat().st_size,
                       "sha256": _sha(package_root / _MANIFEST)}])
        _copy_members(attempt_root / "work", parent / "attempt/work", _inventory(attempt_root / "work"))
        attempt = _read(attempt_root / "batch-attempt.json")
        evidence = [*attempt["raw_evidence"], {"path": "batch-attempt.json",
                    "bytes": (attempt_root / "batch-attempt.json").stat().st_size,
                    "sha256": _sha(attempt_root / "batch-attempt.json")}]
        _copy_members(attempt_root, parent / "attempt", evidence)
        _copy_members(attempt_root / "work", output, attempt["work_members"])
        context = {"package_sha256": _sha(package_root / _MANIFEST),
                   "attempt_sha256": _sha(attempt_root / "batch-attempt.json")}
        _write(output / "metadata/batch-collection.json", context)
        rc = _execute(output / "metadata/aedt_handoff_metadata.json", batch_collection=context)
        if rc:
            raise RuntimeError("native batch collection failed; inspect its retained run receipt")
        return resolve_results(output)
    except BaseException as exc:
        _write(output / "metadata/batch-collection-failure.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _collection_source(run_dir: Path, context: dict) -> tuple[dict, dict, dict]:
    """Verify immutable parent copies before native open and offline Resolve."""
    package_root = run_dir / "batch-source/package"
    attempt_root = run_dir / "batch-source/attempt"
    if (context != _read(run_dir / "metadata/batch-collection.json")
            or context.get("package_sha256") != _sha(package_root / _MANIFEST)
            or context.get("attempt_sha256") != _sha(attempt_root / "batch-attempt.json")):
        raise RuntimeError("native batch collection parent identity differs")
    package = _package(package_root)
    attempt = _attempt(package_root, attempt_root, package)
    prepared = _read(package_root / "native-preparation-receipt.json")
    if (prepared.get("status") != "native_preparation_only"
            or prepared.get("release") != {"ok": True}
            or prepared.get("save", {}).get("ok") is not True
            or prepared.get("solver_invoked") is not False
            or prepared.get("source", {}).get("spec_sha256") != package["spec_sha256"]
            or prepared.get("source", {}).get("planar_model_sha256") != package["model_sha256"]
            or prepared.get("source", {}).get("planar_source_sha256") != package["source_sha256"]
            or _sha(package_root / "native" / package["project"]) != prepared["save"]["project_sha256"]):
        raise RuntimeError("batch parent is not the released original prepared native model")
    for item in package["members"]:
        if item["path"].startswith("inputs/") and item["path"] != "inputs/metadata/aedt_run_receipt.json":
            local = _inside(run_dir, item["path"].removeprefix("inputs/"))
            if not local.is_file() or _sha(local) != item["sha256"]:
                raise RuntimeError("collection input differs from original prepared source")
    return package, attempt, prepared


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
