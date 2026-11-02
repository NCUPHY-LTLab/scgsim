"""Source-bound packaging and owned collection for AEDT native batch runs."""

from __future__ import annotations

import shutil
import tarfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from scgsim.aedt.epr.evidence import PreparedEprEvidence
from scgsim.aedt.preparation.cohort import canonical_handoff_paths
from scgsim.aedt.results.prepared import resolve_prepared_model
from scgsim.aedt.results.provenance import (
    encode_initial_receipt,
    initial_receipt_payload,
)
from scgsim.aedt.results.resolve import resolve_results
from scgsim.aedt.runtime.transaction import ExecutionOptions, execute_handoff
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.hfss import HfssEprSpec
from scgsim.aedt.specs.parse import parse_aedt_spec

from .runtime import batch_protocol as protocol

if TYPE_CHECKING:
    from scgsim.aedt.results.resolve import ResolvedRun

_PACKAGE = protocol.PACKAGE_SCHEMA
_MANIFEST = protocol.MANIFEST


@dataclass(frozen=True)
class NativeBatchPackage:
    """Paths to a source-bound prepared-native package and portable archive."""

    root: Path
    manifest_path: Path
    archive_path: Path


@dataclass(frozen=True)
class PreparationContext:
    """Validated source facts required by the packaging stages."""

    root: Path
    receipt: dict[str, Any]
    spec: HfssEprSpec
    project_path: Path
    resource_payload: dict[str, object]
    solver_executable: str
    dependencies: Sequence[str]


def _prepare_context(
    prepared_run: str | Path,
    solver_executable: str,
    resources: AedtResources,
    dependencies: Sequence[str],
) -> PreparationContext:
    """Validate the released source, request, recipe, and authored evidence."""
    root = Path(prepared_run).expanduser().resolve()
    prepared = resolve_prepared_model(root)
    receipt = cast(
        dict[str, Any],
        protocol.object_payload(protocol.read_json(prepared.receipt_path)),
    )
    spec = parse_aedt_spec(
        protocol.object_payload(protocol.read_json(root / "aedt_spec.json")),
        base_dir=root,
    )
    if (
        not isinstance(spec, HfssEprSpec)
        or spec.epr_request is None
        or spec.run_control is None
    ):
        raise TypeError("native batch requires a complete body-first EPR preparation")
    if not isinstance(resources, AedtResources):
        raise TypeError("resources must be AedtResources")
    resource_payload = protocol.Resources.from_payload(
        resources.to_payload()
    ).to_payload()
    if (
        not isinstance(solver_executable, str)
        or not solver_executable
        or "\0" in solver_executable
    ):
        raise ValueError(
            "solver_executable must be an explicit executable path or command"
        )
    if isinstance(dependencies, (str, bytes)):
        raise TypeError("dependencies must be a sequence of run-relative paths")

    PreparedEprEvidence.from_payload(
        receipt,
        evidence_root=root / "metadata/epr_expressions",
    )
    return PreparationContext(
        root=root,
        receipt=receipt,
        spec=spec,
        project_path=prepared.project_path,
        resource_payload=resource_payload,
        solver_executable=solver_executable,
        dependencies=dependencies,
    )


def _write_handoff_inputs(context: PreparationContext, output: Path) -> None:
    """Copy the handoff cohort with its canonical not-run receipt bytes."""
    inputs = output / "inputs"
    inputs.mkdir()
    receipt = context.receipt
    initial = initial_receipt_payload(
        schema_version=receipt["schema_version"],
        mode=receipt["mode"],
        requested=receipt["requested"],
        pdk_materials=receipt["pdk_materials"],
        vacuum_material_id=receipt["vacuum_material_id"],
        source=receipt["source"],
        prepared_runtime_source_value=receipt["prepared_runtime_source"],
        outputs={},
        prepared_at_utc=receipt["prepared_at_utc"],
    )
    for relative in canonical_handoff_paths(context.spec):
        target = protocol.inside(inputs, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "metadata/aedt_run_receipt.json":
            target.write_bytes(encode_initial_receipt(initial))
        else:
            shutil.copy2(protocol.inside(context.root, relative), target)
    original_manifest = protocol.object_payload(
        protocol.read_json(inputs / "metadata/aedt_handoff_manifest.json")
    )
    protocol.verify_members(inputs, protocol.members(original_manifest["members"]))
    protocol.write_json(output / "native-preparation-receipt.json", receipt)
    _copy_calculator_evidence(context.root, output)


def _copy_calculator_evidence(root: Path, output: Path) -> None:
    """Retain all authored expression evidence without regeneration."""
    evidence = root / "metadata/epr_expressions"
    if not evidence.is_dir():
        raise RuntimeError("native preparation calculator evidence is missing")
    protocol.copy_members(
        evidence, output / "epr-expressions", protocol.inventory(evidence)
    )


def _copy_native_files(context: PreparationContext, output: Path) -> str:
    """Copy the native project, full saved-results tree, and declared dependencies."""
    native = output / "native"
    native.mkdir()
    project_relative = context.project_path.relative_to(context.root).as_posix()
    results_relative = f"{context.spec.project_name}.aedtresults"
    required = [project_relative, *context.dependencies]
    if (context.root / results_relative).exists():
        required.append(results_relative)
    copied: set[str] = set()
    for relative in required:
        source = protocol.inside(context.root, relative)
        if source.is_dir():
            _copy_native_directory(context.root, native, relative, copied)
        elif source.is_file():
            _copy_native_file(source, native, relative, copied)
        else:
            raise FileNotFoundError(
                f"declared native dependency unavailable: {relative}"
            )
    return project_relative


def _copy_native_directory(
    root: Path, native: Path, relative: str, copied: set[str]
) -> None:
    for item in protocol.inventory(protocol.inside(root, relative)):
        child = (Path(relative) / item.path).as_posix()
        if child in copied:
            continue
        target = protocol.inside(native, child)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(protocol.inside(root, child), target)
        copied.add(child)


def _copy_native_file(
    source: Path, native: Path, relative: str, copied: set[str]
) -> None:
    if relative in copied:
        return
    target = protocol.inside(native, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.add(relative)


def _package_manifest(
    context: PreparationContext, output: Path, project_relative: str
) -> dict[str, object]:
    """Build and validate the source-bound portable package manifest."""
    spec = context.spec
    run_control = spec.run_control
    if run_control is None:
        raise TypeError("native batch requires a complete body-first EPR preparation")
    manifest = {
        "schema_version": _PACKAGE,
        "status": "prepared_native",
        "package_id": str(uuid.uuid4()),
        "created_at_unix": time.time(),
        "source_location": str(context.root),
        "requested_aedt_version": spec.aedt_version,
        "prepared_connected": context.receipt["connected"],
        "spec_sha256": protocol.sha256(output / "inputs/aedt_spec.json"),
        "model_sha256": spec.geometry.model_sha256,
        "source_sha256": spec.geometry.source_sha256,
        "project": project_relative,
        "results": f"{spec.project_name}.aedtresults",
        "design": spec.design_name,
        "setup": run_control.setup_name,
        "resources": context.resource_payload,
        "recipe": {
            "executable": context.solver_executable,
            "mode": "prepared_project_batchsolve_ppe",
            "allocation": "one_host_native_auto",
            "target": f"{spec.design_name}:Nominal:{run_control.setup_name}",
        },
        "declared_dependencies": list(context.dependencies),
        "members": [item.to_payload() for item in protocol.inventory(output)],
    }
    typed_manifest = protocol.PackageManifest.from_payload(manifest)
    protocol.write_json(output / _MANIFEST, typed_manifest.to_payload())
    return manifest


def _stage_portable_runner(output: Path) -> None:
    """Copy the standalone stdlib runner and protocol before inventorying."""
    runtime_dir = Path(protocol.__file__).parent
    shutil.copy2(runtime_dir / "batch_runner.py", output / "run_batch.py")
    shutil.copy2(runtime_dir / "batch_protocol.py", output / "batch_protocol.py")


def _archive_package(output: Path, manifest: dict[str, object]) -> Path:
    """Archive the exact member inventory and its manifest."""
    archive = output / "native_batch.tar.gz"
    members = cast(list[dict[str, object]], manifest["members"])
    with tarfile.open(archive, "w:gz") as stream:
        for item in [*members, {"path": _MANIFEST}]:
            relative = cast(str, item["path"])
            stream.add(
                protocol.inside(output, relative),
                arcname=relative,
                recursive=False,
            )
    return archive


def prepare_native_batch(
    prepared_run: str | Path,
    *,
    output_dir: str | Path,
    solver_executable: str,
    resources: AedtResources,
    dependencies: Sequence[str] = (),
) -> NativeBatchPackage:
    """Package a released prepare-only EPR model without launching a solver."""
    context = _prepare_context(prepared_run, solver_executable, resources, dependencies)
    output = protocol.fresh(output_dir, context.root)
    try:
        _write_handoff_inputs(context, output)
        project_relative = _copy_native_files(context, output)
        _stage_portable_runner(output)
        manifest = _package_manifest(context, output, project_relative)
        archive = _archive_package(output, manifest)
        return NativeBatchPackage(output, output / _MANIFEST, archive)
    except BaseException as exc:
        protocol.write_json(
            output / "package-failure.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def _stage_collection_sources(
    package_root: Path,
    attempt_root: Path,
    output: Path,
    package: protocol.PackageManifest,
    attempt: protocol.BatchAttempt,
) -> protocol.CollectionContext:
    """Retain package and process evidence under the owned collection copy."""
    protocol.copy_members(
        package_root / "inputs",
        output,
        protocol.inventory(package_root / "inputs"),
    )
    parent = output / "batch-source"
    manifest_path = package_root / _MANIFEST
    package_records = [
        *package.members,
        protocol.Member(
            _MANIFEST, manifest_path.stat().st_size, protocol.sha256(manifest_path)
        ),
    ]
    protocol.copy_members(package_root, parent / "package", package_records)
    protocol.copy_members(
        attempt_root / "work",
        parent / "attempt/work",
        protocol.inventory(attempt_root / "work"),
    )
    if attempt.raw_evidence is None or attempt.work_members is None:
        raise RuntimeError("batch terminal attempt lacks returned inventories")
    attempt_path = attempt_root / "batch-attempt.json"
    attempt_records = [
        *attempt.raw_evidence,
        protocol.Member(
            "batch-attempt.json",
            attempt_path.stat().st_size,
            protocol.sha256(attempt_path),
        ),
    ]
    protocol.copy_members(attempt_root, parent / "attempt", attempt_records)
    protocol.copy_members(attempt_root / "work", output, attempt.work_members)
    context = protocol.CollectionContext(
        package_sha256=protocol.sha256(manifest_path),
        attempt_sha256=protocol.sha256(attempt_path),
    )
    protocol.write_json(output / "metadata/batch-collection.json", context.to_payload())
    return context


def _run_owned_collection(
    output: Path, context: protocol.CollectionContext
) -> ResolvedRun:
    """Execute the existing handoff transaction on its owned copied inputs."""
    rc = execute_handoff(
        output / "metadata/aedt_handoff_metadata.json",
        options=ExecutionOptions(batch_collection=context),
    )
    if rc:
        raise RuntimeError(
            "native batch collection failed; inspect its retained run receipt"
        )
    return resolve_results(output)


def collect_native_batch(
    package_dir: str | Path,
    attempt_dir: str | Path,
    *,
    output_dir: str | Path,
) -> ResolvedRun:
    """Collect full adaptive EPR on an owned copy; never Analyze or rebuild."""
    package_root = Path(package_dir).expanduser().resolve()
    attempt_root = Path(attempt_dir).expanduser().resolve()
    package = protocol.read_package(package_root)
    attempt = protocol.read_attempt(package_root, attempt_root, package)
    output = protocol.fresh(output_dir, package_root, attempt_root)
    try:
        context = _stage_collection_sources(
            package_root, attempt_root, output, package, attempt
        )
        return _run_owned_collection(output, context)
    except BaseException as exc:
        protocol.write_json(
            output / "metadata/batch-collection-failure.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
