"""Write portable AEDT handoff directories.

Owns: the on-disk handoff layout, the GDS preflight and import-layer rewrite, the
launcher scripts, and the receipt / manifest / archive that seal a handoff.
Does not own: which paths a spec family contains (`cohort.py`), the spec payloads
(`scgsim.aedt.specs`) or hash verification at run time (`scgsim.aedt.runtime`).
Preparing a handoff never opens AEDT.
"""

from __future__ import annotations

import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

from scgsim.aedt._data import (
    detached,
)
from scgsim.aedt._io import file_sha256, write_json
from scgsim.aedt.preparation.cohort import (
    GEOMETRY_SOURCE_PATHS,
    canonical_handoff_paths,
)
from scgsim.aedt.preparation.planar import validate_geometry_workers
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    encode_initial_receipt,
    initial_receipt_payload,
    prepared_runtime_source,
)
from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    REQUIRED_AEDT_VERSION,
    AedtResources,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    SavedSolution,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.aedt.specs.hfss import EigenmodeRunControl
from scgsim.aedt.specs.hfss_prepared import HfssEprAnalysisSpec
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec

_NEW_DIRECTORY_MESSAGE = (
    "output_dir must be new; prepared handoffs never reuse directories"
)


@dataclass(frozen=True)
class HandoffPlan:
    """The fixed, portable file set. Preparing it never opens AEDT."""

    run_dir: Path
    script_path: Path
    spec_path: Path
    metadata_path: Path
    receipt_path: Path
    manifest_path: Path
    archive_path: Path

    @classmethod
    def under(cls, run_dir: Path) -> HandoffPlan:
        """Name the fixed file set below `run_dir` without creating anything."""
        metadata_dir = run_dir / "metadata"
        return cls(
            run_dir,
            run_dir / "run_aedt.sh",
            run_dir / "aedt_spec.json",
            metadata_dir / "aedt_handoff_metadata.json",
            metadata_dir / "aedt_run_receipt.json",
            metadata_dir / "aedt_handoff_manifest.json",
            run_dir / "aedt_handoff.tar.gz",
        )


@dataclass(frozen=True)
class ReceiptSeed:
    """What the initial receipt records about a prepared handoff.

    `requested` is the AEDT / PyAEDT version request; `source` names the sealed
    inputs (spec hash, GDS or planar geometry hashes).
    """

    mode: str
    requested: dict
    pdk_materials: dict
    vacuum_material_id: str
    source: dict
    prepared_runtime_source_value: dict
    prepared_at_utc: str


def prepare_handoff(
    *, spec: AedtSpec, output_dir: str | Path, resources: AedtResources | None = None
) -> HandoffPlan:
    """Bind every required input into a new portable run directory."""
    if resources is not None and not isinstance(resources, AedtResources):
        raise TypeError("resources must be AedtResources or None")
    if not isinstance(spec, Q2dSpec) and getattr(spec, "modeling", None) is None:
        raise ValueError("new AEDT 3D execution requires explicit modeling")
    source_gds, preflight = _preflight_source_gds(spec)

    plan = HandoffPlan.under(new_run_directory(output_dir))
    run_dir = plan.run_dir
    plan.metadata_path.parent.mkdir(parents=True)
    copied_gds: Path | None = None
    if source_gds is not None:
        geometry_dir = run_dir / "geometry"
        geometry_dir.mkdir()
        copied_gds = geometry_dir / "design.gds"
        original_gds = geometry_dir / "source.gds"
        shutil.copy2(source_gds, original_gds)
        _write_import_gds(source_gds, copied_gds, spec.effective_layer_imports)

    prepared_at = utc_now()
    started = time.perf_counter()
    prepared_source = prepared_runtime_source()

    write_script(plan.script_path)
    payload = spec.to_payload()
    if not isinstance(spec, (Q2dSpec, Q3dSpec)):
        _bind_gds_payload(payload, spec, original_gds)
    if isinstance(spec, Q3dSpec) and spec.geometry_source is not None:
        payload["geometry_source"] = _attach_q3d_geometry_source(spec, run_dir)
    write_json(plan.spec_path, payload)
    files = {
        "spec": plan.spec_path.name,
        "receipt": "metadata/aedt_run_receipt.json",
    }
    metadata = {
        "schema_version": "scgsim.aedt.handoff.v1",
        "expected_receipt_schema": RECEIPT_V3,
        "status": "prepared",
        "mode": spec.mode,
        "project": payload["project"],
        "materials": payload["materials"],
        "vacuum_material_id": payload["vacuum_material_id"],
        "run_control": payload["run_control"],
        "execution": {"resources": resources.to_payload() if resources else None},
        "pyaedt": payload["pyaedt"],
        "aedt": payload["aedt"],
        "files": files,
        "prepared_at_utc": prepared_at,
        "preparation_seconds": round(time.perf_counter() - started, 6),
    }
    source = {
        "spec": plan.spec_path.name,
        "spec_sha256": file_sha256(plan.spec_path),
    }
    if copied_gds is not None and preflight is not None:
        _record_gds(files, metadata, source, copied_gds, preflight)
    seal_handoff(
        plan,
        metadata=metadata,
        receipt=ReceiptSeed(
            mode=spec.mode,
            requested={
                "aedt_version": spec.aedt_version,
                "pyaedt_version": spec.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            pdk_materials=payload["materials"],
            vacuum_material_id=payload["vacuum_material_id"],
            source=source,
            prepared_runtime_source_value=prepared_source,
            prepared_at_utc=prepared_at,
        ),
        manifest_schema="scgsim.aedt.handoff-manifest.v1",
        allowed=tuple(run_dir / relative for relative in canonical_handoff_paths(spec)),
    )
    return plan


def _record_gds(files, metadata, source, copied_gds: Path, preflight) -> None:
    """Add the import GDS and its preflight to the file list, metadata and source."""
    files["gds"] = "geometry/design.gds"
    metadata["gds_sha256"] = file_sha256(copied_gds)
    metadata["gds_preflight"] = {
        "polygon_layer_datatypes": [
            list(pair) for pair in sorted(preflight["polygon_layer_datatypes"])
        ],
        "path_count": preflight["path_count"],
        "label_count": preflight["label_count"],
        "labels": "ignored as non-geometry",
    }
    source["gds"] = "geometry/design.gds"
    source["gds_sha256"] = file_sha256(copied_gds)


def seal_handoff(
    plan: HandoffPlan,
    *,
    metadata: dict,
    receipt: ReceiptSeed,
    manifest_schema: str,
    allowed: tuple[Path, ...],
) -> None:
    """Write metadata, initial receipt, manifest and archive, in that order.

    The manifest hashes every allowed path except itself, so it is written after
    the receipt it covers. `allowed` must include the manifest path.
    """
    write_json(plan.metadata_path, metadata)
    plan.receipt_path.write_bytes(
        encode_initial_receipt(
            initial_receipt_payload(
                schema_version=RECEIPT_V3,
                mode=receipt.mode,
                requested=receipt.requested,
                pdk_materials=receipt.pdk_materials,
                vacuum_material_id=receipt.vacuum_material_id,
                source=receipt.source,
                prepared_runtime_source_value=receipt.prepared_runtime_source_value,
                outputs={},
                prepared_at_utc=receipt.prepared_at_utc,
            )
        )
    )
    run_dir = plan.run_dir
    write_json(
        plan.manifest_path,
        {
            "schema_version": manifest_schema,
            "expected_receipt_schema": RECEIPT_V3,
            "allowed_paths": [path.relative_to(run_dir).as_posix() for path in allowed],
            "members": [
                _member(path, run_dir) for path in allowed if path != plan.manifest_path
            ],
        },
    )
    with tarfile.open(plan.archive_path, "w:gz") as archive:
        for path in allowed:
            archive.add(
                path, arcname=path.relative_to(run_dir).as_posix(), recursive=False
            )


def _preflight_source_gds(spec):
    """Check the authored GDS against the declared layer pairs, before any write."""
    if isinstance(spec, (Q2dSpec, Q3dSpec)):
        return None, None
    source_gds = spec.gds_path.resolve()
    if not source_gds.is_file():
        raise FileNotFoundError(f"gds_path must be an existing file: {source_gds}")
    preflight = _gds_preflight(source_gds)
    actual_pairs = preflight["polygon_layer_datatypes"]
    declared_pairs = {(item.layer, item.datatype) for item in spec.layer_imports}
    if actual_pairs != declared_pairs:
        raise ValueError(
            f"GDS layer/datatype pairs {sorted(actual_pairs)!r} do not match spec {sorted(declared_pairs)!r}"
        )
    return source_gds, preflight


def new_run_directory(output_dir: str | Path) -> Path:
    """Resolve `output_dir` and refuse to reuse an existing directory."""
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(_NEW_DIRECTORY_MESSAGE)
    return run_dir


def _bind_gds_payload(payload, spec, original_gds: Path) -> None:
    """Point the spec's GDS record at the copied pair and record the source hash."""
    payload["gds"]["path"] = "geometry/design.gds"
    payload["gds"]["source_path"] = "geometry/source.gds"
    payload["gds"]["source_sha256"] = file_sha256(original_gds)
    payload["gds"]["import_layer_map"] = [
        {
            key: item[key]
            for key in (
                "layer",
                "datatype",
                "layer_name",
                "physical_layer_id",
                "native_import_layer",
            )
        }
        for item in spec.effective_layer_imports
    ]


def _attach_q3d_geometry_source(spec, run_dir: Path):
    """Copy the Q3D source attachments into the handoff and return the rebased record."""
    geometry_source = detached(spec.geometry_source)
    for key, destination in GEOMETRY_SOURCE_PATHS.items():
        reference = geometry_source["files"][key]
        source_path = Path(reference["path"])
        if file_sha256(source_path) != reference["sha256"]:
            raise ValueError(f"Q3D source attachment identity differs: {key}")
        target = run_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        reference["path"] = destination
    return geometry_source


def prepare_epr_analysis_handoff(  # noqa: PLR0913 - one keyword per EPR analysis input
    *,
    saved_solution: SavedSolution,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    epr_request: EprAnalysisRequest,
    output_dir: str | Path,
    geometry_workers: int | None,
) -> HandoffPlan:
    """Prepare one private workcopy-only saved-field analysis transaction."""
    if not isinstance(saved_solution, SavedSolution):
        raise TypeError("saved_solution must be SavedSolution")
    validate_geometry_workers(geometry_workers)
    run_dir = new_run_directory(output_dir)
    saved_root = run_dir / "saved"
    (run_dir / "metadata").mkdir(parents=True)
    copied_paths = _copy_saved_solution_members(saved_solution, saved_root)
    copied_solution = _solution_at(saved_solution, saved_root)
    spec = HfssEprAnalysisSpec(
        project_name=project_name,
        design_name=design_name,
        geometry=geometry,
        run_control=run_control,
        epr_request=epr_request,
        saved_solution=copied_solution,
    )
    plan = HandoffPlan.under(run_dir)
    prepared_at = utc_now()
    started = time.perf_counter()
    prepared_source = prepared_runtime_source()
    payload = spec.to_payload()
    materials = detached(geometry.source)["materials"]
    vacuum_id = single_vacuum_material_id(materials)
    _write_analysis_script(plan.script_path)
    write_json(plan.spec_path, payload)
    files = _analysis_files(plan, copied_solution, saved_root)
    metadata = {
        "schema_version": "scgsim.aedt.handoff.v2",
        "expected_receipt_schema": RECEIPT_V3,
        "status": "prepared",
        "mode": spec.mode,
        "workflow": "epr_analysis",
        "project": payload["project"],
        "materials": materials,
        "vacuum_material_id": vacuum_id,
        "run_control": payload["run_control"],
        "execution": {"geometry_workers": geometry_workers},
        "pyaedt": payload["pyaedt"],
        "aedt": payload["aedt"],
        "files": files,
        "prepared_at_utc": prepared_at,
        "preparation_seconds": round(time.perf_counter() - started, 6),
    }
    source = {
        "spec": plan.spec_path.name,
        "spec_sha256": file_sha256(plan.spec_path),
        "planar_source_sha256": geometry.source_sha256,
        "planar_model_sha256": geometry.model_sha256,
        "saved_solution_sha256": copied_solution.content_sha256,
    }
    seal_handoff(
        plan,
        metadata=metadata,
        receipt=ReceiptSeed(
            mode=spec.mode,
            requested={
                "aedt_version": REQUIRED_AEDT_VERSION,
                "pyaedt_version": LOCKED_PYAEDT,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                "workflow": "epr_analysis",
            },
            pdk_materials=materials,
            vacuum_material_id=vacuum_id,
            source=source,
            prepared_runtime_source_value=prepared_source,
            prepared_at_utc=prepared_at,
        ),
        manifest_schema="scgsim.aedt.handoff-manifest.v2",
        allowed=(
            plan.script_path,
            plan.spec_path,
            *sorted(copied_paths),
            plan.metadata_path,
            plan.receipt_path,
            plan.manifest_path,
        ),
    )
    return plan


def _analysis_files(plan, copied_solution, saved_root) -> dict[str, str]:
    """Name the spec, receipt and saved project / results files of the metadata."""
    return {
        "spec": plan.spec_path.name,
        "receipt": "metadata/aedt_run_receipt.json",
        "saved_project": (
            "saved/" + copied_solution.project_path.relative_to(saved_root).as_posix()
        ),
        "saved_results": (
            "saved/" + copied_solution.result_path.relative_to(saved_root).as_posix()
        ),
    }


def _solution_at(saved_solution: SavedSolution, saved_root: Path) -> SavedSolution:
    """Re-root a saved solution record at its copy inside the handoff."""
    return SavedSolution(
        root=saved_root,
        project_path=saved_root
        / saved_solution.project_path.relative_to(saved_solution.root),
        result_path=saved_root
        / saved_solution.result_path.relative_to(saved_solution.root),
        members=saved_solution.members,
        content_sha256=saved_solution.content_sha256,
        identity=saved_solution.identity,
    )


def _copy_saved_solution_members(
    saved_solution: SavedSolution, saved_root: Path
) -> list[Path]:
    """Copy the sealed saved-solution files, re-checking each against its record."""
    copied_paths: list[Path] = []
    for member in saved_solution.members:
        relative = Path(member["path"])
        source = (saved_solution.root / relative).resolve()
        if not source.is_relative_to(saved_solution.root) or source.is_symlink():
            raise RuntimeError("saved solution member escapes its sealed root")
        if (
            not source.is_file()
            or source.stat().st_size != member["bytes"]
            or file_sha256(source) != member["sha256"]
        ):
            raise RuntimeError(f"saved solution member changed: {relative.as_posix()}")
        target = saved_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_paths.append(target)
    return copied_paths


def single_vacuum_material_id(materials) -> str:
    """Return the only vacuum material id of a prepared geometry's materials."""
    vacuum_ids = sorted(
        material_id
        for material_id, record in materials.items()
        if isinstance(record, dict) and record.get("kind") == "vacuum"
    )
    if len(vacuum_ids) != 1:
        raise ValueError("prepared EPR geometry requires exactly one vacuum material")
    return vacuum_ids[0]


def _gds_preflight(path: Path) -> dict[str, int | set[tuple[int, int]]]:
    """Reject PATH elements and empty layouts; report layer pairs and label count.

    V1 accepts only polygonized CPW geometry, so every PATH is an error, whether
    found in the raw records or in the library gdstk reads.
    """
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "GDS preflight requires the scgsim[aedt] gdstk dependency"
        ) from exc
    path_pairs = _raw_gds_path_datatypes(path)
    if path_pairs:
        raise ValueError(
            "GDS preflight found PATH layer/datatypes "
            f"{sorted(path_pairs)!r}; V1 requires polygonized CPW geometry"
        )
    library = gdstk.read_gds(path)
    path_pairs.update(
        (int(layer), int(datatype))
        for cell in library.cells
        for path_item in cell.paths
        for layer, datatype in zip(path_item.layers, path_item.datatypes, strict=True)
    )
    if path_pairs:
        raise ValueError(
            "GDS preflight found PATH layer/datatypes "
            f"{sorted(path_pairs)!r}; V1 requires polygonized CPW geometry"
        )
    pairs = {
        (polygon.layer, polygon.datatype)
        for cell in library.cells
        for polygon in cell.polygons
    }
    if not pairs:
        raise ValueError("GDS preflight found no polygons")
    return {
        "polygon_layer_datatypes": {
            (int(layer), int(datatype)) for layer, datatype in pairs
        },
        "path_count": 0,
        "label_count": sum(len(cell.labels) for cell in library.cells),
    }


def _raw_gds_path_datatypes(path: Path) -> set[tuple[int, int]]:
    """Read PATH records because gdstk expands them to polygons on GDS import."""
    data = path.read_bytes()
    offset = 0
    in_path = False
    layer: int | None = None
    datatype: int | None = None
    result: set[tuple[int, int]] = set()
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("GDS preflight found a truncated record header")
        size = int.from_bytes(data[offset : offset + 2], "big")
        if size < 4 or offset + size > len(data):
            raise ValueError("GDS preflight found an invalid record length")
        record_type = data[offset + 2]
        payload = data[offset + 4 : offset + size]
        if record_type == 0x09:  # PATH
            in_path, layer, datatype = True, None, None
        elif in_path and record_type == 0x0D and len(payload) == 2:  # LAYER
            layer = int.from_bytes(payload, "big", signed=True)
        elif in_path and record_type == 0x0E and len(payload) == 2:  # DATATYPE
            datatype = int.from_bytes(payload, "big", signed=True)
        elif record_type == 0x11:  # ENDEL
            if in_path:
                if layer is None or datatype is None:
                    raise ValueError("GDS PATH record is missing layer or datatype")
                result.add((layer, datatype))
            in_path = False
        offset += size
    if in_path:
        raise ValueError("GDS preflight found an unterminated PATH record")
    return result


def write_script(path: Path) -> None:
    """Write the launcher that executes the handoff."""
    _write_launcher(path, '--execute "$@"')


def _write_analysis_script(path: Path) -> None:
    """Write the launcher that runs saved-field EPR analysis instead of a solve."""
    _write_launcher(path, '--execute --analyze-epr "$@"')


def _write_launcher(path: Path, arguments: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'run_dir="$(cd "$(dirname "$0")" && pwd)"\n'
        'python -m scgsim.aedt.run --handoff "$run_dir/metadata/aedt_handoff_metadata.json" '
        f"{arguments}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _member(path: Path, root: Path) -> dict[str, str | int]:
    """Describe one manifest member by relative path, size and SHA-256."""
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_import_gds(source: Path, destination: Path, effective_imports) -> None:
    """Lower explicit source pairs for numeric-layer-only native GDS import.

    Cells, references, database unit and polygon vertices stay source-authored;
    only the derived file's import selectors change. The source file is retained.
    """
    import gdstk

    library = gdstk.read_gds(source)
    selectors = {
        (item["layer"], item["datatype"]): item["native_import_layer"]
        for item in effective_imports
    }
    for cell in library.cells:
        for polygon in cell.polygons:
            polygon.layer = selectors[(polygon.layer, polygon.datatype)]
            polygon.datatype = 0
    library.write_gds(destination, max_points=0)
