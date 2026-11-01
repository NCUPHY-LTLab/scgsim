"""Build a Route-A or Route-B Palace mesh from an in-tree geometry plan.

Route A and Route B are the two geometry routes of SGB (the semantic geometry
builder). A Route A stack must carry ``route_a_thin_film`` provenance and a Route B
stack must not. Route A models metal as zero-thickness PEC (perfect electric
conductor) sheets placed at a face, which is why it needs the thin-film provenance.
Route B keeps the metal as a finite-thickness volume whose closed outer shell is PEC
and whose interior is not meshed (docs/geometry-sgb.qmd, "Current Surface-EPR
direction"; docs/concepts/model-and-numerics.qmd, "Metal representation in Palace").

This module owns one mesh build from start to finish: prepare the run directory,
write the GDS and stack inputs, run SGB to produce the XAO geometry and group
sidecar, mesh it in a child Gmsh process, and write the mesh manifest and
provenance JSON. It does not decide which records become solver groups
(``groups.py``) or call Gmsh itself (``lowering.py``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from .controls import mesh_controls
from .identity import (
    GSIM_HIGH_ORDER_DERIVATION_SHA,
    GSIM_SHA,
    SGB_DERIVATION_BASE_SHA,
    SGB_DERIVATION_IMPORTED_SHA,
    SGB_RUNTIME_AUTHORITY,
)
from .lowering import XaoMeshRequest
from .worker_process import mesh_from_route_xao


def _scgsim_version() -> str:
    """Return installed scgsim version from package metadata."""
    try:
        return package_version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError("Unable to determine installed scgsim version.") from exc


@dataclass(frozen=True)
class MeshBuildResult:
    """Result of SGB-backed mesh generation."""

    output_dir: Path
    mesh_path: Path
    groups: dict[str, dict[str, Any]]
    mesh_manifest_path: Path
    provenance_path: Path
    stack_path: Path
    xao_path: Path
    sidecar_path: Path


@dataclass(frozen=True)
class _RunPaths:
    """Where one mesh build reads and writes, all under ``run_dir``."""

    run_dir: Path
    semantic_dir: Path
    gds: Path
    stack: Path
    xao: Path
    sidecar: Path
    mesh: Path
    manifest: Path
    provenance: Path


def build_route_mesh(  # noqa: PLR0913 - one keyword per user-facing mesh knob
    *,
    component: Any,
    stack: Mapping[str, Any],
    route: str,
    output_dir: str | Path,
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    algorithm_3d: str = "Delaunay",
    threads: int = 1,
    surface_threads: int = 1,
    geometry_order: int = 1,
    high_order_optimize: bool = True,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] = (),
    indium_ground_bump_fill: Mapping[str, Any] | None = None,
    source_gds_bytes: bytes | None = None,
    source_geometry_input: Any | None = None,
    representation_context: Any | None = None,
) -> MeshBuildResult:
    """Lower Route-A/Route-B SGB geometry, mesh it, and emit artifacts."""
    _check_build_arguments(stack, route, source_geometry_input, representation_context)
    controls = mesh_controls(
        algorithm_3d=algorithm_3d,
        threads=threads,
        surface_threads=surface_threads,
        geometry_order=geometry_order,
        high_order_optimize=high_order_optimize,
    )
    thin_film = _route_a_thin_film_provenance(stack, route)
    _check_mesh_sizes(refined_mesh_size, max_mesh_size)
    stack = _with_port_sheet_metadata(stack, port_sheet_source_layers)
    paths = _prepare_run_directory(output_dir, route)

    top_cell = _write_geometry_inputs(component, stack, source_gds_bytes, paths)
    from scgsim.geometry.compiler.builder import SemanticGeometryBuilder

    # Source placement, curves, holes and endpoint projection are already bound.
    build_input = source_geometry_input
    source_curve_provenance = {
        "boundary_curves": [asdict(r) for r in build_input.boundary_curves],
        "boundary_reconstruction": [
            asdict(r) for r in build_input.boundary_reconstruction
        ],
    }
    if indium_ground_bump_fill is not None:
        receipt = indium_ground_bump_fill.get("receipt")
        if not isinstance(receipt, Mapping):
            raise TypeError("indium ground bump fill requires a structured receipt.")
        _write_indium_ground_bump_receipt(
            run_dir=paths.run_dir,
            gds_path=paths.gds,
            receipt=receipt,
        )

    # SGB/OCC construction remains in the calling process. Only Gmsh lowering
    # receives a fresh native-process boundary after an XAO has been written.
    SemanticGeometryBuilder().build(
        build_input,
        route=route,
        run_folder=paths.run_dir,
        representation_context=representation_context,
    )
    records = _read_physical_group_records(paths, route)

    groups, meshing = mesh_from_route_xao(
        XaoMeshRequest(
            xao_path=paths.xao,
            route=route,
            records=records,
            stack=stack,
            mesh_path=paths.mesh,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            mesh_controls=controls,
        )
    )

    manifest = _build_mesh_manifest(
        groups=groups,
        mesh_name=paths.mesh.name,
        route=route,
        route_a_thin_film=thin_film,
    )
    manifest["meshing"] = meshing
    manifest["curve_source"] = source_curve_provenance
    _add_native_curve_provenance(manifest, paths, build_input)
    manifest["source"]["gsim_high_order_meshing"] = GSIM_HIGH_ORDER_DERIVATION_SHA
    _write_json_creating_parent(paths.manifest, manifest)
    _write_route_provenance(paths, route, thin_film, top_cell)

    return MeshBuildResult(
        output_dir=paths.run_dir,
        mesh_path=paths.mesh,
        groups=groups,
        mesh_manifest_path=paths.manifest,
        provenance_path=paths.provenance,
        stack_path=paths.stack,
        xao_path=paths.xao,
        sidecar_path=paths.sidecar,
    )


def _check_build_arguments(
    stack: Any, route: str, source_geometry_input: Any, representation_context: Any
) -> None:
    """Reject a call that lacks preparation output or names an unknown route."""
    if representation_context is None or source_geometry_input is None:
        raise ValueError(
            "fresh Palace mesh requires neutral effective geometry input and representation_context from preparation"
        )
    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping after set_stack processing.")
    if route not in {"A", "B"}:
        raise ValueError("route must be either 'A' or 'B'.")


def _check_mesh_sizes(refined_mesh_size: float, max_mesh_size: float) -> None:
    """Require positive sizes with the maximum no smaller than the refined size."""
    if (
        refined_mesh_size <= 0
        or max_mesh_size <= 0
        or max_mesh_size < refined_mesh_size
    ):
        raise ValueError(
            "mesh sizes must be positive and max_mesh_size >= refined_mesh_size"
        )


def _prepare_run_directory(output_dir: str | Path, route: str) -> _RunPaths:
    """Create the empty run directory tree and name every file of this build."""
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"output_dir must be empty before mesh: {run_dir}")
    geometry_dir = run_dir / "geometry"
    semantic_dir = run_dir / "metadata" / "semantic_geometry"

    for path in (
        run_dir,
        geometry_dir,
        semantic_dir,
        run_dir / "logs",
        run_dir / "results" / "palace",
    ):
        path.mkdir(parents=True, exist_ok=True)

    return _RunPaths(
        run_dir=run_dir,
        semantic_dir=semantic_dir,
        gds=geometry_dir / "design.gds",
        stack=geometry_dir / "design.stack.json",
        xao=geometry_dir / f"semantic_geometry_route_{route.lower()}.xao",
        sidecar=semantic_dir / "04_export_physical_groups.json",
        mesh=run_dir / "palace.msh",
        manifest=run_dir / "metadata" / "mesh_manifest.json",
        provenance=run_dir / "metadata" / f"route_{route.lower()}_provenance.json",
    )


def _write_geometry_inputs(
    component: Any,
    stack: Mapping[str, Any],
    source_gds_bytes: bytes | None,
    paths: _RunPaths,
) -> str:
    """Write the design GDS and stack JSON; return the GDS top cell name.

    The GDS comes either from a live component or from immutable plan bytes whose
    hash must match the stack; the two sources are never mixed.
    """
    if source_gds_bytes is None:
        if component is None:
            raise ValueError("component or immutable plan GDS source is required")
        _write_component_gds(component=component, path=paths.gds)
        top_cell = _component_gds_top_cell_name(component=component, gds_path=paths.gds)
    else:
        if component is not None or not isinstance(source_gds_bytes, bytes):
            raise TypeError("plan GDS bytes cannot be mixed with a live component")
        wanted = stack.get("metadata", {}).get("gds_sha256")
        observed = hashlib.sha256(source_gds_bytes).hexdigest()
        if wanted != observed:
            raise ValueError("plan GDS bytes differ from paired stack source")
        top_cell = stack.get("metadata", {}).get("top_cell_name")
        if not isinstance(top_cell, str) or not top_cell:
            raise ValueError("plan stack requires top_cell_name")
        paths.gds.write_bytes(source_gds_bytes)
    paths.stack.write_text(json.dumps(dict(stack), indent=2) + "\n", encoding="utf-8")
    return top_cell


def _read_physical_group_records(paths: _RunPaths, route: str) -> list[Any]:
    """Load the SGB group sidecar after checking SGB wrote both of its outputs."""
    if not paths.xao.is_file():
        raise FileNotFoundError(f"SGB Route-{route} XAO not generated: {paths.xao}")
    if not paths.sidecar.is_file():
        raise FileNotFoundError(
            f"SGB Route-{route} sidecar not generated: {paths.sidecar}"
        )

    records = json.loads(paths.sidecar.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("04_export_physical_groups.json must contain a list")
    return records


def _add_native_curve_provenance(
    manifest: dict[str, Any], paths: _RunPaths, build_input: Any
) -> None:
    """Copy SGB's curved-arrangement record into the manifest, if there are curves."""
    construction_plan = json.loads(
        (paths.semantic_dir / "02_build_route_construction_plan.json").read_text(
            encoding="utf-8"
        )
    )
    native_curves = construction_plan.get("metadata", {}).get("curved_arrangement")
    if native_curves is not None:
        manifest["curved_arrangement"] = copy.deepcopy(native_curves)
    elif build_input.boundary_curves or build_input.boundary_reconstruction:
        raise ValueError(
            "curved SGB construction lacks native curved-arrangement provenance"
        )


def _write_route_provenance(
    paths: _RunPaths, route: str, thin_film: Mapping[str, Any] | None, top_cell: str
) -> None:
    """Write which versions and inputs produced this route's mesh."""
    run_dir = paths.run_dir
    _write_json_creating_parent(
        paths.provenance,
        {
            "scgsim_version": _scgsim_version(),
            "route": route,
            **({"route_a_thin_film": thin_film} if thin_film is not None else {}),
            "source_sha": {
                "gsim": GSIM_SHA,
                "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
                "sgb_derivation": {
                    "base": SGB_DERIVATION_BASE_SHA,
                    "imported_development": SGB_DERIVATION_IMPORTED_SHA,
                },
            },
            "component_top_cell": top_cell,
            "run_dir": ".",
            "stack_path": str(paths.stack.relative_to(run_dir)),
            "xao_path": str(paths.xao.relative_to(run_dir)),
            "sidecar_path": str(paths.sidecar.relative_to(run_dir)),
        },
    )


def _write_indium_ground_bump_receipt(
    *, run_dir: Path, gds_path: Path, receipt: Mapping[str, Any]
) -> None:
    """Persist public fill provenance before the generic handoff archive is made."""
    payload = copy.deepcopy(dict(receipt))
    payload["design_gds"] = {
        "path": gds_path.relative_to(run_dir).as_posix(),
        "sha256": hashlib.sha256(gds_path.read_bytes()).hexdigest(),
    }
    _write_json_creating_parent(
        run_dir / "metadata" / "indium_ground_bumps.json", payload
    )


def _write_json_creating_parent(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_mesh_manifest(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    mesh_name: str,
    route: str,
    route_a_thin_film: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact manifest from structured groups only."""
    entries: list[dict[str, Any]] = []

    for section in (
        "volumes",
        "boundary_surfaces",
        "surface_groups",
        "pec_surfaces",
        "port_surfaces",
    ):
        for name, info in groups.get(section, {}).items():
            if not isinstance(info, Mapping):
                continue
            entry = {
                "name": name,
                "section": section,
                "dimension": int(info.get("dim", 0) or 0),
                "tags": list(info.get("tags", ())),
                "attributes": [
                    int(item) for item in _physical_group_values(info.get("phys_group"))
                ],
                "structured": bool(info.get("sgb_record") == "final_physical_group"),
                "source": str(info.get("source", "")),
            }
            for key in (
                "route",
                "role",
                "solver_use",
                "representation",
                "surface_id",
                "interface_type",
                "face_kind",
                "owner_semantic_ids",
                "adjacent_solution_volume_ids",
                "conductor_component_id",
                "net_id",
                "equipotential_id",
                "source_provenance",
                "physical_attribute",
                "contact_kind",
                "source_id",
                "interface_of",
                "embedded_volume_id",
                "direction",
            ):
                if key in info:
                    entry[key] = info[key]
            entries.append(entry)

    return {
        "schema_version": 1,
        "route": route,
        **(
            {"route_a_thin_film": dict(route_a_thin_film)}
            if route_a_thin_film is not None
            else {}
        ),
        "mesh_file": mesh_name,
        "source": {
            "gsim": GSIM_SHA,
            "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
            "sgb_derivation": {
                "base": SGB_DERIVATION_BASE_SHA,
                "imported_development": SGB_DERIVATION_IMPORTED_SHA,
            },
        },
        "groups": entries,
    }


def _route_a_thin_film_provenance(
    stack: Mapping[str, Any], route: str
) -> dict[str, Any] | None:
    metadata = stack.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping.")
    value = metadata.get("route_a_thin_film")
    if route == "B":
        if value is not None:
            raise ValueError("Route B stack must not define route_a_thin_film.")
        return None
    if value is None:
        raise ValueError("Route A stack requires route_a_thin_film provenance.")
    if not isinstance(value, Mapping):
        raise TypeError("Route A route_a_thin_film provenance must be a mapping.")
    result = copy.deepcopy(dict(value))
    physical = metadata.get("physical_representation")
    if isinstance(physical, Mapping):
        from scgsim.geometry.models.physical import PhysicalRepresentationPlan

        plan = PhysicalRepresentationPlan.from_payload(physical)
        if result.get("variant") != plan.profile.name:
            raise ValueError(
                "Palace profile provenance differs from the neutral physical plan"
            )
        return result
    variant = result.get("variant")
    expected_label = {
        "substrate_face": "A_PRIME",
        "metal_gap_equivalent": "A",
    }.get(variant)
    if (
        result.get("schema_version") != 1
        or result.get("display_label") != expected_label
    ):
        raise ValueError(
            "Route A thin-film provenance variant or display label is invalid."
        )
    source = result.get("source_stack")
    if not isinstance(source, Mapping):
        raise TypeError("Route A thin-film provenance requires source_stack.")
    sha = source.get("sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or source.get("revision") != f"sha256:{sha}"
    ):
        raise ValueError("Route A thin-film source-stack identity is invalid.")
    return result


def _write_component_gds(component: Any, path: Path) -> None:
    """Write the component's GDS, supporting both string and pathlib styles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gds = getattr(component, "write_gds", None)
    if write_gds is None or not callable(write_gds):
        raise TypeError("component must provide write_gds(path)")

    try:
        write_gds(path)
    except TypeError:
        write_gds(str(path))


def _with_port_sheet_metadata(
    stack: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Attach authored ports after marking their exact face-metal owners splittable."""
    payload = copy.deepcopy(dict(stack))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping.")
    metadata = dict(metadata)
    existing = metadata.get("port_sheet_source_layers")
    if existing not in (None, [], ()) and records:
        raise ValueError("stack already defines port_sheet_source_layers.")
    effective_records = records if records else existing or ()
    if effective_records:
        layers = payload.get("layers")
        if not isinstance(layers, list):
            raise TypeError("stack layers must be a list for authored port sheets.")
        rewritten_layers = list(layers)
        for record in effective_records:
            if not isinstance(record, Mapping):
                raise TypeError("port-sheet source records must be mappings.")
            target_layer = record.get("target_layer")
            if not isinstance(target_layer, str) or not target_layer:
                raise ValueError("port-sheet source records require target_layer.")
            matches = [
                index
                for index, layer in enumerate(rewritten_layers)
                if isinstance(layer, Mapping)
                and (
                    layer.get("semantic_id") == target_layer
                    or (
                        isinstance(layer.get("metadata"), Mapping)
                        and layer["metadata"].get("logical_layer_id") == target_layer
                    )
                )
            ]
            if not matches:
                raise ValueError(
                    f"authored port target_layer {target_layer!r} must match a logical stack layer."
                )
            for index in matches:
                layer = rewritten_layers[index]
                if not isinstance(layer, Mapping):
                    raise TypeError("matched logical stack layer must be a mapping.")
                if (
                    layer.get("role") != "metal"
                    or layer.get("part_role") != "face_metal"
                ):
                    raise ValueError(
                        f"authored port target_layer {target_layer!r} must have face-metal semantics."
                    )
                geometry = layer.get("geometry")
                if not isinstance(geometry, Mapping):
                    raise TypeError(
                        f"authored port target_layer {target_layer!r} must define geometry mapping."
                    )
                rewritten = dict(layer)
                rewritten_geometry = dict(geometry)
                rewritten_geometry["split_polygons_as_entities"] = True
                rewritten["geometry"] = rewritten_geometry
                rewritten_layers[index] = rewritten
        payload["layers"] = rewritten_layers
        metadata["port_sheet_source_layers"] = copy.deepcopy(list(effective_records))
    payload["metadata"] = metadata
    return payload


def _component_gds_top_cell_name(*, component: Any, gds_path: Path) -> str:
    """Select a deterministic top cell name for SGB input."""
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError("gdstk is required for SGB input validation") from exc

    library = gdstk.read_gds(str(gds_path))
    cells_by_name = {cell.name: cell for cell in library.cells}

    component_name = str(getattr(component, "name", "component"))
    candidates = [
        component_name,
        component_name.split("$", 1)[0],
        "component",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in cells_by_name:
            return candidate

    top_cells = sorted(
        (cell for cell in library.top_level() if not str(cell.name).startswith("$$$")),
        key=lambda item: str(item.name),
    )
    if len(top_cells) != 1:
        names = ", ".join(sorted(cell.name for cell in top_cells))
        raise ValueError(
            "SGB GDS export needs deterministic top-level cell selection. "
            f"Candidates: {names}"
        )
    return top_cells[0].name


def _physical_group_values(value: Any) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple, set)) else (value,)
    return tuple(
        int(item)
        for item in raw
        if isinstance(item, int) and not isinstance(item, bool)
    )
