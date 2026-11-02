"""Write paired mesh source artifacts and preserve version-bound source provenance."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from scgsim.geometry.models.physical import PhysicalRepresentationPlan

from .provenance import (
    GSIM_SHA,
    SGB_DERIVATION_BASE_SHA,
    SGB_DERIVATION_IMPORTED_SHA,
    SGB_RUNTIME_AUTHORITY,
)


def scgsim_version() -> str:
    """Return installed scgsim version from package metadata."""
    try:
        return package_version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError("Unable to determine installed scgsim version.") from exc


def write_indium_ground_bump_receipt(
    *, run_dir: Path, gds_path: Path, receipt: Mapping[str, Any]
) -> None:
    """Persist public fill provenance before the generic handoff archive is made."""
    payload = copy.deepcopy(dict(receipt))
    payload["design_gds"] = {
        "path": gds_path.relative_to(run_dir).as_posix(),
        "sha256": hashlib.sha256(gds_path.read_bytes()).hexdigest(),
    }
    write_json(run_dir / "metadata" / "indium_ground_bumps.json", payload)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a paired UTF-8 JSON artifact with the established formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_mesh_manifest(
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


def route_a_thin_film_provenance(
    stack: Mapping[str, Any], route: str
) -> dict[str, Any] | None:
    """Validate and detach the declared Route-A film provenance."""
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
        plan = PhysicalRepresentationPlan.from_payload(physical)
        if result.get("variant") != plan.profile.name:
            raise ValueError(
                "Palace profile provenance differs from the neutral physical plan"
            )
        return result
    variant: Any = result.get("variant")
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


def write_component_gds(component: Any, path: Path) -> None:
    """Write the component's GDS, supporting both string and pathlib styles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gds = getattr(component, "write_gds", None)
    if write_gds is None or not callable(write_gds):
        raise TypeError("component must provide write_gds(path)")

    try:
        write_gds(path)
    except TypeError:
        write_gds(str(path))


def with_port_sheet_metadata(
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


def component_gds_top_cell_name(*, component: Any, gds_path: Path) -> str:
    """Select a deterministic top cell name for SGB input."""
    try:
        import gdstk  # noqa: PLC0415 -- GDS validation needs the optional Palace extra.
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
