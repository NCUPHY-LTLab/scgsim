"""Write paired mesh source artifacts (GDS, stack, receipts) and validate the declared Route-A film provenance."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from scgsim.geometry.models.physical import PhysicalRepresentationPlan


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
