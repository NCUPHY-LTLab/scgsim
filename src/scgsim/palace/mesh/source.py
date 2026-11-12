"""Write paired mesh records (JSON, receipts) and validate the declared Route-A film provenance."""

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
