"""The installed SCGSim version, and the Route A film provenance a mesh's stack declares."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from scgsim.geometry.models.physical import PhysicalRepresentationPlan
from scgsim.refusal import refusal


def scgsim_version() -> str:
    """Return installed scgsim version from package metadata."""
    try:
        return package_version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Unable to determine installed scgsim version: no package metadata for scgsim; "
            "install SCGSim (uv sync, or pip install -e .) rather than running it from a source tree."
        ) from exc


def route_a_thin_film_provenance(
    stack: Mapping[str, Any], route: str
) -> dict[str, Any] | None:
    """Validate and detach the declared Route-A film provenance."""
    metadata = stack.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(
            refusal("stack metadata must be a mapping", found=type(metadata).__name__)
        )
    value = metadata.get("route_a_thin_film")
    if route == "B":
        if value is not None:
            raise ValueError(
                refusal("Route B stack must not define route_a_thin_film", found=value)
            )
        return None
    if value is None:
        raise ValueError(
            refusal("Route A stack requires route_a_thin_film provenance", found=value)
        )
    if not isinstance(value, Mapping):
        raise TypeError(
            refusal(
                "Route A route_a_thin_film provenance must be a mapping", found=value
            )
        )
    result = copy.deepcopy(dict(value))
    physical = metadata.get("physical_representation")
    if isinstance(physical, Mapping):
        plan = PhysicalRepresentationPlan.from_payload(physical)
        if result.get("variant") != plan.profile.name:
            raise ValueError(
                refusal(
                    "Palace profile provenance differs from the neutral physical plan",
                    found=result.get("variant"),
                    expected=plan.profile.name,
                )
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
            refusal(
                "Route A thin-film provenance variant or display label is invalid",
                found={
                    "schema_version": result.get("schema_version"),
                    "display_label": result.get("display_label"),
                },
                expected={"schema_version": 1, "display_label": expected_label},
            )
        )
    source = result.get("source_stack")
    if not isinstance(source, Mapping):
        raise TypeError(
            refusal("Route A thin-film provenance requires source_stack", found=source)
        )
    sha = source.get("sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or source.get("revision") != f"sha256:{sha}"
    ):
        raise ValueError(
            refusal(
                "Route A thin-film source-stack identity is invalid", found=dict(source)
            )
        )
    return result
