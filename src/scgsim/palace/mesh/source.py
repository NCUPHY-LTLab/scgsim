"""The installed SCGSim version, and the Route A film provenance a mesh's stack declares."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from scgsim.geometry.models.physical import PhysicalRepresentationPlan
from scgsim.geometry.models.refusals import Declared, internal
from scgsim.refusal import ENVIRONMENT, found_text


def scgsim_version() -> str:
    """Return installed scgsim version from package metadata."""
    try:
        return package_version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            Declared(
                "Unable to determine installed scgsim version", ENVIRONMENT
            ).refusal(
                "no package metadata for scgsim; install SCGSim (uv sync, or pip install -e .) rather than running it from a source tree"
            )
        ) from exc


def route_a_thin_film_provenance(
    stack: Mapping[str, Any], route: str
) -> dict[str, Any] | None:
    """Validate and detach the declared Route-A film provenance."""
    metadata = stack.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(
            internal(
                "stack metadata must be a mapping"
                + ": "
                + found_text(type(metadata).__name__)
            )
        )
    value = metadata.get("route_a_thin_film")
    if route == "B":
        if value is not None:
            raise ValueError(
                internal(
                    "Route B stack must not define route_a_thin_film"
                    + ": "
                    + found_text(value)
                )
            )
        return None
    if value is None:
        raise ValueError(
            internal(
                "Route A stack requires route_a_thin_film provenance"
                + ": "
                + found_text(value)
            )
        )
    if not isinstance(value, Mapping):
        raise TypeError(
            internal(
                "Route A route_a_thin_film provenance must be a mapping"
                + ": "
                + found_text(value)
            )
        )
    result = copy.deepcopy(dict(value))
    physical = metadata.get("physical_representation")
    if isinstance(physical, Mapping):
        plan = PhysicalRepresentationPlan.from_payload(physical)
        if result.get("variant") != plan.profile.name:
            raise ValueError(
                internal(
                    "Palace profile provenance differs from the neutral physical plan"
                    + ": "
                    + found_text(result.get("variant"), plan.profile.name)
                )
            )
        return result
    _validate_standalone_film(result)
    return result


def _validate_standalone_film(result: dict[str, Any]) -> None:
    """A film record without a physical plan names its variant's label and the hash of its source stack."""
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
            internal(
                "Route A thin-film provenance variant or display label is invalid"
                + ": "
                + found_text(
                    {
                        "schema_version": result.get("schema_version"),
                        "display_label": result.get("display_label"),
                    },
                    {"schema_version": 1, "display_label": expected_label},
                )
            )
        )
    source = result.get("source_stack")
    if not isinstance(source, Mapping):
        raise TypeError(
            internal(
                "Route A thin-film provenance requires source_stack"
                + ": "
                + found_text(source)
            )
        )
    sha = source.get("sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or source.get("revision") != f"sha256:{sha}"
    ):
        raise ValueError(
            internal(
                "Route A thin-film source-stack identity is invalid"
                + ": "
                + found_text(dict(source))
            )
        )
