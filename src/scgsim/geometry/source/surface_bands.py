"""Attach explicit distance partition requests without changing source geometry."""

from collections.abc import Sequence
from dataclasses import replace

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.surface_bands import SurfaceDistanceBandSpec


def with_surface_distance_bands(
    build_input: GeometryBuildInput, specs: Sequence[SurfaceDistanceBandSpec]
) -> GeometryBuildInput:
    """Replace distance partition requests; parent surface IDs are compiler IDs."""
    requests = tuple(specs)
    parents = [request.parent_surface_id for request in requests]
    if len(set(parents)) != len(parents):
        raise ValueError(
            "each parent surface may have only one distance partition request"
        )
    return replace(
        build_input,
        metadata={
            **build_input.metadata,
            "surface_distance_bands": tuple(
                request.to_payload() for request in requests
            ),
        },
    )
