"""Expose shared canonical topology checks at established compiler paths."""

from scgsim.geometry.primitives.topology_validation import (
    validate_curve_plan_coverage,
    validate_no_surface_overlap,
    validate_surface_deduplication,
    validate_surface_use_counts,
    validate_volume_surface_closure,
)

__all__ = [
    "validate_curve_plan_coverage",
    "validate_no_surface_overlap",
    "validate_surface_deduplication",
    "validate_surface_use_counts",
    "validate_volume_surface_closure",
]
