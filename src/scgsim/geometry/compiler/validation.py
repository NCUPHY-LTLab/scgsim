"""Expose compiler validators at their established import paths.

Coverage checks own route/interface/partition/tag correspondence. Topology
checks own canonical connectivity, live-surface overlap and closed shells.
These are direct re-exports of the source functions, with no extra validation.
"""

from scgsim.geometry.compiler.coverage_validation import (
    validate_backend_tag_ledger,
    validate_interface_surface_source_of_truth,
    validate_route_operation_coverage,
    validate_route_volume_surface_refs,
    validate_selected_route,
    validate_surface_partition_coverage,
    validate_surface_sheet_interface_coverage,
    validate_tag_plan_coverage,
)
from scgsim.geometry.compiler.topology_validation import (
    validate_curve_plan_coverage,
    validate_no_surface_overlap,
    validate_surface_deduplication,
    validate_surface_use_counts,
    validate_volume_surface_closure,
)

# Existing internal spelling remains the same function as the public validator.
_validate_surface_partition_coverage = validate_surface_partition_coverage

__all__ = [
    "validate_backend_tag_ledger",
    "validate_curve_plan_coverage",
    "validate_interface_surface_source_of_truth",
    "validate_no_surface_overlap",
    "validate_route_operation_coverage",
    "validate_route_volume_surface_refs",
    "validate_selected_route",
    "validate_surface_deduplication",
    "validate_surface_partition_coverage",
    "validate_surface_sheet_interface_coverage",
    "validate_surface_use_counts",
    "validate_tag_plan_coverage",
    "validate_volume_surface_closure",
]
