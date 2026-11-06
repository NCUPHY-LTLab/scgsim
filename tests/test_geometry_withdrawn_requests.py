"""Withdrawn geometry requests fail before planning."""

from dataclasses import replace

import pytest
from test_geometry_context_propagation import conductor_input

from scgsim.geometry import GeometryPlan
from scgsim.geometry.compiler import build_route_construction_plan


def test_withdrawn_geometry_request_fails_before_planning():
    """Reject withdrawn surface-distance metadata before route planning starts."""
    source, _ = conductor_input()
    requested = replace(
        source, metadata={"surface_distance_bands": [{"parent_surface_id": "old"}]}
    )
    with pytest.raises(ValueError, match="withdrawn"):
        build_route_construction_plan(requested, route="B")
    assert not hasattr(GeometryPlan, "set_surface_distance_bands")
