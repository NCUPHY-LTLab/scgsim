"""Bind fresh AEDT preparation to the neutral physical representation owner.

Geometry planning owns classification, endpoint mapping and enclosure projection.
This consumer only selects the explicit AEDT profile and returns its context;
historical coordinate maps remain in the version-bound request reader.
"""
from __future__ import annotations

from typing import Any

from scgsim.aedt._data import canonical_sha256, detached

from scgsim.aedt.specs.modeling import Modeling
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.planning.physical import interpret_physical_geometry


def _interpret_geometry(build_input: Any, stack: Any, modeling: Modeling):
    effective_input, effective_stack, plan = interpret_physical_geometry(
        build_input, stack, ApproximationProfile(modeling, "v1", {})
    )
    effective_stack["metadata"]["physical_source_stack_sha256"] = canonical_sha256(detached(stack))
    return effective_input, effective_stack, plan


def _effective_geometry_input(build_input: Any, stack: Any, modeling: Modeling):
    """Return common effective geometry, stack and its planner projection."""
    effective_input, effective_stack, plan = _interpret_geometry(
        build_input, stack, modeling
    )
    return effective_input, effective_stack, plan.context
