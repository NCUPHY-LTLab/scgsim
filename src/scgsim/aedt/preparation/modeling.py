"""Interpret a prepared geometry under the requested AEDT modeling and return its effective input and stack.

``scgsim.geometry.interpretation`` classifies layers, maps endpoints and projects
the enclosure. This module only selects the explicit AEDT profile and returns its context.
"""

from __future__ import annotations

from typing import Any

from scgsim.aedt.data import canonical_sha256, detached
from scgsim.aedt.specs.modeling import Modeling
from scgsim.geometry.interpretation.physical import interpret_physical_geometry
from scgsim.geometry.models.physical import ApproximationProfile


def interpret_geometry(build_input: Any, stack: Any, modeling: Modeling):
    """Bind the requested physical model and retain its source-stack identity."""
    effective_input, effective_stack, plan = interpret_physical_geometry(
        build_input, stack, ApproximationProfile(modeling, "v1", {})
    )
    effective_stack["metadata"]["physical_source_stack_sha256"] = canonical_sha256(
        detached(stack)
    )
    return effective_input, effective_stack, plan


def effective_geometry_input(build_input: Any, stack: Any, modeling: Modeling):
    """Return the effective geometry input, the effective stack and the plan's context."""
    effective_input, effective_stack, plan = interpret_geometry(
        build_input, stack, modeling
    )
    return effective_input, effective_stack, plan.context
