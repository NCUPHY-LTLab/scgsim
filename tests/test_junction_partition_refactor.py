"""Preserve authored support binding and exact model-coordinate identity."""

import sys
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from types import FrameType

import pytest
from test_rectangular_junction_geometry import fixture

from scgsim.geometry import with_rectangular_junctions
from scgsim.geometry.grid.scope import UNGRIDDED
from scgsim.geometry.junctions.contact import (
    JunctionContactRequest,
    junction_contact_inventory,
    junction_source_binding,
)
from scgsim.geometry.junctions.core import (
    DEFAULT_FORMAT,
    canonical_sha256,
    plain_record,
)
from scgsim.geometry.junctions.derivation import derive_rectangular_junctions
from scgsim.geometry.junctions.local_frame import JunctionLocalFrame
from scgsim.geometry.junctions.model_state import JunctionModel
from scgsim.geometry.junctions.partition import partition_junctions
from scgsim.geometry.models.grid import GeometryGridContext, geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput, LayoutPolygonSpec


def test_contact_request_keeps_source_binding_distinct_from_effective_fields() -> None:
    """Keep the authored support binding reusable without effective-only fields."""
    source, spec = fixture()
    bound = with_rectangular_junctions(source, (spec,))
    inventory = junction_contact_inventory(
        bound, JunctionContactRequest(spec.source_polygon_id, "A", "B")
    )
    binding = junction_source_binding(inventory)
    assert inventory["source_z_um"] == inventory["effective_z_um"] == 0
    assert binding["source_hosts"]
    assert "effective_z_um" not in binding and "eligible_entity_ids" not in binding
    assert (
        junction_contact_inventory(
            bound, JunctionContactRequest(spec.source_polygon_id, "A", "B", binding)
        )
        == inventory
    )


def test_missing_physical_plan_fails_before_stored_binding_comparison() -> None:
    """Read the source projection authority before comparing stored source facts."""
    source, spec = fixture()
    first = source.entities[0]
    first = replace(first, metadata={**first.metadata, "physical_representation": {}})
    invalid = replace(source, entities=(first, *source.entities[1:]))
    with pytest.raises(
        ValueError, match="junction lowered source requires its common physical plan"
    ):
        junction_contact_inventory(
            invalid,
            JunctionContactRequest(spec.source_polygon_id, "A", "B", {"invalid": True}),
        )


def test_local_frame_keeps_complete_ring_order_and_deviation_ledger() -> None:
    """Project an orthogonal source frame without changing its polygon or ring order."""
    polygon = LayoutPolygonSpec("source", "metal", ((0, 0), (2, 0), (2, 1), (0, 1)))
    deviations: list[float] = []
    frame = JunctionLocalFrame(0.0, 1.0, -1.0, 0.0, (0.0, -2.0), deviations)
    assert frame.convert(polygon) == (
        (
            (0, 2_000_000_000),
            (0, 0),
            (1_000_000_000, 0),
            (1_000_000_000, 2_000_000_000),
        ),
    )
    assert deviations == [0.0, 0.0, 0.0, 0.0]
    assert polygon.exterior == ((0, 0), (2, 0), (2, 1), (0, 1))


def test_incidence_registry_retains_first_parameter_on_collapsed_event_error() -> None:
    """Reject distinct source parameters at one registered point without overwriting it."""
    source, _ = fixture()
    registry = JunctionModel(source, {}, {}, DEFAULT_FORMAT, UNGRIDDED)
    registry.events = defaultdict(dict)
    registry.members = defaultdict(set)
    edge = ("source", 0, 0)
    point = (0, 0)
    registry.event(edge, point, Fraction(0))
    with pytest.raises(
        ValueError,
        match="numerical junction incidence collapses distinct source parameters",
    ):
        registry.event(edge, point, Fraction(1))
    assert registry.events[edge][point] == Fraction(0)
    assert registry.members[point] == {edge}


def test_independent_derivations_keep_the_complete_source_bound_payload() -> None:
    """Keep deterministic derived identities while leaving authored source unchanged."""
    source, spec = fixture()
    bound = with_rectangular_junctions(source, (spec,))
    before = canonical_sha256(plain_record(bound))
    first = derive_rectangular_junctions(bound)
    second = derive_rectangular_junctions(bound)
    assert canonical_sha256(plain_record(first)) == canonical_sha256(
        plain_record(second)
    )
    assert canonical_sha256(plain_record(bound)) == before
    record = first.metadata["junction_partitions"][spec.junction_id]
    assert tuple(end["source_entity_id"] for end in record["ends"]) == (
        "chip/left",
        "chip/right",
    )


def test_empty_partition_reads_the_declared_grid_once_and_returns_its_input() -> None:
    """An empty selection still reads the input's declared grid, which refuses an input without one."""
    grid = GeometryGridContext(0.25)
    source = GeometryBuildInput((), (), metadata={"geometry_grid": grid.to_payload()})
    calls: list[str] = []

    def profile(frame: FrameType, event: str, argument: object) -> None:
        if event == "call" and frame.f_code is geometry_grid_context.__code__:
            calls.append("lookup")

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = partition_junctions(source, (), 0.5)
    finally:
        sys.setprofile(previous)
    assert result is source
    assert calls == ["lookup"]
    with pytest.raises(ValueError, match="declares no geometry grid"):
        partition_junctions(GeometryBuildInput((), ()), (), 0.5)
