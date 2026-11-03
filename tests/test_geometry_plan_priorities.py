"""``GeometryPlan.set_entity_priorities``: the assembly-level overlap priority (higher wins)."""

from __future__ import annotations

import pytest
from sgb_inputs import electrode_plan

from scgsim.geometry import ApproximationProfile, interpret_physical_geometry
from scgsim.geometry.sgb.errors import OverlapError
from scgsim.geometry.sgb.model import build_conformal_model


def priorities_of(plan) -> dict[str, int]:
    """The priority of each conductor Entity of a prepared plan, by qualified Entity id (polygon suffix removed)."""
    entities = plan.prepare().geometry_input.entities
    return {
        e.semantic_id.rsplit("_", 1)[0]: e.priority
        for e in entities
        if e.semantic_id.startswith("chip/")
    }


def test_a_priority_is_carried_into_the_snapshot_entities(monkeypatch):
    """The Entity the caller ranked carries the number; the others keep the source value."""
    plan = electrode_plan(monkeypatch)
    source = priorities_of(plan)
    plan.set_entity_priorities({"chip/pad": 3})
    carried = priorities_of(plan)
    assert carried["chip/pad"] == 3
    assert carried["chip/electrode"] == source["chip/electrode"]


def test_an_unknown_entity_is_refused_and_the_previous_priorities_stay(monkeypatch):
    """Like ``set_nets``, the replacement is atomic: one bad entry changes nothing."""
    plan = electrode_plan(monkeypatch)
    source = priorities_of(plan)
    plan.set_entity_priorities({"chip/pad": 2})
    with pytest.raises(ValueError, match=r"unknown Entity ID 'chip/missing'"):
        plan.set_entity_priorities({"chip/electrode": 5, "chip/missing": 1})
    assert priorities_of(plan)["chip/pad"] == 2
    assert priorities_of(plan)["chip/electrode"] == source["chip/electrode"]


@pytest.mark.parametrize("value", [1.5, "2", True, None])
def test_a_priority_must_be_an_integer(monkeypatch, value):
    """A float, text or bool is not a priority."""
    plan = electrode_plan(monkeypatch)
    with pytest.raises(TypeError, match="integer"):
        plan.set_entity_priorities({"chip/pad": value})


def test_the_mapping_is_replaced_as_a_whole_and_must_be_a_mapping(monkeypatch):
    """A second call forgets the first; a list is not accepted."""
    plan = electrode_plan(monkeypatch)
    source = priorities_of(plan)
    plan.set_entity_priorities({"chip/pad": 2})
    plan.set_entity_priorities({"chip/electrode": 7})
    assert priorities_of(plan) == {**source, "chip/electrode": 7}
    with pytest.raises(TypeError, match="map qualified Entity IDs"):
        plan.set_entity_priorities([("chip/pad", 1)])


def build(plan):
    """The SGB model of a plan's snapshot under the solid profile, Route B."""
    snapshot = plan.prepare()
    effective, _, physical = interpret_physical_geometry(
        snapshot.geometry_input, snapshot.stack, ApproximationProfile("solid", "v1", {})
    )
    return build_conformal_model(effective, physical.context, "B")


def conductor_volume_um3(model, entity: str) -> float:
    """The volume (um^3) of the conductor volume of an Entity."""
    volume = next(v for v in model.topology.volumes.values() if v.domain.name == entity)
    return (
        float(sum(s.signed_volume for s in volume.shells)) * model.topology.grid_um**2
    )


def test_overlapping_same_net_entities_need_a_priority_and_the_higher_one_owns_the_overlap(
    monkeypatch,
):
    """The overlap error names the plan API; with a priority the overlap belongs to the higher Entity."""
    plan = electrode_plan(monkeypatch, gap_um=-5.0, same_net=True)
    with pytest.raises(OverlapError, match="set_entity_priorities|priority"):
        build(plan)
    plan.set_entity_priorities({"chip/pad": 5})
    model = build(plan)
    assert conductor_volume_um3(model, "chip/pad_0000") == pytest.approx(20 * 20 * 0.2)
    plan.set_entity_priorities({"chip/electrode": 9})
    assert conductor_volume_um3(build(plan), "chip/pad_0000") == pytest.approx(
        (20 * 20 - 5 * 20) * 0.2
    )
