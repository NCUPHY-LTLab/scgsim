"""Grouped physical surfaces retain independent component membership."""

from dataclasses import replace

import pytest

from scgsim.geometry.construction.export import export_physical_group_records
from scgsim.geometry.models.construction import ConstructionPlanRecord
from scgsim.geometry.models.tags import (
    BackendEntityTagRecord,
    StructuredFinalPhysicalGroupRecord,
    TagPlanRecord,
)
from scgsim.geometry.models.topology import SurfacePlanRecord


def _source(name, component, *, net="GND", equipotential="EQ_GND", ledger=True):
    provenance = {"source_polygon_ids": (f"polygon-{name}",)}
    if ledger:
        provenance.update(
            surface_contribution_schema="scgsim.surface-contributions.v1",
            surface_contribution_ledger=(
                {"contribution_id": f"contribution-{name}", "source_owner_ids": (name,)},
            ),
        )
    return SurfacePlanRecord(
        surface_id=name,
        owner_semantic_id=name,
        surface_role="conductor_surface",
        geometry_ref={},
        metadata={
            "route": "A",
            "representation": "surface_sheet",
            "interface_type": "MS_MA",
            "face_kind": "sheet",
            "owner_semantic_ids": (name,) if ledger else ("group-owner",),
            "boundary_volume_ids": ("substrate", "vacuum"),
            "source_provenance": provenance,
            "conductor_component_id": component,
            "net_id": net,
            "equipotential_id": equipotential,
        },
    )


def _export(*sources):
    plan = ConstructionPlanRecord(
        route="A",
        surfaces=sources,
        tags=tuple(
            TagPlanRecord("MS_MA__bumps", 2, "surface", s.surface_id, "conductor")
            for s in sources
        ),
        backend_entity_tags=tuple(
            BackendEntityTagRecord("surface", s.surface_id, (2, i + 1))
            for i, s in enumerate(sources)
        ),
    )
    return export_physical_group_records(plan)[0]


def test_group_preserves_distinct_components_and_every_source():
    sources = (_source("bump-a", "COMP_A"), _source("bump-b", "COMP_B"),
               _source("bump-a-side", "COMP_A"))
    group = _export(*sources)
    assert group.entity_tags == (1, 2, 3)
    assert group.conductor_component_id is None
    assert group.conductor_component_ids == ("COMP_A", "COMP_B")
    assert group.conductor_component_ids_by_source == {
        "bump-a": "COMP_A", "bump-b": "COMP_B", "bump-a-side": "COMP_A",
    }
    assert group.owner_semantic_ids == ("bump-a", "bump-b", "bump-a-side")
    assert group.source_provenance["sources"] == tuple(
        s.metadata["source_provenance"] for s in sources
    )


def test_single_component_retains_legacy_singular_field():
    group = _export(_source("a", "COMP_A", ledger=False),
                    _source("b", "COMP_A", ledger=False))
    assert group.conductor_component_id == "COMP_A"
    assert group.conductor_component_ids == ("COMP_A",)
    assert group.conductor_component_ids_by_source == {"a": "COMP_A", "b": "COMP_A"}
    legacy = replace(group, conductor_component_ids=(), conductor_component_ids_by_source={})
    assert legacy.conductor_component_id == "COMP_A"


def test_component_tuple_is_canonical_without_reordering_sources():
    group = _export(_source("b", "COMP_B"), _source("a", "COMP_A"))
    assert group.conductor_component_ids == ("COMP_A", "COMP_B")
    assert group.source_provenance["source_record_ids"] == ("b", "a")
    assert tuple(group.conductor_component_ids_by_source.items()) == (
        ("b", "COMP_B"), ("a", "COMP_A"),
    )
    assert group.entity_tags == (1, 2)
    with pytest.raises(ValueError, match="sorted unique"):
        replace(group, conductor_component_ids=("COMP_B", "COMP_A"))


def test_identity_free_surface_keeps_empty_component_fields():
    group = _export(_source("a", None, net=None, equipotential=None))
    assert group.conductor_component_id is None
    assert group.conductor_component_ids == ()
    assert group.conductor_component_ids_by_source == {}


@pytest.mark.parametrize("second", [
    _source("b", None),
    _source("b", "COMP_B", net="SIGNAL"),
    _source("b", "COMP_B", equipotential="EQ_OTHER"),
    _source("b", "COMP_B", ledger=False),
])
def test_incompatible_members_fail_without_partial_export(second):
    with pytest.raises(ValueError):
        _export(_source("a", "COMP_A"), second)


@pytest.mark.parametrize("field", ["net_id"])
def test_plural_components_require_explicit_electrical_identity(field):
    sources = tuple(_source(name, component) for name, component in (("a", "A"), ("b", "B")))
    sources = tuple(replace(s, metadata={**s.metadata, field: None}) for s in sources)
    with pytest.raises(ValueError, match=f"explicit {field}"):
        _export(*sources)


def test_plural_components_preserve_optional_equipotential_identity():
    group = _export(
        _source("a", "A", equipotential=None),
        _source("b", "B", equipotential=None),
    )
    assert group.equipotential_id is None
    assert group.net_id == "GND"
    assert group.conductor_component_ids == ("A", "B")


@pytest.mark.parametrize("value", [None, "", 42])
def test_optional_equipotential_rejects_mixed_or_malformed_values(value):
    with pytest.raises(ValueError):
        _export(_source("a", "A"), _source("b", "B", equipotential=value))
    if value is not None:
        with pytest.raises(ValueError):
            _export(_source("a", "A", equipotential=value),
                    _source("b", "B", equipotential=value))


@pytest.mark.parametrize("changes", [
    {"conductor_component_ids_by_source": {"a": "A"}},
    {"conductor_component_ids_by_source": {"a": "A", "b": "UNKNOWN"}},
    {"conductor_component_ids": ("A", "A")},
    {"conductor_component_id": "A"},
    {"net_id": None},
    {"equipotential_id": ""},
    {"equipotential_id": 42},
    {"source_provenance": {"source_record_ids": ("a", "b")}},
])
def test_typed_group_rejects_incomplete_or_conflicting_bindings(changes):
    group = _export(_source("a", "A"), _source("b", "B"))
    assert isinstance(group, StructuredFinalPhysicalGroupRecord)
    with pytest.raises(ValueError):
        replace(group, **changes)
