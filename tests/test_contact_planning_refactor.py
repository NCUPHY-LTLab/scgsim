"""Exercise contact geometry, membership and source-order failure behavior."""

from dataclasses import replace

import pytest

from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.planning.contact_planning.membership import (
    plan_mm_contact_records_with_components,
)
from scgsim.geometry.planning.contact_planning.patches import (
    plan_conductor_contact_patches,
)


def _context(source: GeometryBuildInput) -> RepresentationContext:
    return RepresentationContext(
        {
            entity.semantic_id: EntityRepresentation(
                physical_layer_id=None,
                representation="solid",
                source_z_interval_um=(
                    float(entity.geometry["z_min_um"]),
                    float(entity.geometry["z_max_um"]),
                ),
                effective_z_interval_um=(
                    float(entity.geometry["z_min_um"]),
                    float(entity.geometry["z_max_um"]),
                ),
            )
            for entity in source.entities
            if entity.material_kind == "conductor"
        }
    )


def _conductor(name: str, z_min: float, z_max: float) -> SemanticEntitySpec:
    return SemanticEntitySpec(
        semantic_id=name,
        role="conductor",
        material_id="pec",
        material_kind="conductor",
        priority=1,
        geometry_kind="polygon",
        geometry={
            "outer_loop": ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
            "hole_loops": (),
            "z_min_um": z_min,
            "z_max_um": z_max,
        },
        route_representations={"B": "cutout_boundary_shell", "C": "material_volume"},
        host_void_semantic_id="vacuum",
        net_id="signal",
        polygon_ids=(f"{name}-polygon",),
        metadata={"source_layer_name": f"layer-{name}"},
    )


def _host() -> SemanticEntitySpec:
    return SemanticEntitySpec(
        semantic_id="vacuum",
        role="solution_region",
        material_id="vacuum",
        material_kind="vacuum",
        priority=0,
        geometry_kind="polygon",
        geometry={
            "domain_bounds_um": {
                "x_min_um": -1.0,
                "x_max_um": 3.0,
                "y_min_um": -1.0,
                "y_max_um": 3.0,
            },
            "z_min_um": -1.0,
            "z_max_um": 3.0,
        },
    )


def test_contact_patch_keeps_source_fragments_and_one_component() -> None:
    """Join stacked same-net bodies without replacing their source identities."""
    lower, upper = (_conductor("lower", 0.0, 1.0), _conductor("upper", 1.0, 2.0))
    source = GeometryBuildInput((), (lower, upper, _host()))
    interfaces = plan_conductor_contact_patches(
        source, route="B", interfaces=(), representation_context=_context(source)
    )
    enriched, contacts, components = plan_mm_contact_records_with_components(
        source,
        route="B",
        interfaces=interfaces,
        representation_context=_context(source),
    )
    assert len(interfaces) == len(contacts) == 1
    contact = contacts[0]
    assert contact.area_um2 == 4.0
    assert contact.lower_source_fragment_ids == ("lower-polygon",)
    assert contact.upper_source_fragment_ids == ("upper-polygon",)
    assert contact.layer_provenance == {
        "lower": {
            "source_layer_name": "layer-lower",
            "gds_layer": None,
            "gds_datatype": None,
        },
        "upper": {
            "source_layer_name": "layer-upper",
            "gds_layer": None,
            "gds_datatype": None,
        },
    }
    assert components == (
        {
            "conductor_component_id": "COMP__lower",
            "members": ("lower", "upper"),
            "contact_ids": (contact.contact_id,),
            "net_id": "signal",
            "equipotential_id": None,
        },
    )
    assert enriched[0].metadata["hidden_solver_contact"] is True
    assert source.entities[:2] == (lower, upper)


def test_route_c_retains_its_contact_metadata_shape() -> None:
    """Keep the material-contact route's distinct source metadata fields."""
    source = GeometryBuildInput(
        (), (_conductor("lower", 0.0, 1.0), _conductor("upper", 1.0, 2.0))
    )
    (interface,) = plan_conductor_contact_patches(
        source, route="C", interfaces=(), representation_context=_context(source)
    )
    assert interface.metadata["contact_policy"] == "retained_material_contact"
    assert interface.metadata["boundary_volume_ids"] == ("lower", "upper")
    assert "contact_index" not in interface.metadata
    assert "contact_kind" not in interface.metadata


def test_first_contact_net_error_precedes_volumetric_overlap_error() -> None:
    """Resolve ordered declared contacts before validating unrelated body overlaps."""
    lower, upper = (_conductor("lower", 0.0, 1.0), _conductor("upper", 1.0, 2.0))
    source = GeometryBuildInput((), (lower, upper, _host()))
    interfaces = plan_conductor_contact_patches(
        source, route="B", interfaces=(), representation_context=_context(source)
    )
    conflicting = replace(upper, net_id="other")
    overlapping = _conductor("overlapping", 0.0, 1.0)
    invalid = replace(
        source, entities=(lower, conflicting, overlapping, source.entities[-1])
    )
    with pytest.raises(ValueError, match="shorts different nets 'signal' and 'other'"):
        plan_mm_contact_records_with_components(
            invalid,
            route="B",
            interfaces=interfaces,
            representation_context=_context(invalid),
        )
