"""Direct regressions for shared geometry primitive contracts."""

from dataclasses import replace

from scgsim.geometry.grid.coordinates import coordinate_2d_key, segment_overlap_interval
from scgsim.geometry.models.entities import (
    is_solution_entity,
    is_vacuum_solution_entity,
    requires_route_representation,
)
from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.geometry.source.contact_pads import (
    entities_have_finite_overlap,
    resolve_contact_pad_attachment,
)


def _entity(
    semantic_id: str,
    *,
    geometry: dict[str, object] | None = None,
) -> SemanticEntitySpec:
    """Build one small semantic entity for primitive contract checks."""
    return SemanticEntitySpec(
        semantic_id=semantic_id,
        role="conductor",
        material_id="pec",
        material_kind="conductor",
        priority=1,
        geometry_kind="polygon",
        geometry=geometry or {},
    )


def test_coordinate_keys_round_to_the_model_grid():
    """Keep 2D coordinate rounding stable."""
    assert coordinate_2d_key((1.0000000004, -2.0)) == (1.0, -2.0)


def test_segment_intervals_ignore_endpoint_direction():
    """Keep collinear overlap parameters stable under reversal."""
    overlap = segment_overlap_interval((0, 0), (10, 0), (8, 0), (2, 0))
    reverse_overlap = segment_overlap_interval((10, 0), (0, 0), (8, 0), (2, 0))

    assert overlap == (0.2, 0.8)
    assert reverse_overlap == overlap
    assert segment_overlap_interval((0, 0), (10, 0), (10, 0), (12, 0)) is None
    assert segment_overlap_interval((1, 1), (1, 1), (0, 0), (2, 2)) is None
    assert segment_overlap_interval((0, 0), (10, 0), (2, 1e-8), (8, 1e-8)) is None


def test_contact_attachment_uses_declared_identity_net_and_finite_overlap():
    """Keep contact-pad attachment checks tied to the declared face metal."""
    loop = ((0, 0), (2, 0), (2, 2), (0, 2))
    pad = replace(
        _entity(
            "pad",
            geometry={"z_min_um": 0.0, "thickness_um": 1.0, "outer_loop": loop},
        ),
        part_role="contact_pad",
        net_id="signal",
        attached_face_metal_semantic_id="metal",
    )
    metal = replace(
        _entity(
            "metal",
            geometry={"z_min_um": 0.0, "thickness_um": 1.0, "outer_loop": loop},
        ),
        part_role="face_metal",
        net_id="signal",
    )
    touching = _entity(
        "touching",
        geometry={
            "z_min_um": 0.0,
            "thickness_um": 1.0,
            "outer_loop": ((2, 0), (4, 0), (4, 2), (2, 2)),
        },
    )

    assert entities_have_finite_overlap(pad, metal)
    assert not entities_have_finite_overlap(pad, touching)
    assert resolve_contact_pad_attachment(pad, (metal,)) is metal
    assert resolve_contact_pad_attachment(pad, (metal,), defer_overlap=True) is None
    vacuum = replace(_entity("vacuum"), role="solution_region", material_kind="vacuum")
    assert is_solution_entity(vacuum)
    assert is_vacuum_solution_entity(vacuum)
    assert not requires_route_representation(vacuum)
    assert requires_route_representation(metal)
