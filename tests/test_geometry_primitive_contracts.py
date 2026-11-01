"""Direct regressions for shared geometry primitive contracts."""

from dataclasses import replace

from scgsim.geometry._primitives.entity_validation import (
    _entities_have_finite_overlap,
    _is_solution_entity,
    _is_vacuum_solution_entity,
    _requires_route_representation,
    _resolve_contact_pad_attachment,
)
from scgsim.geometry._primitives.geometry_refs import (
    entity_geometry_ref,
    route_entity_geometry_ref,
    sidewall_geometry_refs,
)
from scgsim.geometry._primitives.spatial import (  # private-import: lock down shared tolerance and canonicalization contracts
    _bounds_contains_point,
    _bounds_overlap,
    _coordinate_2d_key,
    _coordinate_key,
    _geometry_ref_surface_z_um,
    _interpolate_2d,
    _interpolate_3d,
    _intersect_bounds,
    _interval_complement,
    _line_key_2d,
    _range_within_bounds,
    _segment_overlap_interval,
    _undirected_xy_edge,
    _vector_cross,
    _vector_subtract,
    _z_key,
)
from scgsim.geometry._primitives.surface_records import (
    _canonical_face_signature_3d,
    _interface_kinds,
    _surface_boundary_volume_ids,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import InterfacePlanRecord, SurfacePlanRecord


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


def test_spatial_keys_and_vector_operations_keep_coordinate_order():
    """Keep coordinate rounding, directed math, and undirected edge identity stable."""
    assert _coordinate_key((1.0000000004, -2.0, 3.25)) == (1.0, -2.0, 3.25)
    assert _coordinate_2d_key((1.0000000004, -2.0)) == (1.0, -2.0)
    assert _z_key(2.0000000004) == 2.0
    assert _undirected_xy_edge((2, 1), (0, 3)) == ((0.0, 3.0), (2.0, 1.0))
    assert _undirected_xy_edge((0, 3), (2, 1)) == ((0.0, 3.0), (2.0, 1.0))
    assert _vector_subtract((4, 3, 2), (1, 1, 1)) == (3, 2, 1)
    assert _vector_cross((1, 0, 0), (0, 1, 0)) == (0, 0, 1)
    assert _interpolate_2d((0, 2), (4, 6), 0.25) == (1.0, 3.0)
    assert _interpolate_3d((0, 2, 4), (4, 6, 8), 0.25) == (1.0, 3.0, 5.0)


def test_bounds_distinguish_tolerance_overlap_from_positive_intersection():
    """Preserve tolerant overlap tests and exact positive-area intersection bounds."""
    first = {"x_min_um": 0.0, "x_max_um": 1.0, "y_min_um": 0.0, "y_max_um": 1.0}
    touching = {
        "x_min_um": 1.0,
        "x_max_um": 2.0,
        "y_min_um": 0.0,
        "y_max_um": 1.0,
    }
    close = {**touching, "x_min_um": 1.0 - 5e-10}

    assert not _bounds_overlap(first, touching)
    assert _intersect_bounds(first, touching) is None
    assert not _bounds_overlap(first, close)
    assert _intersect_bounds(first, close) == {
        "x_min_um": 1.0 - 5e-10,
        "x_max_um": 1.0,
        "y_min_um": 0.0,
        "y_max_um": 1.0,
    }
    assert _bounds_contains_point(first, (1.0, 1.0))
    assert not _bounds_contains_point(first, (1.0 + 1e-12, 1.0))
    assert _range_within_bounds(-5e-10, 1.0 + 5e-10, 0.0, 1.0)
    assert not _range_within_bounds(-2e-9, 1.0, 0.0, 1.0)


def test_segment_intervals_and_line_keys_ignore_endpoint_direction():
    """Keep collinear overlap parameters and line identity stable under reversal."""
    overlap = _segment_overlap_interval((0, 0), (10, 0), (8, 0), (2, 0))
    reverse_overlap = _segment_overlap_interval((10, 0), (0, 0), (8, 0), (2, 0))

    assert overlap == (0.2, 0.8)
    assert reverse_overlap == overlap
    assert _segment_overlap_interval((0, 0), (10, 0), (10, 0), (12, 0)) is None
    assert _segment_overlap_interval((1, 1), (1, 1), (0, 0), (2, 2)) is None
    assert _segment_overlap_interval((0, 0), (10, 0), (2, 1e-8), (8, 1e-8)) is None

    forward_key = _line_key_2d((1, 2), (5, 4))
    reverse_key = _line_key_2d((5, 4), (1, 2))
    assert forward_key == reverse_key
    assert _line_key_2d((1, 2), (1, 2)) is None


def test_interval_complement_clips_and_merges_with_shared_tolerance():
    """Preserve clipping, touching-interval merge, and exposed-span ordering."""
    assert _interval_complement(()) == ((0.0, 1.0),)
    assert _interval_complement(((-0.2, 0.25), (0.25, 0.5), (0.8, 1.2))) == (
        (0.5, 0.8),
    )
    assert _interval_complement(((0.2, 0.4), (0.4000000005, 0.7))) == (
        (0.0, 0.2),
        (0.7, 1.0),
    )


def test_geometry_ref_surface_plane_precedence_is_unchanged():
    """Keep shell, explicit contact plane, and entity-base plane precedence."""
    assert (
        _geometry_ref_surface_z_um(
            {"z_min_um": 1.0, "z_max_um": 4.0, "shell_part": "top"}
        )
        == 4.0
    )
    assert (
        _geometry_ref_surface_z_um(
            {"z_min_um": 1.0, "thickness_um": 2.0, "shell_part": "top"}
        )
        == 3.0
    )
    assert (
        _geometry_ref_surface_z_um(
            {"z_min_um": 1.0, "shell_part": "bottom", "z_max_um": 4.0}
        )
        == 1.0
    )
    assert (
        _geometry_ref_surface_z_um(
            {"z_min_um": 1.0, "plane": {"axis": "z", "value_um": 2.5}}
        )
        == 2.5
    )
    assert _geometry_ref_surface_z_um({"z_min_um": 1.0}) == 1.0


def test_route_a_cutout_ref_uses_recognized_contact_planes_only():
    """Extend cutout bounds from recognized upper-bottom and lower-top contacts."""
    entity = _entity(
        "metal",
        geometry={"z_min_um": 1.0, "thickness_um": 2.0, "z_um": 1.0},
    )
    upper_contact = InterfacePlanRecord(
        interface_id="MM__upper",
        kind="MM",
        owner_semantic_ids=("metal", "upper"),
        recognition_rule="coplanar_conductor_contact_patch",
        metadata={
            "upper_entity_id": "metal",
            "upper_face": "bottom",
            "contact_plane": {"axis": "z", "value_um": 0.0},
        },
    )
    lower_contact = InterfacePlanRecord(
        interface_id="MM__lower",
        kind="MM",
        owner_semantic_ids=("lower", "metal"),
        recognition_rule="coplanar_conductor_contact_patch",
        metadata={
            "lower_entity_id": "metal",
            "lower_face": "top",
            "plane": {"axis": "z", "value_um": 5.0},
        },
    )
    unrelated = InterfacePlanRecord(
        interface_id="MM__unrelated",
        kind="MM",
        owner_semantic_ids=("metal", "other"),
        recognition_rule="unrecognized_contact",
        metadata={
            "upper_entity_id": "metal",
            "upper_face": "bottom",
            "contact_plane": {"axis": "z", "value_um": -10.0},
        },
    )

    projected = route_entity_geometry_ref(
        build_input=GeometryBuildInput((), (entity,)),
        route="A",
        entity=entity,
        representation="cutout_boundary_shell",
        interfaces=(upper_contact, unrelated, lower_contact),
    )

    assert projected["z_min_um"] == 0.0
    assert projected["z_max_um"] == 5.0
    assert projected["thickness_um"] == 5.0
    assert projected["route_a_cutout_z_range_um"] == (0.0, 5.0)
    assert entity_geometry_ref(entity, representation="solid")["z_um"] == 1.0


def test_sidewall_refs_preserve_outer_and_hole_edge_order():
    """Build ordered sidewall quads for the exterior and each hole edge."""
    refs = sidewall_geometry_refs(
        {
            "z_min_um": 1.0,
            "thickness_um": 2.0,
            "outer_loop": ((0, 0), (2, 0), (2, 2), (0, 2)),
            "hole_loops": (((0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)),),
        }
    )

    assert len(refs) == 8
    assert refs[0] == {
        "quad_points": ((0, 0, 1.0), (2, 0, 1.0), (2, 0, 3.0), (0, 0, 3.0)),
        "sidewall_ring_role": "outer",
        "sidewall_edge_index": 0,
    }
    assert refs[4]["sidewall_ring_role"] == "hole_0000"
    assert refs[4]["sidewall_edge_index"] == 0


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

    assert _entities_have_finite_overlap(pad, metal)
    assert not _entities_have_finite_overlap(pad, touching)
    assert _resolve_contact_pad_attachment(pad, (metal,)) is metal
    assert _resolve_contact_pad_attachment(pad, (metal,), defer_overlap=True) is None
    vacuum = replace(_entity("vacuum"), role="solution_region", material_kind="vacuum")
    assert _is_solution_entity(vacuum)
    assert _is_vacuum_solution_entity(vacuum)
    assert not _requires_route_representation(vacuum)
    assert _requires_route_representation(metal)


def test_surface_record_helpers_keep_canonical_kind_and_face_order():
    """Keep canonical interface order, face identity, and owner deduplication."""
    interface = InterfacePlanRecord(
        interface_id="MM__surface",
        kind="MM",
        owner_semantic_ids=("metal", "substrate"),
        recognition_rule="test",
        metadata={"interface_kinds": ("SA", "MA", "SA")},
    )
    ring = ((0, 0, 1), (2, 0, 1), (2, 3, 1), (0, 3, 1))
    rotated = ring[2:] + ring[:2]

    assert _interface_kinds(interface) == ("MM", "MA", "SA")
    assert _canonical_face_signature_3d(ring) == _canonical_face_signature_3d(
        tuple(reversed(rotated))
    )
    surface = SurfacePlanRecord(
        surface_id="surface",
        owner_semantic_id="metal",
        surface_role="boundary",
        geometry_ref={},
        metadata={
            "boundary_volume_ids": ("substrate", "metal", "substrate", "unknown")
        },
    )
    assert _surface_boundary_volume_ids(
        surface, known_entity_ids={"metal", "substrate"}
    ) == ("substrate", "metal")

    embedded_sheet = SurfacePlanRecord(
        surface_id="sheet",
        owner_semantic_id="sheet",
        surface_role="sheet",
        geometry_ref={},
        metadata={"embedded_surface_sheet": True},
    )
    assert (
        _surface_boundary_volume_ids(embedded_sheet, known_entity_ids={"sheet"}) == ()
    )
