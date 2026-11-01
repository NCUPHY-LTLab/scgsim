"""Exercise ordered vacuum sweeps and exact planar solution ownership."""

from dataclasses import replace

from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.planning.domain import (
    ConductorFaceRequest,
    conductor_face_solution_pieces,
    prepare_auto_vacuum_solution_regions,
)
from scgsim.geometry.planning.domain_planning.sidewalls import SolutionSidewallPlanner

LOOP = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))


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


def _obstacle() -> SemanticEntitySpec:
    return SemanticEntitySpec(
        semantic_id="metal",
        role="conductor",
        material_id="pec",
        material_kind="conductor",
        priority=1,
        geometry_kind="polygon",
        geometry={
            "outer_loop": LOOP,
            "hole_loops": (),
            "z_min_um": 0.0,
            "z_max_um": 2.0,
        },
        route_representations={"B": "cutout_boundary_shell"},
        host_void_semantic_id="vacuum",
    )


def _vacuum() -> SemanticEntitySpec:
    return SemanticEntitySpec(
        semantic_id="vacuum",
        role="solution_region",
        material_id="vacuum",
        material_kind="vacuum",
        priority=0,
        geometry_kind="polygon",
        geometry={},
        metadata={
            "is_auto_vacuum_region": True,
            "vacuum_region_padding_um": {
                "x_minus_um": 1.0,
                "x_plus_um": 1.0,
                "y_minus_um": 1.0,
                "y_plus_um": 1.0,
                "z_minus_um": 1.0,
                "z_plus_um": 1.0,
            },
        },
    )


def test_vacuum_sweep_retains_ordered_component_and_boundary_ledgers() -> None:
    """Keep the below, around and above components bound to the authored obstacle."""
    source = GeometryBuildInput((), (_vacuum(), _obstacle()))
    prepared = prepare_auto_vacuum_solution_regions(
        source, "B", representation_context=_context(source)
    )
    components = prepared.entities[:3]
    assert tuple((entity.semantic_id for entity in components)) == (
        "vacuum",
        "vacuum__0002",
        "vacuum__0003",
    )
    assert tuple(
        (entity.metadata["auto_vacuum_z_range_um"] for entity in components)
    ) == ((-1.0, 0.0), (0.0, 2.0), (2.0, 3.0))
    assert components[0].metadata["auto_vacuum_boundary_entity_ids"] == {
        "bottom": (),
        "top": ("metal",),
    }
    assert components[1].metadata["auto_vacuum_subtracting_entity_ids"] == ("metal",)
    assert len(components[1].geometry["hole_loops"]) == 1
    assert components[2].metadata["auto_vacuum_boundary_entity_ids"] == {
        "bottom": ("metal",),
        "top": (),
    }
    assert source.entities[0].geometry == {}
    planner = SolutionSidewallPlanner(
        prepared,
        route="B",
        solution=components[1],
        outer_loop=components[1].geometry["outer_loop"],
        z_min_um=0.0,
        z_max_um=2.0,
        representation_context=_context(source),
    )
    first = planner.build()
    assert first == planner.build()
    assert planner.z_min_um == 0.0 and planner.z_max_um == 2.0


def test_conductor_face_partitions_cover_in_solution_id_order() -> None:
    """Split a face between two adjacent solutions without choosing one host."""
    left = replace(
        _vacuum(),
        semantic_id="left",
        geometry={
            "domain_bounds_um": {
                "x_min_um": 0.0,
                "x_max_um": 1.0,
                "y_min_um": 0.0,
                "y_max_um": 2.0,
            },
            "z_min_um": 2.0,
            "z_max_um": 3.0,
        },
        metadata={},
    )
    right = replace(
        left,
        semantic_id="right",
        geometry={
            **left.geometry,
            "domain_bounds_um": {
                "x_min_um": 1.0,
                "x_max_um": 2.0,
                "y_min_um": 0.0,
                "y_max_um": 2.0,
            },
        },
    )
    entity = _obstacle()
    source = GeometryBuildInput((), (right, left, entity))
    pieces = conductor_face_solution_pieces(
        ConductorFaceRequest(
            source,
            route="B",
            entity=entity,
            shell_part="top",
            geometry_refs=({"outer_loop": LOOP, "hole_loops": ()},),
        )
    )
    assert tuple((owner for owner, _ in pieces)) == ("left", "right")
    assert all((reference["outer_loop"] for _, reference in pieces))
