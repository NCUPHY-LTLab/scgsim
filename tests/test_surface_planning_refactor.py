"""Preserve ordered sheet contributions and final surface ownership decisions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.representation import (
    EntityRepresentation,
    RepresentationContext,
)
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.surface_planning.contracts import (
    with_surface_contract_metadata,
)
from scgsim.geometry.planning.surface_planning.contributions import (
    plan_representation_contribution_geometry,
)


def conductor_input():
    """Build one conductor with its vacuum host and source stack."""
    metal_loop = ((0, 0), (4, 0), (4, 4), (0, 4))
    vacuum_loop = ((-2, -2), (6, -2), (6, 6), (-2, 6))
    polygon = LayoutPolygonSpec("metal_polygon", "metal", metal_loop)
    metal = SemanticEntitySpec(
        "metal",
        "conductor",
        "metal_material",
        "conductor",
        1,
        "polygon",
        net_id="signal",
        polygon_ids=(polygon.polygon_id,),
        host_void_semantic_id="vacuum",
        geometry={
            "outer_loop": metal_loop,
            "z_min_um": 0,
            "z_max_um": 1,
            "z_um": 0,
            "thickness_um": 1,
        },
    )
    vacuum = SemanticEntitySpec(
        "vacuum",
        "solution_region",
        "vacuum_material",
        "vacuum",
        0,
        "polygon",
        geometry={
            "outer_loop": vacuum_loop,
            "z_min_um": -2,
            "z_max_um": 6,
            "thickness_um": 8,
            "domain_bounds_um": {
                "x_min_um": -2,
                "x_max_um": 6,
                "y_min_um": -2,
                "y_max_um": 6,
                "z_min_um": -2,
                "z_max_um": 6,
            },
        },
    )
    stack = {
        "layers": [
            {"semantic_id": "metal", "geometry": dict(metal.geometry), "metadata": {}}
        ],
        "solution_regions": {
            "vacuum": {
                "semantic_id": "vacuum",
                "geometry": dict(vacuum.geometry),
                "metadata": {},
            }
        },
        "materials": {
            "metal_material": {"kind": "conductor"},
            "vacuum_material": {"kind": "vacuum", "permittivity": 1},
        },
        "metadata": {},
    }
    return GeometryBuildInput((polygon,), (metal, vacuum)), stack


@pytest.mark.parametrize("same_material_sides", (False, True))
@pytest.mark.parametrize("with_hole", (False, True))
def test_sheet_children_keep_both_ordered_side_contributions(
    same_material_sides, with_hole
):
    """Keep each local sheet child's bottom and top evidence, including holes."""
    source, context = sheet_source(
        same_material_sides=same_material_sides, with_hole=with_hole
    )
    _, surfaces = plan_representation_contribution_geometry(source, context)
    sheets = tuple(s for s in surfaces if s.metadata.get("embedded_surface_sheet"))
    assert len(sheets) == 1
    for surface in sheets:
        provenance = surface.metadata["source_provenance"]
        ledger = provenance["surface_contribution_ledger"]
        assert tuple(row["side"] for row in ledger) == ("bottom", "top")
        assert tuple(row["classification"] for row in ledger) == (
            ("MS", "MS") if same_material_sides else ("MS", "MA")
        )
        assert ledger[0]["effective_domain_ids"] == (
            surface.metadata["boundary_volume_ids"][0],
        )
        assert ledger[1]["effective_domain_ids"] == ("vacuum",)
        assert provenance["native_aggregation"]["contribution_count"] == 2
        assert provenance["native_aggregation"]["policy"] == (
            "shared_native_surface_once"
            if same_material_sides
            else "one_row_per_interface_kind"
        )
    assert tuple(s.metadata["boundary_volume_ids"][0] for s in sheets) == ("substrate",)
    assert any(bool(s.geometry_ref.get("hole_loops")) for s in sheets) == with_hole


@pytest.mark.parametrize(
    "failure,exception,message",
    (
        ("net", ValueError, "shared has ambiguous conductor net"),
        ("component", ValueError, "shared spans independent conductor components"),
        ("equipotential", ValueError, "shared has ambiguous equipotential id"),
        ("layer", ValueError, "shared has ambiguous conductor source_layer_name"),
        ("provenance", TypeError, "shared source_provenance must be a mapping"),
    ),
)
def test_surface_contract_preserves_failure_priority(failure, exception, message):
    """Report the first conflicting ownership field before later invalid metadata."""
    source, surface, ledger = contract_source(failure)
    with pytest.raises(exception, match=f"^{message}$"):
        with_surface_contract_metadata(
            source,
            route="B",
            interfaces=(),
            mm_contacts=(),
            surfaces=(surface,),
            _conductor_component_ledger=ledger,
        )


def test_surface_contract_uses_the_complete_declared_component_ledger():
    """Retain declared component facts even when no planar MM record exists."""
    source, surface, ledger = contract_source("valid")
    (bound,) = with_surface_contract_metadata(
        source,
        route=None,
        interfaces=(),
        mm_contacts=(),
        surfaces=(surface,),
        _conductor_component_ledger=ledger,
    )
    assert bound.metadata["conductor_component_id"] == "declared"
    assert bound.metadata["net_id"] == "declared-net"
    assert bound.metadata["equipotential_id"] == "declared-potential"
    assert bound.metadata["owner_semantic_ids"] == ("metal", "second")
    assert bound.metadata["source_provenance"]["conductor_source_layer_name"] == "metal"


def sheet_source(*, same_material_sides=False, with_hole=False):
    """Declare a collapsed sheet between nonoverlapping dielectric and vacuum domains."""
    source, _ = conductor_input()
    holes = (((1, 1), (3, 1), (3, 3), (1, 3)),) if with_hole else ()
    polygon = replace(source.polygons[0], holes=holes)
    metal = replace(
        source.entities[0],
        part_role="face_metal",
        route_representations={"A": "surface_sheet"},
        geometry={
            **source.entities[0].geometry,
            "z_max_um": 0,
            "thickness_um": 0,
            "hole_loops": holes,
        },
    )
    domain = source.entities[1]
    substrate = replace(
        domain,
        semantic_id="substrate",
        material_id="Si",
        material_kind="dielectric",
        priority=1,
        geometry={
            **domain.geometry,
            "z_min_um": -2,
            "z_max_um": 0,
            "z_um": -2,
            "thickness_um": 2,
            "domain_bounds_um": {**domain.geometry["domain_bounds_um"], "z_max_um": 0},
        },
    )
    vacuum = replace(
        domain,
        material_id="Si" if same_material_sides else domain.material_id,
        material_kind="dielectric" if same_material_sides else domain.material_kind,
        geometry={
            **domain.geometry,
            "z_min_um": 0,
            "z_um": 0,
            "thickness_um": 6,
            "domain_bounds_um": {**domain.geometry["domain_bounds_um"], "z_min_um": 0},
        },
    )
    context = RepresentationContext(
        {"metal": EntityRepresentation(None, "sheet", (0, 1), (0, 0))}
    )
    return replace(
        source, polygons=(polygon,), entities=(metal, substrate, vacuum)
    ), context


def contract_source(failure):
    """Declare ordered conflicts, retaining later conflicts to test failure priority."""
    source, _ = conductor_input()
    stages = ("net", "component", "equipotential", "layer", "provenance", "valid")
    stage = stages.index(failure)
    first = replace(
        source.entities[0],
        net_id="shared",
        metadata={"equipotential_id": "one", "source_layer_name": "metal"},
    )
    second = replace(
        first,
        semantic_id="second",
        net_id="other" if stage == 0 else "shared",
        metadata={
            "equipotential_id": "other" if stage <= 2 else "one",
            "source_layer_name": "other" if stage <= 3 else "metal",
        },
    )
    surface = SurfacePlanRecord(
        "shared",
        "metal",
        "interface",
        {},
        metadata={
            "owner_semantic_ids": ("metal", "second"),
            "source_provenance": () if stage <= 4 else {},
        },
    )
    ledger = (
        {
            "conductor_component_id": "declared",
            "members": ("metal", "second"),
            "net_id": "declared-net",
            "equipotential_id": "declared-potential",
        },
    )
    if stage <= 1:
        ledger = tuple(
            {**ledger[0], "conductor_component_id": owner, "members": (owner,)}
            for owner in ("metal", "second")
        )
    return (
        replace(source, entities=(first, second, source.entities[1])),
        surface,
        ledger,
    )
