"""Preserve ordered sheet contributions and final surface ownership decisions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.surface_planning.contracts import (
    with_surface_contract_metadata,
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
