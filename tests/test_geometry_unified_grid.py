"""Shared polygon-grid contracts with real generic geometry, without native meshing."""

import json
from dataclasses import replace
from math import cos, sin

import pytest
from geometry_grid_fixtures import GRID, assert_lattice, rectangle, ring_area
from test_rectangular_junction_geometry import fixture as junction_fixture

from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    with_rectangular_junctions,
)
from scgsim.geometry.models.grid import GeometryGridContext, geometry_grid_context
from scgsim.geometry.planning.junction_partition import derive_rectangular_junctions
from scgsim.geometry.primitives.grid import GridOperand, build_grid_arrangement
from scgsim.geometry.source.grid import apply_geometry_grid, source_grid_context


def test_context_codec_and_source_propagation():
    """Verify that context codec and source propagation."""
    context = source_grid_context(0.001)
    assert (
        GeometryGridContext.from_payload(json.loads(json.dumps(context.to_payload())))
        == GRID
    )
    polygon = LayoutPolygonSpec("polygon", "layer", rectangle(0, 0, 8, 8))
    entity = SemanticEntitySpec(
        "entity",
        "substrate",
        "Si",
        "dielectric",
        0,
        "polygon",
        polygon_ids=("polygon",),
    )
    build = GeometryBuildInput((polygon,), (entity,))
    assert geometry_grid_context(build) is None
    bound = apply_geometry_grid(build, context)
    assert geometry_grid_context(bound) == GRID
    assert "geometry_grid" not in build.metadata
    assert bound.polygons[0].polygon_id == build.polygons[0].polygon_id
    assert bound.entities == build.entities
    assert apply_geometry_grid(bound, context).polygons == bound.polygons


@pytest.mark.parametrize("angle", (0, 0.37))
def test_crossing_membership_shared_edges_and_idempotence(angle):
    """Verify that crossing membership shared edges and idempotence."""

    def rotated(ring):
        return tuple(
            (cos(angle) * x - sin(angle) * y, sin(angle) * x + cos(angle) * y)
            for x, y in ring
        )

    operands = (
        GridOperand("A", rotated(rectangle(0, 0, 2, 2))),
        GridOperand("B", rotated(rectangle(1, -1, 3, 1))),
    )
    result = build_grid_arrangement(operands, GRID)
    assert {face.operand_ids for face in result.faces} == {("A",), ("B",), ("A", "B")}
    assert_lattice(result.vertices)
    edge_faces = {}
    for face in result.faces:
        for ring in (face.exterior, *face.holes):
            for a, b in zip(ring, (*ring[1:], ring[0]), strict=False):
                edge_faces.setdefault(tuple(sorted((a, b))), set()).add(face.face_id)
    assert any(len(owners) == 2 for owners in edge_faces.values())
    assert set(edge_faces) == set(result.edges)
    rebuilt = build_grid_arrangement(
        tuple(
            GridOperand(name, outer, holes)
            for name in ("A", "B")
            for outer, holes in result.operand_regions(name)
        ),
        GRID,
    )
    assert rebuilt.faces == result.faces
    assert rebuilt.vertices == result.vertices
    assert rebuilt.edges == result.edges
    if angle == 0:
        assert {
            face.operand_ids: ring_area(face.exterior) for face in result.faces
        } == {
            ("A",): 3,
            ("B",): 3,
            ("A", "B"): 1,
        }


def test_hole_and_disconnected_operands_keep_identity():
    """Verify that hole and disconnected operands keep identity."""
    operands = (
        GridOperand("holed", rectangle(0, 0, 4, 4), (rectangle(1, 1, 3, 3),)),
        GridOperand("island", rectangle(6, 0, 7, 1)),
    )
    result = build_grid_arrangement(operands, GRID)
    assert len(result.operand_regions("holed")) == 1
    outer, holes = result.operand_regions("holed")[0]
    assert ring_area(outer) - sum(map(ring_area, holes)) == 12
    assert len(holes) == 1
    assert len(result.operand_regions("island")) == 1
    assert {a.operand_id for a in result.audits} == {"holed", "island"}


@pytest.mark.parametrize(
    "operands,reason",
    (
        ((GridOperand("thin", rectangle(0, 0, 0.0002, 1)),), "collapsed"),
        (
            (
                GridOperand(
                    "hole", rectangle(0, 0, 4, 4), (rectangle(1, 1, 1.0002, 2),)
                ),
            ),
            "hole",
        ),
        (
            (
                GridOperand("A", rectangle(0, 0, 1.0001, 1)),
                GridOperand("B", rectangle(1.0002, 0, 2, 1)),
            ),
            "contact changed",
        ),
        (
            (
                GridOperand("A", rectangle(0, 0, 1.0002, 1)),
                GridOperand("B", rectangle(1, 0, 2, 1)),
            ),
            "contact",
        ),
    ),
)
def test_physical_collapse_or_changed_contact_is_explicit(operands, reason):
    """Verify that physical collapse or changed contact is explicit."""
    with pytest.raises(ValueError, match=reason):
        build_grid_arrangement(operands, GRID)


def test_junction_keeps_declared_terminal_identity_and_central_rectangle():
    """Verify that junction keeps declared terminal identity and central rectangle."""
    build, spec = junction_fixture()
    build = replace(build, metadata={**build.metadata, "source_dbu_um": 0.001})
    bound = apply_geometry_grid(build, GRID)
    partitioned = derive_rectangular_junctions(
        with_rectangular_junctions(bound, (spec,))
    )
    record = partitioned.metadata["junction_partitions"][spec.junction_id]
    central = next(
        p for p in partitioned.polygons if p.polygon_id == record["central_polygon_id"]
    )
    assert {(3, 0), (7, 0), (7, 2), (3, 2)} <= set(central.exterior)
    assert all(
        (x in (3, 7) and 0 <= y <= 2) or (y in (0, 2) and 3 <= x <= 7)
        for x, y in central.exterior
    )
    assert ring_area(central.exterior) == 8
    assert central.holes == ()
    assert_lattice(central.exterior)
    assert [end["source_entity_id"] for end in record["ends"]] == [
        "chip/left",
        "chip/right",
    ]
    assert geometry_grid_context(partitioned) == GRID


def test_public_gds_adapter_takes_the_grid_from_the_emitted_dbu(tmp_path):
    """The adapter's grid is the database unit the GDS is written in."""
    import gdstk  # noqa: PLC0415 -- optional GDS dependency is needed only by this adapter test

    from scgsim.geometry import build_gds_stack_geometry_input  # noqa: PLC0415 -- keep this test-specific dependency scoped to its contract test

    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("GRID_COUPON")
    cell.add(gdstk.rectangle((0, 0), (4, 4), layer=1))
    gds = tmp_path / "source.gds"
    library.write_gds(gds)
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "semantic_id": "substrate",
                        "layer": 1,
                        "datatype": 0,
                        "role": "substrate",
                        "material_id": "Si",
                        "z_um": -1,
                        "thickness_um": 1,
                    }
                ],
                "materials": {"Si": {"kind": "dielectric", "permittivity": 11.9}},
                "solution_regions": {},
            }
        )
    )
    result = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
    assert result.polygons
    assert result.metadata["source_dbu_um"] == 0.001
    assert geometry_grid_context(result) == GRID
    assert_lattice(p for polygon in result.polygons for p in polygon.exterior)
