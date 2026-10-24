"""Shared polygon-grid contracts with real generic geometry, without native meshing."""

from dataclasses import replace
from decimal import Decimal
import json
from math import cos, sin

import pytest

from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement
from scgsim.geometry.models.grid import GeometryGridContext, geometry_grid_context
from scgsim.geometry.source.grid import apply_geometry_grid, source_grid_context
from scgsim.geometry.planning.junction_partition import derive_rectangular_junctions
from scgsim.geometry.planning.surface_bands import partition_surface_distance_bands
from scgsim.geometry.planning.topology import plan_canonical_topology
from scgsim.geometry import with_rectangular_junctions
from test_rectangular_junction_geometry import fixture as junction_fixture
from test_surface_distance_bands import fixture as band_fixture


GRID = GeometryGridContext(0.001)


def rectangle(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def assert_lattice(points):
    for point in points:
        for value in point[:2]:
            scaled = Decimal(str(value)) / Decimal("0.001")
            assert scaled == scaled.to_integral_value()


def ring_area(ring):
    return (
        abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(ring, (*ring[1:], ring[0]))))
        / 2
    )


@pytest.mark.parametrize("requested", (None, 0.001))
def test_context_codec_and_source_propagation(requested):
    context = source_grid_context(0.001, requested)
    assert (
        GeometryGridContext.from_payload(json.loads(json.dumps(context.to_payload())))
        == GRID
    )
    build, _ = band_fixture()
    assert geometry_grid_context(build) is None
    bound = apply_geometry_grid(build, context)
    assert geometry_grid_context(bound) == GRID
    assert "geometry_grid" not in build.metadata
    assert bound.polygons[0].polygon_id == build.polygons[0].polygon_id
    assert bound.entities == build.entities
    assert apply_geometry_grid(bound, context).polygons == bound.polygons
    with pytest.raises(ValueError, match="source_dbu_um"):
        source_grid_context(0.005, 0.001)


@pytest.mark.parametrize("angle", (0, 0.37))
def test_crossing_membership_shared_edges_and_idempotence(angle):
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
            for a, b in zip(ring, (*ring[1:], ring[0])):
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
    with pytest.raises(ValueError, match=reason):
        build_grid_arrangement(operands, GRID)


def test_junction_keeps_declared_terminal_identity_and_central_rectangle():
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


@pytest.mark.parametrize("neighboring", (False, True))
def test_band_context_mismatch_and_shared_atomic_topology(neighboring):
    from scgsim.geometry import (
        SourceBoundarySegment,
        SurfaceDistanceBandSpec,
        with_surface_distance_bands,
    )
    from scgsim.geometry.planning.surface_bands import partition_source_boundary

    build, parent = band_fixture()
    parents = (parent,)
    if neighboring:
        origin = partition_source_boundary(parent.geometry_ref, parent.surface_id)
        parents = tuple(
            replace(
                parent,
                surface_id=name,
                parent_surface_id=parent.surface_id,
                partition_label=name,
                geometry_ref={
                    **parent.geometry_ref,
                    "outer_loop": rectangle(x0, 0, x1, 8),
                },
                metadata={"surface_partition_origin": origin},
            )
            for name, x0, x1 in (("left", 0, 4), ("right", 4, 8))
        )
        build = with_surface_distance_bands(
            build,
            tuple(
                SurfaceDistanceBandSpec(
                    name,
                    "entity",
                    "polygon",
                    (cut,),
                    0.001,
                    (SourceBoundarySegment("exterior", 0),),
                )
                for name, cut in (("left", 0.5), ("right", 1.0))
            ),
        )
    bound = apply_geometry_grid(build, GRID)
    children, _, summary = partition_surface_distance_bands(bound, surfaces=parents)
    if neighboring:
        edge_nodes = {}
        for name in ("left", "right"):
            edge_nodes[name] = {
                y
                for child in children
                if child.parent_surface_id == name
                for ring in child.geometry_ref["planar_loops_3d"]
                for x, y, z in ring
                if x == 4
            }
        assert edge_nodes["left"] == edge_nodes["right"] == {0, 0.5, 1.0, 8}
    points, curves, loops, canonical = plan_canonical_topology(surfaces=children)
    assert points and curves and loops
    assert {child.parent_surface_id for child in canonical} == {
        p.surface_id for p in parents
    }
    assert all(row["precision_um"] == GRID.grid_size_um for row in summary)
    assert_lattice(p.coordinate for p in points)
    bad = replace(
        bound,
        metadata={
            **bound.metadata,
            "geometry_grid": GeometryGridContext(0.002).to_payload(),
        },
    )
    with pytest.raises(ValueError, match="precision_um"):
        partition_surface_distance_bands(bad, surfaces=parents)


@pytest.mark.parametrize("requested", (None, 0.001))
def test_public_gds_adapter_derives_and_checks_emitted_dbu(tmp_path, requested):
    import gdstk
    from scgsim.geometry import build_gds_stack_geometry_input

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
    result = build_gds_stack_geometry_input(
        gds_file=gds, stack_file=stack, grid_size_um=requested
    )
    assert result.polygons
    assert result.metadata["source_dbu_um"] == 0.001
    assert geometry_grid_context(result) == GRID
    assert_lattice(p for polygon in result.polygons for p in polygon.exterior)
    with pytest.raises(ValueError, match="source_dbu_um"):
        build_gds_stack_geometry_input(
            gds_file=gds, stack_file=stack, grid_size_um=0.002
        )


@pytest.mark.parametrize("other_y,changed", ((2.01, False), (2.0002, True)))
def test_source_grid_detects_new_port_contact_to_unselected_host(other_y, changed):
    build, spec = junction_fixture()
    other = replace(
        build.polygons[0], polygon_id="other", exterior=rectangle(4, other_y, 5, 3)
    )
    owner = replace(
        build.entities[0],
        semantic_id="chip/other",
        net_id="C",
        polygon_ids=(other.polygon_id,),
    )
    build = replace(
        build,
        polygons=(*build.polygons, other),
        entities=(*build.entities, owner),
        metadata={**build.metadata, "source_dbu_um": 0.001},
    )
    authored = with_rectangular_junctions(build, (spec,))
    assert {
        row.host_semantic_id for row in authored.port_sheet_regions[0].overlaps
    } == {"chip/left", "chip/right"}
    if changed:
        with pytest.raises(ValueError, match="contact"):
            apply_geometry_grid(authored, GRID)
    else:
        result = apply_geometry_grid(authored, GRID)
        assert {
            row.host_semantic_id for row in result.port_sheet_regions[0].overlaps
        } == {"chip/left", "chip/right"}


@pytest.mark.parametrize("mask_width,collapses", ((0.1, False), (0.0002, True)))
def test_declared_ground_hole_is_audited_before_quantized_source_output(
    mask_width, collapses
):
    import gdstk
    from scgsim.geometry._primitives.grid import geometry_grid_scope
    from scgsim.geometry.source.ground import _derived_ground_polygons

    build, _ = junction_fixture()
    ground = replace(
        build.entities[0],
        semantic_id="chip/ground",
        net_id="Ground",
        geometry={"mask_layer": (1, 0)},
    )
    arguments = dict(
        polygons_by_layer={(1, 0): (gdstk.rectangle((1, 1), (1 + mask_width, 2)),)},
        cell_bounds_um={"x_min_um": 0, "x_max_um": 4, "y_min_um": 0, "y_max_um": 4},
        domain_bounds_by_semantic_id={},
        plane_polygons=(gdstk.rectangle((0, 0), (4, 4)),),
    )
    with geometry_grid_scope(GRID):
        if collapses:
            with pytest.raises(
                ValueError, match="declared physical source component/hole topology"
            ):
                _derived_ground_polygons(ground, **arguments)
        else:
            result = _derived_ground_polygons(ground, **arguments)
            assert len(result) == 1 and len(result[0].holes) == 1


@pytest.mark.parametrize("right_x,changed", ((1.01, False), (1.0002, True)))
def test_public_ground_recipe_retains_pre_lowering_contact_identity(
    tmp_path, right_x, changed
):
    import gdstk
    from scgsim.geometry import build_gds_stack_geometry_input

    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("GROUND_CONTACT")
    cell.add(gdstk.rectangle((0, 0), (2, 1), layer=99))
    gds = tmp_path / "source.gds"
    library.write_gds(gds)
    layers, domains = [], {}
    for name, x0, x1 in (("left", 0, 1.0001), ("right", right_x, 2)):
        domain = name + "_plane"
        domains[domain] = {
            "material_id": "vacuum",
            "geometry": {
                "domain_bounds_um": {
                    "x_min_um": x0,
                    "x_max_um": x1,
                    "y_min_um": 0,
                    "y_max_um": 1,
                }
            },
        }
        layers.append(
            {
                "semantic_id": name,
                "layer": 1,
                "datatype": 0,
                "role": "conductor",
                "material_id": "metal",
                "net_id": name,
                "geometry": {
                    "geometry_source": "die_face_minus_ground_mask",
                    "mask_layer": [1, 0],
                    "plane_bounds_ref": domain,
                    "z_um": 0,
                    "thickness_um": 0,
                },
            }
        )
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps(
            {
                "layers": layers,
                "solution_regions": domains,
                "materials": {
                    "metal": {"kind": "conductor"},
                    "vacuum": {"kind": "vacuum"},
                },
            }
        )
    )
    if changed:
        with pytest.raises(ValueError, match="contact"):
            build_gds_stack_geometry_input(
                gds_file=gds, stack_file=stack, grid_size_um=0.001
            )
    else:
        result = build_gds_stack_geometry_input(
            gds_file=gds, stack_file=stack, grid_size_um=0.001
        )
        audits = result.metadata["geometry_grid_source_audits"]
        assert {row["source_entity_id"] for row in audits} == {"left", "right"}
        assert all(row["requested_polygons"] for row in audits)
        assert geometry_grid_context(result) == GRID
