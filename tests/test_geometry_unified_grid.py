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


def _mixed_bridge_operands(*, physical=False):
    # A valid polygon whose sub-grid neck becomes line support at the grid.
    bridge = (
        (0, 0),
        (2, 0),
        (2, 1.1),
        (4, 1.1),
        (4, 0),
        (6, 0),
        (6, 4),
        (4, 4),
        (4, 1.2),
        (2, 1.2),
        (2, 4),
        (0, 4),
    )
    return (
        GridOperand("left", rectangle(0, 0, 1, 4), contact=False),
        GridOperand("cutoff", bridge, physical=physical, contact=False),
        GridOperand("right", rectangle(5, 0, 6, 4), contact=False),
    )


def test_mixed_intermediate_chain_preserves_area_and_auxiliary_residue():
    from shapely import Polygon, intersection

    operands = _mixed_bridge_operands()
    grid = GeometryGridContext(1)
    source = Polygon(operands[1].exterior)
    assert source.is_valid
    mixed = intersection(source, source, grid_size=1)
    assert mixed.geom_type == "GeometryCollection"
    assert {p.geom_type for p in mixed.geoms} == {"Polygon", "LineString"}

    result = build_grid_arrangement(operands, grid)
    areas = {}
    for face in result.faces:
        areas[face.operand_ids] = areas.get(face.operand_ids, 0) + ring_area(
            face.exterior
        )
    assert areas == {("cutoff", "left"): 4, ("cutoff",): 8, ("cutoff", "right"): 4}
    assert set(result.edges) == {
        tuple(sorted((a, b)))
        for face in result.faces
        for ring in (face.exterior, *face.holes)
        for a, b in zip(ring, (*ring[1:], ring[0]))
    }
    assert set(result.vertices) == {
        p
        for face in result.faces
        for ring in (face.exterior, *face.holes)
        for p in ring
    }

    from scgsim.geometry._primitives.grid import _account_zero_area, _subdivide_grid

    shapes = {v.operand_id: Polygon(v.exterior, v.holes) for v in operands}
    subdivision = _subdivide_grid(operands, shapes, 1)
    accounting = _account_zero_area(subdivision.residues, subdivision.cells, 1)
    assert sum(r.geometry.length for r in accounting.auxiliary) == 2
    assert all(r.operand_ids == ("cutoff",) for r in accounting.auxiliary)
    assert {tuple(p) for r in accounting.auxiliary for p in r.geometry.coords} == {
        (2, 1),
        (4, 1),
    }


def test_zero_area_contact_sideband_nodes_faces_without_orphan_edges():
    from shapely import Polygon
    from scgsim.geometry._primitives.grid import _account_zero_area, _subdivide_grid

    operands = (
        GridOperand("A", rectangle(0, 0, 4, 4)),
        GridOperand("B", rectangle(4, 1, 6, 3)),
        GridOperand("C", rectangle(6, 3, 8, 5)),
    )
    result = build_grid_arrangement(operands, GeometryGridContext(1))
    assert {f.operand_ids: ring_area(f.exterior) for f in result.faces} == {
        ("A",): 16,
        ("B",): 4,
        ("C",): 4,
    }
    assert {(4, 1), (4, 3), (6, 3)} <= set(result.vertices)
    shapes = {v.operand_id: Polygon(v.exterior, v.holes) for v in operands}
    subdivision = _subdivide_grid(operands, shapes, 1)
    accounting = _account_zero_area(subdivision.residues, subdivision.cells, 1)
    assert not accounting.auxiliary
    assert any(
        r.geometry.geom_type == "LineString" and r.geometry.length == 2
        for r in accounting.boundary
    )
    assert any(
        r.geometry.geom_type == "Point" and tuple(r.geometry.coords[0]) == (6, 3)
        for r in accounting.boundary
    )
    assert set(result.edges) == {
        tuple(sorted((a, b)))
        for face in result.faces
        for ring in (face.exterior, *face.holes)
        for a, b in zip(ring, (*ring[1:], ring[0]))
    }


def test_mixed_residue_does_not_hide_physical_component_loss():
    with pytest.raises(ValueError, match="component/hole"):
        build_grid_arrangement(
            _mixed_bridge_operands(physical=True), GeometryGridContext(1)
        )


def test_overlay_native_error_retains_coordinate_free_boundary_context():
    from shapely.errors import GEOSException

    from shapely import Polygon
    from scgsim.geometry._primitives.grid import _overlay_call

    invalid = Polygon(((0, 0), (2, 2), (0, 2), (2, 0)))
    with pytest.raises(GEOSException) as caught:
        _overlay_call("intersection", invalid, invalid, 1, 0, None)
    import json

    note = next(
        note
        for note in caught.value.__notes__
        if note.startswith("scgsim.grid.overlay_failure: ")
    )
    context = json.loads(note.split(": ", 1)[1])
    assert context["operation"] == "intersection"
    assert context["left"] == {
        "geom_type": "Polygon",
        "component_dimensions": [2],
        "component_count": 1,
    }
    assert context["operand_index"] == 0
    assert context["cell_index"] is None
    assert context["grid_size_um"] == 1
    assert "source" not in context


def test_cut_subdivision_keeps_unselected_enclosing_parent_topology():
    from shapely import Polygon

    def rotate(ring):
        return tuple(
            (cos(0.37) * x - sin(0.37) * y, sin(0.37) * x + cos(0.37) * y)
            for x, y in ring
        )

    operands = (
        GridOperand("lower", rectangle(-2, -2, 4, 4), contact=False),
        GridOperand("upper", rectangle(-2, -2, 4, 4), contact=False),
        GridOperand(
            "cut_A", rotate(rectangle(0, 0, 2, 2)), physical=False, contact=False
        ),
        GridOperand(
            "cut_B", rotate(rectangle(1, -1, 3, 1)), physical=False, contact=False
        ),
    )
    result = build_grid_arrangement(operands, GRID)
    assert all({"lower", "upper"} <= set(face.operand_ids) for face in result.faces)
    for parent in ("lower", "upper"):
        regions = result.operand_regions(parent)
        assert len(regions) == 1
        assert not regions[0][1]
        assert Polygon(*regions[0]).area == 36


def test_realized_envelope_candidates_keep_snapped_touching_support():
    from shapely import Polygon
    from scgsim.geometry._primitives.grid import _account_zero_area, _subdivide_grid

    operands = (
        GridOperand("A", rectangle(0, 0, 1.4, 2), contact=False),
        GridOperand("B", rectangle(1.49, 0, 3, 2), contact=False),
    )
    result = build_grid_arrangement(operands, GeometryGridContext(1))
    assert {face.operand_ids: ring_area(face.exterior) for face in result.faces} == {
        ("A",): 2,
        ("B",): 4,
    }
    assert {(1, 0), (1, 2)} <= set(result.vertices)
    shapes = {value.operand_id: Polygon(value.exterior) for value in operands}
    subdivision = _subdivide_grid(operands, shapes, 1)
    accounting = _account_zero_area(subdivision.residues, subdivision.cells, 1)
    assert any(
        residue.geometry.length == 2 and residue.operand_ids == ("A", "B")
        for residue in accounting.boundary
    )


def test_area_minus_line_keeps_hole_and_all_contact_nodes():
    from shapely import LineString, Point, Polygon
    from scgsim.geometry._primitives.grid import (
        _OverlayParts,
        _parts_overlay,
        _regions,
    )

    area = Polygon(rectangle(0, 0, 8, 8), (rectangle(2, 2, 6, 6),))
    line = LineString(((-1, 4), (9, 4)))
    left, right = _OverlayParts((area,)), _OverlayParts((line,))
    remaining = _parts_overlay("difference", left, right, 1, 0, 0)
    assert remaining.geometries == (area,)
    assert remaining.geometries[0] is area
    contact = _parts_overlay("intersection", left, right, 1, 0, 0)
    nodes = tuple(
        tuple(point)
        for geometry in contact.geometries
        for part in (
            geometry.geoms if geometry.geom_type == "MultiLineString" else (geometry,)
        )
        for point in part.coords
    )
    outer, holes = _regions(area, nodes, 1)[0]
    assert {(0, 4), (8, 4)} <= set(outer)
    assert {(2, 4), (6, 4)} <= set(holes[0])
    assert Polygon(outer, holes).area == 48
    # A genuine line/line difference must still split at the exact crossing.
    split = _parts_overlay(
        "difference", right, _OverlayParts((LineString(((4, 0), (4, 8))),)), 1, 0, 0
    )
    assert len(split.geometries) == 1
    assert split.geometries[0].geom_type == "MultiLineString"
    assert len(split.geometries[0].geoms) == 2
    assert (
        _parts_overlay("intersection", right, _OverlayParts((Point(4, 4),)), 1, 0, 0)
        .geometries[0]
        .equals(Point(4, 4))
    )


def test_zero_area_dedup_keeps_labels_and_retained_collinear_vertices():
    from shapely import LineString
    from scgsim.geometry._primitives.grid import _ZeroAreaResidue, _deduplicate_residues

    line = LineString(((0, 0), (1, 0), (2, 0)))
    first = _ZeroAreaResidue(line, ("A",))
    duplicate = _ZeroAreaResidue(LineString(tuple(reversed(line.coords))), ("A",))
    other_label = _ZeroAreaResidue(line, ("B",))
    other_nodes = _ZeroAreaResidue(LineString(((0, 0), (2, 0))), ("A",))
    result = _deduplicate_residues((first, duplicate, other_label, other_nodes))
    assert result == (first, other_label, other_nodes)
    assert result[0] is first
    assert tuple(result[0].geometry.coords) == ((0, 0), (1, 0), (2, 0))


def test_topology_failure_note_keeps_requested_and_realized_counts():
    import json

    operand = GridOperand("thin", rectangle(0, 0, 0.25, 4), contact=False)
    with pytest.raises(ValueError, match="physical geometry collapsed") as caught:
        build_grid_arrangement((operand,), GeometryGridContext(1))
    note = next(
        note
        for note in caught.value.__notes__
        if note.startswith("scgsim.grid.physical_topology: ")
    )
    metrics = json.loads(note.split(": ", 1)[1])
    assert metrics["requested"] == {
        "empty": False,
        "is_valid": True,
        "components": 1,
        "holes": 0,
        "area_um2": 1,
    }
    assert metrics["realized"] == {
        "empty": True,
        "is_valid": True,
        "components": 0,
        "holes": 0,
        "area_um2": 0,
    }
    # Self-realization preserves a nonempty zero-area line, while the labelled
    # positive-area cell inventory is empty. Neither is accepted as physical area.
    for stage, empty in (
        ("initial_grid_realization", False),
        ("labelled_cells_without_resnap", True),
        ("reconstructed_faces_without_resnap", True),
    ):
        assert metrics[stage] == {
            "empty": empty,
            "is_valid": True,
            "components": 0,
            "holes": 0,
            "area_um2": 0,
        }
        assert set(metrics[stage]) == set(metrics["requested"])
    assert metrics["operand_index"] == 0
    assert metrics["grid_size_um"] == 1
    assert "thin" not in note


def test_parallel_grid_is_identical_for_mixed_sidebands_and_contact_nodes():
    from scgsim.preparation import preparation_resources

    operands = _mixed_bridge_operands()
    serial = build_grid_arrangement(operands, GeometryGridContext(1))
    with preparation_resources(max_workers=2):
        parallel = build_grid_arrangement(operands, GeometryGridContext(1))
    assert parallel == serial
    contacts = (
        GridOperand("A", rectangle(0, 0, 4, 4)),
        GridOperand("B", rectangle(4, 1, 6, 3)),
        GridOperand("C", rectangle(6, 3, 8, 5)),
    )
    serial = build_grid_arrangement(contacts, GeometryGridContext(1))
    with preparation_resources(max_workers=2):
        parallel = build_grid_arrangement(contacts, GeometryGridContext(1))
    assert parallel == serial


def test_crossing_cut_partition_conserves_parent_at_each_union_stage():
    from shapely import Polygon, intersection, union_all
    from scgsim.geometry._primitives.grid import _subdivide_grid

    # Public coarse lattice makes snapping visible; no study geometry/values.
    context = GeometryGridContext(1)
    operands = (
        GridOperand("parent", rectangle(0, 0, 10, 10), contact=False),
        GridOperand("cutA", ((1, 1), (9, 2), (3, 9)), physical=False, contact=False),
        GridOperand("cutB", ((1, 8), (8, 1), (9, 9)), physical=False, contact=False),
        GridOperand("remote", rectangle(20, 20, 21, 21), contact=False),
    )
    shapes = {
        value.operand_id: Polygon(value.exterior, value.holes) for value in operands
    }
    subdivision = _subdivide_grid(operands, shapes, 1)
    selected = [cell for cell, labels in subdivision.cells if "parent" in labels]
    raw = shapes["parent"]
    self_realized = intersection(raw, raw, grid_size=1)
    unsnapped_cell_union = union_all(selected)
    effective_union = union_all(selected, grid_size=1)
    arrangement = build_grid_arrangement(operands, context)
    reconstructed = union_all(
        [
            Polygon(outer, holes)
            for outer, holes in arrangement.operand_regions("parent")
        ]
    )
    for stage in (self_realized, unsnapped_cell_union, effective_union, reconstructed):
        assert stage.equals(raw)
        assert stage.area == 100
        assert stage.geom_type == "Polygon"
        assert stage.is_valid
        assert len(stage.interiors) == 0
    # This is a direct area-partition witness before another fixed-precision
    # operation can change a cut boundary. Every cell retains its parent label.
    assert sum(cell.area for cell in selected) == 100
    for index, cell in enumerate(selected):
        for other in selected[index + 1 :]:
            assert intersection(cell, other).area == 0


@pytest.mark.parametrize("workers", (1, 2))
def test_source_recipe_map_isolates_context_and_merges_audits_in_source_order(workers):
    from scgsim.preparation import preparation_resources
    from scgsim.geometry._primitives.grid import (
        _SOURCE_AUDITS,
        _SOURCE_INTENT,
        _frame_point,
        _source_recipe_map,
        current_grid_context,
        geometry_grid_scope,
        polygon_world_frame,
        source_grid_audits,
    )

    with geometry_grid_scope(GRID), polygon_world_frame((1, 0, 0, 1, 3, 4)):
        parent = _SOURCE_AUDITS.get()
        parent.append({"seed": 0})
        intent = _SOURCE_INTENT.set(True)
        try:

            def recipe(index):
                local = _SOURCE_AUDITS.get()
                assert local is not parent
                assert local == []
                assert current_grid_context() == GRID
                assert _SOURCE_INTENT.get() is True
                assert _frame_point((0, 0)) == (3, 4)
                local.extend(({"seed": 0}, {"recipe": index}))
                assert parent == [{"seed": 0}]
                return index * 2

            with preparation_resources(max_workers=workers):
                assert _source_recipe_map(recipe, (2, 0, 1)) == [4, 0, 2]
            assert source_grid_audits() == (
                {"seed": 0},
                {"recipe": 2},
                {"recipe": 0},
                {"recipe": 1},
            )
            assert _SOURCE_AUDITS.get() is parent
        finally:
            _SOURCE_INTENT.reset(intent)
    assert _SOURCE_AUDITS.get() is None

    def disabled(index):
        assert _SOURCE_AUDITS.get() is None
        return index

    with preparation_resources(max_workers=workers):
        assert _source_recipe_map(disabled, (1, 2)) == [1, 2]
    assert _SOURCE_AUDITS.get() is None


@pytest.mark.parametrize("workers", (1, 2))
def test_source_recipe_map_preserves_first_error_and_audit_prefix_then_cleans_up(
    workers,
):
    from scgsim.preparation import preparation_resources
    from scgsim.geometry._primitives.grid import (
        _SOURCE_AUDITS,
        _source_recipe_map,
        geometry_grid_scope,
        source_grid_audits,
    )

    first, later = ValueError("first recipe"), ValueError("later recipe")
    with geometry_grid_scope(GRID):
        parent = _SOURCE_AUDITS.get()

        def recipe(index):
            _SOURCE_AUDITS.get().append({"recipe": index})
            assert parent == []
            if index == 1:
                raise first
            if index == 2:
                raise later
            return index

        with preparation_resources(max_workers=workers):
            with pytest.raises(ValueError) as caught:
                _source_recipe_map(recipe, (0, 1, 2))
            assert caught.value is first
            assert source_grid_audits() == ({"recipe": 0}, {"recipe": 1})
            assert _SOURCE_AUDITS.get() is parent

            def clean(index):
                assert _SOURCE_AUDITS.get() == []
                return index

            assert _source_recipe_map(clean, (3, 4)) == [3, 4]
            assert source_grid_audits() == ({"recipe": 0}, {"recipe": 1})
    assert _SOURCE_AUDITS.get() is None


def test_realized_grid_nodes_use_one_coordinate_per_integer_key():
    from shapely import Polygon
    from scgsim.geometry._primitives.grid import _regions, _region_key

    grid = 0.1
    endpoint = (3 * grid, 3 * grid)
    same_key = (3 / 10, 3 / 10)
    assert endpoint != same_key
    assert _region_key(endpoint, grid) == _region_key(same_key, grid)
    triangle = Polygon(((0, 0), endpoint, (0, 1)))
    realized = _regions(triangle, (same_key, endpoint), grid)[0][0]
    keys = tuple(_region_key(point, grid) for point in realized)
    assert len(keys) == len(set(keys)) == 3
    assert realized == ((0, 0), same_key, (0, 1))
    assert Polygon(realized).equals(Polygon(((0, 0), same_key, (0, 1))))
    assert tuple(triangle.exterior.coords)[1] == endpoint
    assert _regions(Polygon(realized), (same_key, endpoint), grid)[0][0] == realized
    # The same distinction remains real in an ungridded parent plane; its
    # original coordinates are not rounded by this layout-node identity rule.
    untouched = _regions(triangle, (), None)[0][0]
    assert untouched == ((0, 0), endpoint, (0, 1))
    separate_triangle = Polygon(((0, 0), same_key, (0, 1)))
    assert _regions(separate_triangle, (), None)[0][0][1] == same_key


def test_horizontal_node_plane_keeps_raw_subgrid_physical_guard():
    from scgsim.geometry.planning.surface_bands import _node_band_planes

    build, parent = band_fixture()
    bound = apply_geometry_grid(build, GRID)
    children, _, summaries = partition_surface_distance_bands(bound, surfaces=(parent,))
    thin = replace(
        parent,
        surface_id="thin_neighbor",
        geometry_ref={**parent.geometry_ref, "outer_loop": rectangle(0, 0, 0.0002, 1)},
    )
    # Genuine generated band children activate common horizontal noding, while
    # the unbanded source operand must retain its positive-area raw baseline.
    with pytest.raises(ValueError, match="physical geometry collapsed.*thin_neighbor"):
        _node_band_planes((thin, *children), summaries, GRID)
    assert thin.geometry_ref["outer_loop"] == rectangle(0, 0, 0.0002, 1)


def test_horizontal_node_plane_emits_canonical_xy_without_reverting_raw_spelling():
    from scgsim.geometry._primitives.grid import _canonical_grid_point
    from scgsim.geometry.planning.surface_bands import _node_band_planes

    endpoint = 571 * GRID.grid_size_um
    assert endpoint != 0.571
    build, parent = band_fixture(outer=rectangle(0, 0, endpoint, 1), cuts=(0.1,))
    bound = apply_geometry_grid(build, GRID)
    children, _, summaries = partition_surface_distance_bands(bound, surfaces=(parent,))
    emitted, _, _ = _node_band_planes((parent, *children), summaries, GRID)
    points = [
        point
        for surface in emitted
        for ring in surface.geometry_ref["planar_loops_3d"]
        for point in ring
    ]
    assert points
    assert any(point[0] == 0.571 for point in points)
    assert all(
        point[:2] == _canonical_grid_point(point[:2], GRID.grid_size_um)
        for point in points
    )
    assert {point[2] for point in points} == {0}
    assert parent.geometry_ref["outer_loop"][1][0] == endpoint


def test_candidate_pairs_parallelize_real_work_with_identical_subdivision_and_graph():
    import threading
    from shapely import Polygon
    from scgsim.preparation import preparation_resources
    from scgsim.geometry._primitives.grid import _cell_overlay_chunk, _subdivide_grid

    values = _mixed_bridge_operands()
    shapes = {
        value.operand_id: Polygon(value.exterior, value.holes) for value in values
    }
    serial = _subdivide_grid(values, shapes, 1)
    serial_graph = build_grid_arrangement(values, GeometryGridContext(1))
    main_thread = threading.get_ident()
    seen = set()
    lock = threading.Lock()
    old_profile = threading.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _cell_overlay_chunk.__code__:
            with lock:
                seen.add(threading.get_ident())

    threading.setprofile(observe)
    try:
        with preparation_resources(max_workers=2):
            parallel = _subdivide_grid(values, shapes, 1)
            parallel_graph = build_grid_arrangement(values, GeometryGridContext(1))
    finally:
        threading.setprofile(old_profile)
    assert seen
    assert main_thread not in seen
    assert [(cell.wkb, labels) for cell, labels in parallel.cells] == [
        (cell.wkb, labels) for cell, labels in serial.cells
    ]
    assert [residue.exact_key for residue in parallel.residues] == [
        residue.exact_key for residue in serial.residues
    ]
    assert parallel_graph == serial_graph


def test_candidate_pair_chunks_retain_real_geos_failures_at_original_ordinals():
    from shapely import Polygon
    from shapely.errors import GEOSException
    from scgsim.geometry._primitives.grid import _OverlayParts, _cell_overlay_chunk
    from scgsim.preparation import _ordered_map, preparation_resources

    invalid = Polygon(((0, 0), (2, 2), (0, 2), (2, 0)))
    valid = Polygon(rectangle(0, 0, 4, 4))
    operand = _OverlayParts((valid,))
    work = (
        (((3, invalid), (4, valid)), operand, 1, 7),
        (((5, invalid),), operand, 1, 7),
    )
    serial = [_cell_overlay_chunk(chunk) for chunk in work]
    with preparation_resources(max_workers=2):
        parallel = _ordered_map(_cell_overlay_chunk, work)
    for expected_chunk, actual_chunk in zip(serial, parallel, strict=True):
        assert [index for index, _ in actual_chunk] == [
            index for index, _ in expected_chunk
        ]
        for (index, expected), (_, actual) in zip(
            expected_chunk, actual_chunk, strict=True
        ):
            if index == 4:
                assert actual.error is None
                assert actual.common == expected.common
            else:
                assert isinstance(actual.error, GEOSException)
                assert type(actual.error) is type(expected.error)
                assert actual.error.__notes__ == expected.error.__notes__
                note = json.loads(actual.error.__notes__[0].split(": ", 1)[1])
                assert note["cell_index"] == index
                assert note["operand_index"] == 7
                assert note["operation"] == "intersection"
                assert actual.traceback is actual.error.__traceback__
