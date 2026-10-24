"""Actual polygon partitions and shared topology; no solver or private cases."""

from dataclasses import replace

import gdstk
import pytest

from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceBoundarySegment,
    SurfaceDistanceBandSpec,
    with_surface_distance_bands,
)
from scgsim.geometry.models.topology import SurfacePlanRecord
from scgsim.geometry.planning.surface_bands import partition_surface_distance_bands
from scgsim.geometry.planning.topology import plan_canonical_topology


SQUARE = ((0, 0), (8, 0), (8, 8), (0, 8))


def fixture(outer=SQUARE, holes=(), points3d=None, cuts=(1, 2), subset=None):
    polygon = LayoutPolygonSpec("polygon", "layer", outer, holes)
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
    ref = {
        "outer_loop": outer,
        "hole_loops": holes,
        "z_um": 0,
        "source_polygon_ids": ("polygon",),
    }
    if points3d is not None:
        ref = {"quad_points": points3d, "source_polygon_ids": ("polygon",)}
    surface = SurfacePlanRecord(
        "parent", "entity", "domain_boundary", ref, interface_id="SA__entity__vacuum"
    )
    spec = SurfaceDistanceBandSpec("parent", "entity", "polygon", cuts, 0.001, subset)
    return with_surface_distance_bands(build, (spec,)), surface


def regions(children):
    return [
        gdstk.Polygon([(x, y) for x, y, _ in s.geometry_ref["planar_loops_3d"][0]])
        for s in children
    ]


def area(children):
    from fractions import Fraction

    def ring_area(ring):
        points = tuple((Fraction(str(x)), Fraction(str(y))) for x, y, _ in ring)
        return (
            abs(
                sum(
                    a[0] * b[1] - a[1] * b[0]
                    for a, b in zip(points, (*points[1:], points[0]))
                )
            )
            / 2
        )

    return sum(
        ring_area(s.geometry_ref["planar_loops_3d"][0])
        - sum(ring_area(r) for r in s.geometry_ref["planar_loops_3d"][1:])
        for s in children
    )


def test_square_bands_cover_parent_and_emit_real_topology():
    build, parent = fixture()
    children, partitions, summary = partition_surface_distance_bands(
        build, surfaces=(parent,)
    )
    assert [
        area(
            [
                s
                for s in children
                if s.metadata["surface_distance_band"]["band_index"] == i
            ]
        )
        for i in range(3)
    ] == [28, 20, 16]
    assert area(children) == 64
    assert len(partitions) == len(children)
    points, curves, loops, canonical = plan_canonical_topology(surfaces=children)
    assert points and curves and loops
    assert all(s.outer_loop_ref for s in canonical)
    assert len(summary) == 3
    assert parent.geometry_ref["outer_loop"] == SQUARE


def test_subset_uses_original_parent_edge_only():
    build, parent = fixture(subset=(SourceBoundarySegment("exterior", 0),))
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))
    assert [
        area(
            [
                s
                for s in children
                if s.metadata["surface_distance_band"]["band_index"] == i
            ]
        )
        for i in range(3)
    ] == [8, 8, 48]


def test_holes_empty_core_and_concavity():
    hole = ((2, 2), (6, 2), (6, 6), (2, 6))
    build, parent = fixture(holes=(hole,), cuts=(1, 10))
    children, _, summary = partition_surface_distance_bands(build, surfaces=(parent,))
    assert area(children) == 48
    assert summary[-1]["child_surface_ids"] == []
    outer = ((0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6))
    build, parent = fixture(outer=outer, cuts=(0.5, 2))
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))
    assert area(children) == 20


def test_disconnected_core_and_hole_subset():
    outer = (
        (0, 0),
        (4, 0),
        (4, 1),
        (8, 1),
        (8, 0),
        (12, 0),
        (12, 4),
        (8, 4),
        (8, 3),
        (4, 3),
        (4, 4),
        (0, 4),
    )
    build, parent = fixture(outer=outer, cuts=(1,))
    _, _, summary = partition_surface_distance_bands(build, surfaces=(parent,))
    assert summary[-1]["component_count"] == 2
    build, parent = fixture(
        holes=(((2, 2), (6, 2), (6, 6), (2, 6)),),
        subset=(SourceBoundarySegment("hole", 0, 0),),
    )
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))
    assert area(children) == 48


def test_rotated_wall_preserves_original_vertices_and_shared_topology():
    wall = ((0, 0, 0), (8, 8, 0), (8, 8, 8), (0, 0, 8))
    build, parent = fixture(points3d=wall)
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))
    actual = {
        p for s in children for ring in s.geometry_ref["planar_loops_3d"] for p in ring
    }
    assert set(wall) <= actual
    assert all(x == y for x, y, _ in actual)
    assert plan_canonical_topology(surfaces=children)[0]


def test_reject_unknown_origin_curve_and_repartitioned_child():
    build, parent = fixture()
    bad = replace(
        build,
        metadata={
            "surface_distance_bands": [
                SurfaceDistanceBandSpec(
                    "missing", "entity", "polygon", (1,), 0.001
                ).to_payload()
            ]
        },
    )
    with pytest.raises(ValueError, match="unknown"):
        partition_surface_distance_bands(bad, surfaces=(parent,))
    child = replace(parent, metadata={"surface_distance_band": {}})
    with pytest.raises(ValueError, match="children"):
        partition_surface_distance_bands(build, surfaces=(child,))
    with pytest.raises(ValueError):
        SurfaceDistanceBandSpec("parent", "entity", "polygon", (2, 1), 0.001)


def test_nonplanar_parent_is_explicitly_unsupported():
    build, parent = fixture(points3d=((0, 0, 0), (8, 0, 0), (8, 8, 1), (0, 8, 0)))
    with pytest.raises(NotImplementedError, match="nonplanar"):
        partition_surface_distance_bands(build, surfaces=(parent,))


@pytest.mark.parametrize("route", ("B", "C"))
def test_partitioned_top_and_rotated_wall_close_native_volume_and_mesh(tmp_path, route):
    import gmsh
    from scgsim.geometry.construction.gmsh_occ import write_occ_geometry_from_plan
    from scgsim.geometry.compiler.validation import validate_volume_surface_closure
    from scgsim.geometry.models.construction import ConstructionPlanRecord
    from scgsim.geometry.models.topology import (
        SurfaceRefRecord,
        VolumePlanRecord,
        RouteABVolumePlanRecord,
    )
    from scgsim.geometry.planning.tags import plan_route_tags
    from scgsim.geometry.planning.volumes import _surface_normal_vector

    # A rotated square prism: original sidewall contacts must stay incident.
    base = ((0, 0, 0), (8, 8, 0), (0, 16, 0), (-8, 8, 0))
    top = tuple((x, y, 8) for x, y, _ in base)
    faces = [
        SurfacePlanRecord(
            "bottom",
            "entity",
            "domain_boundary",
            {"quad_points": tuple(reversed(base))},
        ),
        SurfacePlanRecord("parent", "entity", "domain_boundary", {"quad_points": top}),
    ]
    for index in range(4):
        nxt = (index + 1) % 4
        faces.append(
            SurfacePlanRecord(
                f"wall{index}",
                "entity",
                "domain_boundary",
                {
                    "quad_points": (base[index], base[nxt], top[nxt], top[index]),
                },
            )
        )
    build, _ = fixture()
    specs = (
        SurfaceDistanceBandSpec("parent", "entity", "polygon", (1, 2), 0.001),
        SurfaceDistanceBandSpec("wall0", "entity", "polygon", (1, 2), 0.001),
    )
    build = with_surface_distance_bands(build, specs)
    children, partitions, summary = partition_surface_distance_bands(
        build, surfaces=tuple(faces)
    )
    points, curves, loops, surfaces = plan_canonical_topology(surfaces=children)
    center = (0, 8, 4)
    refs = []
    for surface in surfaces:
        ring = surface.geometry_ref.get(
            "planar_loops_3d", (surface.geometry_ref.get("quad_points"),)
        )[0]
        centroid = tuple(sum(point[i] for point in ring) / len(ring) for i in range(3))
        normal = _surface_normal_vector(surface)
        dot = sum(normal[i] * (centroid[i] - center[i]) for i in range(3))
        refs.append(
            SurfaceRefRecord(
                surface.surface_id,
                "forward" if dot > 0 else "reversed",
                "planned_boundary",
            )
        )
    volume_type = RouteABVolumePlanRecord if route == "B" else VolumePlanRecord
    volume = volume_type("volume", "entity", "Si", tuple(refs))
    validate_volume_surface_closure(
        volumes=(volume,), surfaces=surfaces, surface_loops=loops
    )
    plan = ConstructionPlanRecord(
        route,
        surface_partitions=partitions,
        points=points,
        curves=curves,
        surface_loops=loops,
        surfaces=surfaces,
        volumes=(volume,),
        tags=plan_route_tags(route=route, surfaces=surfaces, volumes=(volume,)),
        metadata={"surface_distance_bands": summary},
    )
    lowered = write_occ_geometry_from_plan(plan, xao_path=tmp_path / "bands.xao")
    assert len(lowered.backend_entity_tags) == len(surfaces) + 1
    if route == "C":
        from scgsim.geometry.construction.export import export_physical_group_records

        groups = export_physical_group_records(lowered)
        child_groups = [
            group for group in groups if "surface_distance_band" in group.metadata
        ]
        assert {group.physical_name for group in child_groups} == {
            child for entry in summary for child in entry["child_surface_ids"]
        }
        assert all(len(group.entity_tags) == 1 for group in child_groups)
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        for dimension in (1, 2, 3):
            gmsh.option.setNumber(f"Mesh.MaxNumThreads{dimension}D", 1)
        gmsh.merge(str(tmp_path / "bands.xao"))
        assert len(gmsh.model.getEntities(3)) == 1
        assert round(gmsh.model.occ.getMass(3, gmsh.model.getEntities(3)[0][1])) == 1024
        gmsh.option.setNumber("Mesh.MeshSizeMin", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(tmp_path / "bands.msh"))
        assert len(gmsh.model.mesh.getElements(3)[1][0]) > 0
        names = {
            gmsh.model.getPhysicalName(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
        import json

        (tmp_path / "native-readback.json").write_text(
            json.dumps(
                {
                    "route": route,
                    "volume_um3": gmsh.model.occ.getMass(
                        3, gmsh.model.getEntities(3)[0][1]
                    ),
                    "expected_volume_um3": 1024,
                    "physical_names": sorted(names),
                    "tetrahedron_count": len(gmsh.model.mesh.getElements(3)[1][0]),
                    "backend_entity_tags": len(lowered.backend_entity_tags),
                    "children": [
                        child
                        for entry in summary
                        for child in entry["child_surface_ids"]
                    ],
                    "gmsh_version": gmsh.__version__,
                    "threads": gmsh.option.getNumber("General.NumThreads"),
                    "mesh_threads_3d": gmsh.option.getNumber("Mesh.MaxNumThreads3D"),
                },
                indent=2,
            )
        )
        assert all(
            child in names for entry in summary for child in entry["child_surface_ids"]
        )
    finally:
        gmsh.finalize()


def test_compiler_consumes_public_request_before_volume_assembly():
    from scgsim.geometry import surface_distance_band_catalogue
    from scgsim.geometry.compiler import build_route_construction_plan
    from scgsim.geometry.models.representation import RepresentationContext

    build, _ = fixture()
    entity = replace(
        build.entities[0],
        role="solution_region",
        geometry={
            "outer_loop": SQUARE,
            "z_min_um": 0,
            "z_max_um": 8,
            "thickness_um": 8,
            "domain_bounds_um": {
                "x_min_um": 0,
                "x_max_um": 8,
                "y_min_um": 0,
                "y_max_um": 8,
                "z_min_um": 0,
                "z_max_um": 8,
            },
        },
    )
    build = replace(build, entities=(entity,), metadata={})
    context = RepresentationContext({})
    original = build_route_construction_plan(
        build, route="C", representation_context=context
    )
    catalogue = surface_distance_band_catalogue(original)
    parent = next(
        item
        for item in catalogue
        if all(point[2] == 8 for point in item["rings_3d_um"][0])
    )
    request = SurfaceDistanceBandSpec(
        parent["parent_surface_id"], "entity", "polygon", (1, 2), 0.001
    )
    partitioned = build_route_construction_plan(
        with_surface_distance_bands(build, (request,)),
        route="C",
        representation_context=context,
    )
    assert partitioned.surface_partitions
    assert partitioned.metadata["surface_distance_bands"]
    assert all(
        ref.surface_id != request.parent_surface_id
        for volume in partitioned.volumes
        for ref in volume.surface_refs
    )


def test_child_interiors_are_disjoint_and_cover_parent_with_hole():
    from itertools import combinations

    build, parent = fixture(holes=(((2, 2), (6, 2), (6, 6), (2, 6)),))
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))

    def region(surface):
        rings = surface.geometry_ref["planar_loops_3d"]
        outer = [gdstk.Polygon([(x, y) for x, y, _ in rings[0]])]
        holes = [gdstk.Polygon([(x, y) for x, y, _ in ring]) for ring in rings[1:]]
        return gdstk.boolean(outer, holes, "not", precision=0.001) if holes else outer

    areas = [region(child) for child in children]
    for first, second in combinations(areas, 2):
        assert gdstk.boolean(first, second, "and", precision=0.001) == []
    original = gdstk.boolean(
        [gdstk.Polygon(SQUARE)],
        [gdstk.Polygon(((2, 2), (6, 2), (6, 6), (2, 6)))],
        "not",
        precision=0.001,
    )
    assert (
        gdstk.boolean(
            [polygon for area in areas for polygon in area],
            original,
            "xor",
            precision=0.001,
        )
        == []
    )


def test_active_exact_curve_request_fails_without_polygon_substitution():
    from scgsim.geometry import BoundaryCurveChainSpec, SourceCurveSpec
    from scgsim.geometry.compiler import build_route_construction_plan
    from scgsim.geometry.models.representation import RepresentationContext

    build, _ = fixture()
    curve = BoundaryCurveChainSpec(
        "boundary", "entity", (1, 1), (SourceCurveSpec("line", ((0, 0), (8, 0))),)
    )
    build = replace(build, boundary_curves=(curve,))
    with pytest.raises(NotImplementedError, match="polygon-only"):
        build_route_construction_plan(
            build, route="B", representation_context=RepresentationContext({})
        )


def test_generated_sa_group_anchor_retains_all_members_and_holes():
    from scgsim.geometry import surface_distance_band_catalogue
    from scgsim.geometry.models.construction import ConstructionPlanRecord

    build, parent = fixture(holes=(((2, 2), (6, 2), (6, 6), (2, 6)),))
    second = replace(build.polygons[0], polygon_id="second")
    entity = replace(build.entities[0], polygon_ids=("polygon", "second"))
    build = replace(build, polygons=(*build.polygons, second), entities=(entity,))
    parent = replace(
        parent,
        geometry_ref={
            key: value
            for key, value in parent.geometry_ref.items()
            if key != "source_polygon_ids"
        },
        metadata={
            "owner_semantic_ids": ("entity", "vacuum"),
            "source_provenance": {"members": ["first", "second"]},
        },
    )
    catalogue = surface_distance_band_catalogue(
        ConstructionPlanRecord("B", surfaces=(parent,)), build_input=build
    )
    assert catalogue[0]["source_polygon_ids"] == ("polygon", "second")
    assert len(catalogue[0]["rings_3d_um"]) == 2
    children, _, summary = partition_surface_distance_bands(build, surfaces=(parent,))
    assert area(children) == 48
    assert all(
        item["parent_source_polygon_ids"] == ("polygon", "second") for item in summary
    )
    assert all(
        child.metadata["source_provenance"] == parent.metadata["source_provenance"]
        for child in children
    )


@pytest.mark.parametrize("subset", (None, (SourceBoundarySegment("exterior", 0),)))
def test_nested_partition_without_source_boundary_is_explicitly_unsupported(subset):
    from scgsim.geometry import surface_distance_band_catalogue
    from scgsim.geometry.models.construction import ConstructionPlanRecord

    build, parent = fixture(subset=subset)
    parent = replace(
        parent, parent_surface_id="original", partition_label="existing_child"
    )
    row = surface_distance_band_catalogue(
        ConstructionPlanRecord("B", surfaces=(parent,)), build_input=build
    )[0]
    assert row["distance_origin_status"] == "unsupported"
    assert row["distance_origin_rings_3d_um"] is None
    assert "lineage" in row["distance_origin_reason"]
    with pytest.raises(ValueError, match="original source-boundary lineage"):
        partition_surface_distance_bands(build, surfaces=(parent,))


def test_existing_partition_is_flattened_and_seam_is_not_distance_origin():
    from scgsim.geometry.compiler import validate_surface_partition_coverage
    from scgsim.geometry.models.construction import SurfacePartitionRecord
    from scgsim.geometry.models.topology import InterfacePlanRecord
    from scgsim.geometry.models.representation import RepresentationContext
    from scgsim.geometry.planning.surfaces import plan_route_surfaces
    from scgsim.geometry.planning.surface_bands import synchronize_distance_partitions

    build, parent = fixture()
    interface = InterfacePlanRecord(
        parent.interface_id,
        "SA",
        ("entity", "vacuum"),
        "explicit",
        source_polygon_ids=("polygon",),
        metadata={"outer_loop": SQUARE, "plane": {"axis": "z", "value_um": 0}},
    )
    original = (
        SurfacePartitionRecord(
            "left_partition",
            parent.interface_id,
            "left",
            "left",
            metadata={
                "source_fact": "retain me",
                "outer_loop": ((0, 0), (4, 0), (4, 8), (0, 8)),
            },
        ),
        SurfacePartitionRecord(
            "right_partition",
            parent.interface_id,
            "right",
            "right",
            metadata={"outer_loop": ((4, 0), (8, 0), (8, 8), (4, 8))},
        ),
    )
    # Genuine existing explicit-partition lowering produces the retained origin.
    lowered = plan_route_surfaces(
        build,
        route="C",
        interfaces=(interface,),
        surface_partitions=original,
        construction_bodies=(),
        representation_context=RepresentationContext({}),
    )
    assert {surface.surface_id for surface in lowered} == {"left", "right"}
    build = with_surface_distance_bands(
        build, (SurfaceDistanceBandSpec("left", "entity", "polygon", (1, 2), 0.001),)
    )
    children, generated, summary = partition_surface_distance_bands(
        build, surfaces=lowered
    )
    flattened = synchronize_distance_partitions(original, generated)
    assert all(record.child_surface_id != "left" for record in flattened)
    assert original[1] in flattened
    assert len(flattened) == len(children)
    assert [
        area(
            [
                child
                for child in children
                if child.metadata.get("surface_distance_band", {}).get("band_index")
                == index
            ]
        )
        for index in range(3)
    ] == [14, 10, 8]
    assert all(
        row["distance_origin_parent_surface_id"] == f"SURF__{parent.interface_id}"
        for row in summary
    )
    for record in flattened:
        if record.child_surface_id != "right":
            assert record.parent_interface_id == parent.interface_id
            ancestor = record.metadata["ancestor_surface_partitions"][0]
            assert ancestor["partition_id"] == "left_partition"
            assert ancestor["metadata"] == original[0].metadata
    validate_surface_partition_coverage(
        interfaces=(interface,), surface_partitions=flattened, surfaces=children
    )


def test_nested_explicit_subset_uses_original_ring_indices():
    from scgsim.geometry.planning.surface_bands import partition_source_boundary

    build, parent = fixture()
    child = replace(
        parent,
        parent_surface_id="original",
        partition_label="left",
        geometry_ref={
            **parent.geometry_ref,
            "outer_loop": ((0, 0), (4, 0), (4, 8), (0, 8)),
        },
        metadata={
            "surface_partition_origin": partition_source_boundary(
                parent.geometry_ref, "original"
            )
        },
    )
    # Original edge1 is x=8. Artificial child edge1 is x=4 and must not be used.
    request = SurfaceDistanceBandSpec(
        "parent",
        "entity",
        "polygon",
        (1, 2),
        0.001,
        (SourceBoundarySegment("exterior", 1),),
    )
    build = with_surface_distance_bands(build, (request,))
    children, _, summary = partition_surface_distance_bands(build, surfaces=(child,))
    assert summary[0]["child_surface_ids"] == []
    assert summary[1]["child_surface_ids"] == []
    assert area(children) == 32


@pytest.mark.parametrize("start", (0, 4))
def test_collinear_source_contact_vertices_survive_boolean_edge_merging(start):
    outer = ((0, 0), (8, 0), (8, 8), (0, 8), (0, 5))
    outer = (*outer[start:], *outer[:start])
    build, parent = fixture(outer=outer, cuts=(0.5, 1))
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))
    assert area(children) == 64
    assert [area([child for child in children
                  if child.metadata["surface_distance_band"]["band_index"] == i])
            for i in range(3)] == [15, 13, 36]
    assert (0, 5, 0) in {point for child in children
                         for ring in child.geometry_ref["planar_loops_3d"] for point in ring}
    assert parent.geometry_ref["outer_loop"] == outer
    points, curves, loops, canonical = plan_canonical_topology(surfaces=children)
    assert points and curves and loops and all(child.outer_loop_ref for child in canonical)


def test_collinear_hole_contact_vertices_survive_boolean_edge_merging():
    hole = ((2, 2), (6, 2), (6, 6), (2, 6), (2, 4))
    build, parent = fixture(holes=(hole,), cuts=(0.5, 1))
    children, _, _ = partition_surface_distance_bands(build, surfaces=(parent,))
    assert area(children) == 48
    assert (2, 4, 0) in {point for child in children
                         for ring in child.geometry_ref["planar_loops_3d"] for point in ring}
    assert parent.geometry_ref["hole_loops"] == (hole,)


def test_noncollinear_nearby_source_segments_remain_ambiguous():
    from scgsim.geometry.planning.surface_bands import _restore_parent_boundary

    ring = ((0.0001, 0.0001), (0.0002, 0.0002), (1, 1))
    with pytest.raises(ValueError, match="lineage is ambiguous"):
        _restore_parent_boundary(ring, (SQUARE,), 0.001)



def _domain_boundary_distance_band_record(interface_type, banded):
    """Exercise real source planning/export; transport token 1 is not a native observation."""
    from scgsim.geometry import ApproximationProfile, interpret_physical_geometry
    from scgsim.geometry.compiler import build_route_construction_plan
    from scgsim.geometry.construction.export import _structured_surface_group
    from dataclasses import asdict

    def region(entity_id, material_id, kind, bounds):
        x0, x1, y0, y1, z0, z1 = bounds
        return SemanticEntitySpec(
            entity_id, "solution_region", material_id, kind, 0, "polygon",
            geometry={"outer_loop": ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                      "z_min_um": z0, "z_max_um": z1, "thickness_um": z1-z0,
                      "domain_bounds_um": {"x_min_um": x0, "x_max_um": x1,
                                           "y_min_um": y0, "y_max_um": y1,
                                           "z_min_um": z0, "z_max_um": z1}},
        )
    substrate = region("substrate", "Si", "dielectric", (0, 8, 0, 8, -2, 0))
    vacuum = region("vacuum", "air", "vacuum", (-2, 10, -2, 10, -2, 0))
    vacuum = replace(vacuum, geometry={**vacuum.geometry,
                                      "hole_loops": (substrate.geometry["outer_loop"],)})
    top = region("vacuum_top", "air", "vacuum", (-2, 10, -2, 10, 0, 6))
    source = GeometryBuildInput((), (substrate, vacuum, top))
    stack = {"layers": [], "solution_regions": {
        entity.semantic_id: {"semantic_id": entity.semantic_id,
                             "geometry": dict(entity.geometry), "metadata": {}}
        for entity in source.entities},
        "materials": {"Si": {"kind": "dielectric", "permittivity": 4},
                      "air": {"kind": "vacuum", "permittivity": 1}}, "metadata": {}}
    effective, _, physical = interpret_physical_geometry(
        source, stack, ApproximationProfile("route_b", "v1", {})
    )
    construction = build_route_construction_plan(effective, route="B", representation_context=physical.context)
    parent = next(surface for surface in construction.surfaces
                  if (interface_type == "AA" or surface.surface_role == "domain_boundary")
                  and surface.metadata["interface_type"] == interface_type)
    if banded:
        spec = SurfaceDistanceBandSpec(parent.surface_id, parent.owner_semantic_id, None, (0.5,), 0.001)
        construction = build_route_construction_plan(
            with_surface_distance_bands(effective, (spec,)), route="B", representation_context=physical.context)
        parent = next(surface for surface in construction.surfaces
                      if surface.metadata.get("surface_distance_band", {}).get("parent_surface_id") == parent.surface_id)
    tags = tuple(tag for tag in construction.tags if tag.source_record_id == parent.surface_id)
    assert len(tags) == 1
    record = asdict(_structured_surface_group(construction, tags, (parent,), (1,)))
    return record


@pytest.mark.parametrize("interface_type,banded,expected", (
    ("SA", True, True), ("SA", False, False), ("AA", True, False), ("AA", False, False),
    ("boundary", True, False), ("boundary", False, False),
))
def test_selected_sa_domain_boundary_bands_are_integrable(interface_type, banded, expected):
    from scgsim.palace.mesh.build import _add_surface_group_records

    record = _domain_boundary_distance_band_record(interface_type, banded)
    groups = {"boundary_surfaces": {}, "surface_groups": {}, "pec_surfaces": {}}
    name = record["physical_name"]
    _add_surface_group_records(groups=groups, name=name, phys_group=0, entity_tags=(),
                               record=record, route="B",
                               solution_material_kinds={"substrate": "dielectric", "vacuum": "vacuum", "vacuum_top": "vacuum"})
    info = groups["boundary_surfaces"][name]
    assert info["surface_epr"] is expected
    assert info["interface_type"] == interface_type
    if expected:
        assert info["metadata"]["surface_distance_band"] == record["metadata"]["surface_distance_band"]
        assert info["source_provenance"] == record["source_provenance"]


@pytest.mark.parametrize("missing", ("ledger", "adjacency"))
def test_selected_sa_boundary_band_still_requires_ledger_and_material_adjacency(missing):
    from scgsim.palace.mesh.build import _add_surface_group_records

    record = _domain_boundary_distance_band_record("SA", True)
    material_kinds = {"substrate": "dielectric", "vacuum": "vacuum", "vacuum_top": "vacuum"}
    if missing == "ledger":
        record["source_provenance"] = {}
        message = "contribution provenance"
    else:
        material_kinds["substrate"] = "vacuum"
        message = "dielectric/vacuum adjacency"
    with pytest.raises(ValueError, match=message):
        _add_surface_group_records(
            groups={"boundary_surfaces": {}, "surface_groups": {}, "pec_surfaces": {}},
            name=record["physical_name"], phys_group=0, entity_tags=(), record=record,
            route="B", solution_material_kinds=material_kinds)
