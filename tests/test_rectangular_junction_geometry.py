"""Shared real polygon partitioning; no native model or private study data."""

from dataclasses import replace
import json

import gdstk
import pytest

from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    JunctionTerminalBinding,
    RectangularJunctionSpec,
    PortSheetRegionRecord,
    with_rectangular_junctions,
)
from scgsim.geometry.planning.junction_partition import derive_rectangular_junctions


SHEET = ((0, 0), (10, 0), (10, 2), (0, 2))
LEFT = ((-2, 0), (3, 0), (3, 1), (2, 1), (2, 2), (-2, 2))
RIGHT = ((7, 0), (12, 0), (12, 2), (8, 2), (8, 1), (7, 1))


def fixture(*, other=False, b_z=0):
    polygons = [
        LayoutPolygonSpec("left", "metal", LEFT),
        LayoutPolygonSpec("right", "metal", RIGHT),
    ]
    entities = []
    for name, net, z in (("left", "A", 0), ("right", "B", b_z)):
        entities.append(
            SemanticEntitySpec(
                "chip/" + name,
                "conductor",
                "metal",
                "conductor",
                1,
                "polygon",
                part_role="face_metal",
                net_id=net,
                polygon_ids=(name,),
                route_representations={"A": "surface_sheet", "B": "solid"},
                geometry={
                    "geometry_source": "gds_polygon",
                    "z_um": z,
                    "thickness_um": 0,
                },
                metadata={
                    "source_occurrence_path": "chip",
                    "logical_layer_id": "metal",
                },
            )
        )
    if other:
        polygons.append(
            LayoutPolygonSpec("other", "metal", ((4, 0), (5, 0), (5, 1), (4, 1)))
        )
        entities.append(
            replace(
                entities[0],
                semantic_id="chip/other",
                net_id="C",
                polygon_ids=("other",),
            )
        )
    region = PortSheetRegionRecord(
        "sheet",
        "junction",
        "sheet_polygon",
        SHEET,
        (),
        (),
        {
            "source": "palace_lumped_port_sheet",
            "source_name": "chip/jj/port",
            "source_occurrence_path": "chip/jj",
            "target_layer": "metal",
            "port_index": 1,
            "direction": (1, 0, 0),
            "direction_raw": (1, 0, 0),
            "direction_sign_convention": "gdsfactory_port_orientation_outward",
        },
    )
    build = GeometryBuildInput(
        tuple(polygons),
        tuple(entities),
        metadata={"source_dbu_um": 0.5},
        port_sheet_regions=(region,),
    )
    spec = RectangularJunctionSpec(
        "junction",
        "sheet",
        "sheet_polygon",
        "chip/jj",
        JunctionTerminalBinding("chip/left", "chip", "A"),
        JunctionTerminalBinding("chip/right", "chip", "B"),
    )
    return build, spec


def _region(rings):
    return gdstk.boolean(
        [gdstk.Polygon(rings[0])],
        [gdstk.Polygon(hole) for hole in rings[1:]],
        "not",
        precision=1e-9,
    )


def test_explicit_child_sheet_parent_terminals_and_request_codec():
    build, spec = fixture()
    assert build.port_sheet_regions[0].overlaps == ()
    restored = RectangularJunctionSpec.from_payload(
        json.loads(json.dumps(spec.to_payload()))
    )
    bound = with_rectangular_junctions(build, (restored,))
    assert {item.host_semantic_id for item in bound.port_sheet_regions[0].overlaps} == {
        "chip/left",
        "chip/right",
    }
    assert build.port_sheet_regions[0].overlaps == ()
    assert bound.metadata["rectangular_junctions"] == (spec.to_payload(),)


def test_rectangle_and_terminal_owned_ends_cover_free_region_once():
    build, spec = fixture()
    result = derive_rectangular_junctions(with_rectangular_junctions(build, (spec,)))
    record = result.metadata["junction_partitions"]["junction"]
    polygons = {item.polygon_id: item for item in result.polygons}
    central = polygons[record["central_polygon_id"]]
    assert len(central.exterior) == 4 and central.holes == ()
    assert (
        len({x for x, _ in central.exterior})
        == len({y for _, y in central.exterior})
        == 2
    )
    parts = [[gdstk.Polygon(central.exterior)]]
    for end, expected in zip(record["ends"], ("chip/left", "chip/right")):
        assert end["source_entity_id"] == expected
        owner = next(
            entity for entity in result.entities if entity.semantic_id == expected
        )
        assert end["net_id"] == owner.net_id
        assert end["material_id"] == owner.material_id == "metal"
        assert end["polygon_id"] in owner.polygon_ids
        parts.append(
            _region(
                (
                    polygons[end["polygon_id"]].exterior,
                    *polygons[end["polygon_id"]].holes,
                )
            )
        )
    union = gdstk.boolean(
        [poly for part in parts for poly in part], [], "or", precision=1e-9
    )
    free = gdstk.boolean(
        [gdstk.Polygon(SHEET)],
        [gdstk.Polygon(LEFT), gdstk.Polygon(RIGHT)],
        "not",
        precision=1e-9,
    )
    assert sum(poly.area() for poly in union) == 10
    assert not gdstk.boolean(union, free, "xor", precision=1e-9)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not gdstk.boolean(parts[i], parts[j], "and", precision=1e-9)
    assert result.port_sheet_regions[0].exterior == central.exterior
    assert record["support_plane"]["source_z_um"] == record["z_um"] == 0
    assert build.polygons[0].exterior == LEFT
    assert "junction_partitions" not in build.metadata


@pytest.mark.parametrize("change", ("missing", "occurrence", "net", "sheet"))
def test_explicit_binding_rejects_wrong_source_ownership(change):
    build, spec = fixture()
    if change == "sheet":
        spec = replace(spec, source_polygon_id="missing")
    else:
        terminal = replace(
            spec.terminal_a,
            **{
                "missing": {"source_entity_id": "missing"},
                "occurrence": {"source_occurrence_path": "wrong"},
                "net": {"net_id": "wrong"},
            }[change],
        )
        spec = replace(spec, terminal_a=terminal)
    with pytest.raises(ValueError):
        with_rectangular_junctions(build, (spec,))


def test_real_third_net_contact_is_not_filtered_by_explicit_two_terminals():
    build, spec = fixture(other=True)
    with pytest.raises(ValueError):
        derive_rectangular_junctions(with_rectangular_junctions(build, (spec,)))


def test_incompatible_actual_source_planes_fail():
    build, spec = fixture(b_z=1)
    with pytest.raises(ValueError, match="support plane|authored support"):
        derive_rectangular_junctions(with_rectangular_junctions(build, (spec,)))


def test_aedt_adapter_keeps_legacy_partition_identity_and_geometry():
    from scgsim.aedt.specs.epr import PlanarJunction
    from scgsim.aedt.preparation.junction_partition import partition_junctions

    build, spec = fixture()
    bound = with_rectangular_junctions(build, (spec,))
    junction = PlanarJunction("junction", "sheet_polygon", "A", "B", (1, 0), 2, 1e-9, 0)
    legacy = partition_junctions(bound, (junction,), 0.5)
    record = legacy.metadata["aedt_junction_partitions"]["junction"]
    common = derive_rectangular_junctions(bound).metadata["junction_partitions"][
        "junction"
    ]
    assert record["method"] == "scgsim.aedt.junction-partition.v1"
    assert (
        record["support_plane"]["schema_version"]
        == "scgsim.aedt.junction-support-plane.v1"
    )
    assert record["central_polygon_id"].startswith("aedt_junction_")
    assert record["central_model_regions"] == common["central_model_regions"]
    assert [end["model_regions"] for end in record["ends"]] == [
        end["model_regions"] for end in common["ends"]
    ]


def physical_fixture():
    from scgsim.geometry import PhysicalLayerDeclaration, ApproximationProfile

    build, spec = fixture()
    declaration = PhysicalLayerDeclaration(
        "film",
        "metal",
        {"level": "metal", "gds_pair": [1, 0]},
        (0, 0.25),
        "metal",
        "stack_film",
        {"normal_thickness_um": 0.25},
        {"layer_attachment_face": "lower"},
    )
    metals = tuple(
        replace(
            item,
            host_void_semantic_id="upper_air",
            geometry={**item.geometry, "thickness_um": 0.25},
            metadata={**item.metadata, "physical_layer": declaration.to_payload()},
        )
        for item in build.entities
    )

    def solution(name, kind, material, bounds, holes=()):
        x0, x1, y0, y1, z0, z1 = bounds
        return SemanticEntitySpec(
            name,
            "solution_region",
            material,
            kind,
            0,
            "polygon",
            geometry={
                "outer_loop": ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                "hole_loops": holes,
                "z_min_um": z0,
                "z_max_um": z1,
                "thickness_um": z1 - z0,
                "domain_bounds_um": {
                    "x_min_um": x0,
                    "x_max_um": x1,
                    "y_min_um": y0,
                    "y_max_um": y1,
                    "z_min_um": z0,
                    "z_max_um": z1,
                },
            },
        )

    substrate = solution("substrate", "dielectric", "Si", (-3, 13, -1, 3, -2, 0))
    lower = solution(
        "lower_air",
        "vacuum",
        "air",
        (-4, 14, -2, 4, -2, 0),
        (substrate.geometry["outer_loop"],),
    )
    upper = solution("upper_air", "vacuum", "air", (-4, 14, -2, 4, 0, 4.25))
    regions = {
        item.semantic_id: {
            "semantic_id": item.semantic_id,
            "geometry": dict(item.geometry),
            "metadata": {},
        }
        for item in (substrate, lower, upper)
    }
    stack = {
        "layers": [
            {
                "semantic_id": item.semantic_id,
                "geometry": dict(item.geometry),
                "metadata": dict(item.metadata),
            }
            for item in metals
        ],
        "solution_regions": regions,
        "materials": {
            "Si": {"kind": "dielectric", "permittivity": 4, "loss_tangent": 0},
            "air": {"kind": "vacuum", "permittivity": 1, "loss_tangent": 0},
            "metal": {"kind": "conductor"},
        },
        "metadata": {},
    }
    source = replace(
        build, entities=(*metals, substrate, lower, upper), solution_regions=regions
    )
    profile = ApproximationProfile(
        "substrate_face",
        "v1",
        {
            "film_bindings": {
                "film": {
                    "supporting_entity_id": "substrate",
                    "supporting_side": "upper",
                }
            },
        },
    )
    return with_rectangular_junctions(source, (spec,)), stack, profile


def test_physical_attachment_then_full_construction_uses_same_derived_rectangle():
    from scgsim.geometry import (
        interpret_physical_geometry,
    )
    from scgsim.geometry.compiler import build_route_construction_plan

    source, stack, profile = physical_fixture()
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    record = effective.metadata["junction_partitions"]["junction"]
    assert record["support_plane"]["source_z_um"] == record["z_um"] == 0
    assert record["support_plane"]["source_hosts"][0]["source_interval_um"] == [0, 0.25]
    construction = build_route_construction_plan(
        effective, route="A", representation_context=physical.context
    )
    port = next(
        item for item in construction.surfaces if item.surface_role == "lumped_port"
    )
    central = next(
        item
        for item in effective.polygons
        if item.polygon_id == record["central_polygon_id"]
    )
    assert port.geometry_ref["outer_loop"] == central.exterior
    assert port.geometry_ref["plane"]["value_um"] == 0
    assert len(port.metadata["boundary_volume_ids"]) == 2
    assert construction.surfaces




def test_aedt_consumes_common_geometry_without_partitioning_twice():
    from scgsim.aedt.specs.epr import PlanarJunction
    from scgsim.aedt.preparation.junction_partition import partition_junctions

    build, spec = fixture()
    shared = derive_rectangular_junctions(with_rectangular_junctions(build, (spec,)))
    junction = PlanarJunction("junction", "sheet_polygon", "A", "B", (1, 0), 2, 1e-9, 0)
    adapted = partition_junctions(shared, (junction,), 0.5)
    assert adapted.polygons is shared.polygons
    assert adapted.entities is shared.entities
    record = adapted.metadata["aedt_junction_partitions"]["junction"]
    assert (
        record["central_polygon_id"]
        == shared.metadata["junction_partitions"]["junction"]["central_polygon_id"]
    )
    assert record["shared_geometry_method"] == "scgsim.geometry.junction-partition.v1"
    assert shared.metadata.get("aedt_junction_partitions") is None
    with pytest.raises(ValueError, match="source/terminal binding"):
        partition_junctions(shared, (replace(junction, terminal_b_net="C"),), 0.5)
