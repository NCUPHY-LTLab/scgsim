"""Route A Thin Film lumped ports land on the profile's sheets without a junction."""

from dataclasses import replace

import pytest

from scgsim.geometry import (
    ApproximationProfile,
    GeometryBuildInput,
    LayoutPolygonSpec,
    PhysicalLayerDeclaration,
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
    SemanticEntitySpec,
    interpret_physical_geometry,
)
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.geometry.models.grid import GeometryGridContext

GRID_CELL_UM = 0.001
GENERATED_SHEET_ORIGIN = "generated_representation_context_surface_sheet"
LEFT_EDGE_UM = 4.0
RIGHT_EDGE_UM = 6.0


def rectangle(x0, x1, y0=1.0, y1=3.0):
    """Return one counter-clockwise axis-aligned ring."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def pads_and_port(landing_um, *, left_pad_edge_um=LEFT_EDGE_UM):
    """Build two plain face metals joined by a port landing ``landing_um`` on each.

    No rectangular junction is declared, so only the selected Thin Film profile
    can say that these conductors are Route A sheets.
    """
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
    pad_loops = {
        "left": rectangle(0.0, left_pad_edge_um, 0.0, 4.0),
        "right": rectangle(RIGHT_EDGE_UM, 10.0, 0.0, 4.0),
    }
    metals = tuple(
        SemanticEntitySpec(
            f"chip/{name}",
            "conductor",
            "metal",
            "conductor",
            1,
            "polygon",
            part_role="face_metal",
            net_id=name,
            polygon_ids=(name,),
            host_void_semantic_id="upper_air",
            geometry={
                "geometry_source": "gds_polygon",
                "z_um": 0,
                "thickness_um": 0.25,
                "outer_loop": loop,
            },
            metadata={
                "logical_layer_id": "metal",
                "physical_layer": declaration.to_payload(),
            },
        )
        for name, loop in pad_loops.items()
    )
    port_loop = rectangle(LEFT_EDGE_UM - landing_um, RIGHT_EDGE_UM + landing_um)
    port = PortSheetRegionRecord(
        "port_sheet",
        "port",
        "port_polygon",
        port_loop,
        overlaps=(
            PortSheetOverlapRecord(
                "left_overlap",
                "port_sheet",
                "port_polygon",
                "chip/left",
                "left",
                rectangle(LEFT_EDGE_UM - landing_um, LEFT_EDGE_UM),
            ),
            PortSheetOverlapRecord(
                "right_overlap",
                "port_sheet",
                "port_polygon",
                "chip/right",
                "right",
                rectangle(RIGHT_EDGE_UM, RIGHT_EDGE_UM + landing_um),
            ),
        ),
        metadata={
            "source_name": "chip/o_junction_lumped",
            "target_layer": "metal",
            "port_index": 1,
            "direction": (1, 0, 0),
            "direction_raw": (1, 0, 0),
            "direction_sign_convention": "gdsfactory_port_orientation_outward",
        },
    )
    polygons = (
        *(LayoutPolygonSpec(name, "metal", loop) for name, loop in pad_loops.items()),
        LayoutPolygonSpec("port_polygon", "port", port_loop),
    )
    return metals, polygons, port


def solution(name, kind, material, bounds, holes=()):
    """Return one solution-region entity spanning the given x, y and z bounds."""
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


def thin_film_source(landing_um, **pad_options):
    """Compose the source input, source stack and Thin Film profile."""
    metals, polygons, port = pads_and_port(landing_um, **pad_options)
    substrate = solution("substrate", "dielectric", "Si", (-3, 13, -1, 5, -2, 0))
    lower = solution(
        "lower_air",
        "vacuum",
        "air",
        (-4, 14, -2, 6, -2, 0),
        (substrate.geometry["outer_loop"],),
    )
    upper = solution("upper_air", "vacuum", "air", (-4, 14, -2, 6, 0, 4.25))
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
    source = GeometryBuildInput(
        polygons,
        (*metals, substrate, lower, upper),
        port_sheet_regions=(port,),
        solution_regions=regions,
        metadata={"geometry_grid": GeometryGridContext(GRID_CELL_UM).to_payload()},
    )
    return source, stack, ApproximationProfile("thin_film", "v1", {})


def plan_route_a(landing_um, **options):
    """Interpret the Thin Film source and plan Route A without any mesh."""
    source, stack, profile = thin_film_source(landing_um, **options)
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    construction = build_route_construction_plan(
        effective, route="A", representation_context=physical.context
    )
    return effective, construction


def test_thin_film_generates_sheet_intents_without_a_rectangular_junction():
    """The selected profile alone makes both conductors Route A sheets."""
    source, stack, profile = thin_film_source(GRID_CELL_UM)
    assert not source.metadata.get("rectangular_junctions")
    assert "interface_intents_2d" not in source.metadata
    effective, _, _ = interpret_physical_geometry(source, stack, profile)
    intents = effective.metadata["interface_intents_2d"]["interfaces"]
    assert {item["owner_semantic_ids"][0] for item in intents} == {
        "chip/left",
        "chip/right",
    }
    assert {item["intent_origin"] for item in intents} == {GENERATED_SHEET_ORIGIN}


@pytest.mark.parametrize("landing_um", (GRID_CELL_UM, 0.5))
def test_port_landing_on_each_host_sheet_is_planned(landing_um):
    """A port landing one grid cell (or more) on each sheet finds its patch."""
    _, construction = plan_route_a(landing_um)
    ports = [
        surface
        for surface in construction.surfaces
        if surface.surface_role == "lumped_port"
    ]
    assert len(ports) == 1
    port = ports[0]
    assert port.metadata["owner_semantic_ids"] == ("chip/left", "chip/right")
    assert port.metadata["route_a_boundary_port"] is True
    assert len(set(port.metadata["boundary_volume_ids"])) == 2
    assert port.geometry_ref["plane"]["value_um"] == 0
    xs = {x for x, _ in port.geometry_ref["outer_loop"]}
    assert xs == {LEFT_EDGE_UM, RIGHT_EDGE_UM}
    # The host sheets keep the port-free remainder, so the landing strip is
    # carved out of the sheet and never covered twice.
    host_sheet_ids = {
        surface.owner_semantic_id
        for surface in construction.surfaces
        if surface.surface_role == "A_planned_interface"
    }
    assert host_sheet_ids == {"chip/left", "chip/right"}


def test_port_that_misses_the_host_sheet_still_fails_clearly():
    """A port footprint lying only in the gap, off the left sheet, is rejected."""
    with pytest.raises(
        ValueError,
        match="chip/left port overlap intersects 0 local Route A surface_sheet",
    ):
        # The left sheet ends two grid cells short of the one-cell port landing.
        plan_route_a(GRID_CELL_UM, left_pad_edge_um=LEFT_EDGE_UM - 3 * GRID_CELL_UM)


def test_host_without_any_sheet_is_reported_as_unrepresented():
    """A host with no sheet patch at all is named as such, not as a port miss."""
    from scgsim.geometry.planning.port_planning.route_a import (  # noqa: PLC0415 -- the contract is the unit under test
        route_a_port_sheet_sheet_contract,
    )

    overlap = PortSheetOverlapRecord(
        "left_overlap",
        "port_sheet",
        "port_polygon",
        "chip/left",
        "left",
        rectangle(LEFT_EDGE_UM - GRID_CELL_UM, LEFT_EDGE_UM),
    )
    with pytest.raises(
        ValueError, match="chip/left has no Route A surface_sheet patches"
    ):
        route_a_port_sheet_sheet_contract(("chip/left",), (), (overlap,))


def test_effective_input_keeps_explicit_intents_and_regenerates_generated_ones():
    """Only generated sheet intents are replaced; authored declarations remain."""
    source, stack, profile = thin_film_source(GRID_CELL_UM)
    explicit = {"interface_id": "authored", "intent_origin": "authored"}
    stale = {
        "interface_id": "stale",
        "intent_origin": "generated_route_a_surface_sheet",
    }
    source = replace(
        source,
        metadata={
            **source.metadata,
            "interface_intents_2d": {"interfaces": (explicit, stale)},
        },
    )
    effective, _, _ = interpret_physical_geometry(source, stack, profile)
    intents = effective.metadata["interface_intents_2d"]["interfaces"]
    assert intents[0] == explicit
    assert "stale" not in {item["interface_id"] for item in intents}
    assert len(intents) == 3


@pytest.mark.pdk
def test_public_simplepad_prepares_route_a_thin_film_ports():
    """The public 100 nm SimplePad plans Route A with its one-cell port landing."""
    pytest.importorskip("gdsfactory", reason="SimplePad layout needs gdsfactory")
    pytest.importorskip(
        "orcapen_sc_pdk", reason="public OrcaPen SC PDK is not installed"
    )
    from orcapen_sc_pdk import activate  # noqa: PLC0415 -- optional PDK, imported only when installed.
    from orcapen_sc_pdk.cells.simulation.scgsim import (  # noqa: PLC0415 -- optional PDK, imported only when installed.
        square_pad_capacitor,
    )
    from orcapen_sc_pdk.materials import get_material_records  # noqa: PLC0415 -- optional PDK, imported only when installed.
    from orcapen_sc_pdk.tech import get_single_die_layer_stack  # noqa: PLC0415 -- optional PDK, imported only when installed.

    from scgsim.geometry import GeometryPlan  # noqa: PLC0415 -- needs the optional layout stack checked above.
    from scgsim.palace.preparation.workflow import prepare_mesh_input  # noqa: PLC0415 -- needs the optional layout stack checked above.

    activate()
    plan = GeometryPlan(
        component=square_pad_capacitor(
            pad_side_um=200.0, gap_um=20.0, die_side_um=1000.0, junction_width_um=2.0
        ),
        root_id="chip",
        layer_stack=get_single_die_layer_stack(),
        material_records=get_material_records(),
        coupon_padding_um=0.0,
    )
    plan.set_nets({"pad": ("chip/PAD",), "ground": ("chip/GROUND",)})
    plan.set_vacuum_region(padding=(250.0, 250.0, 1000.0))
    snapshot = plan.prepare()
    prepared = prepare_mesh_input(
        stack=snapshot.stack,
        route="A",
        route_a_thin_film=None,
        approximation_profile=ApproximationProfile("thin_film", "v1", {}),
        vacuum_region=None,
        build_input=snapshot.geometry_input,
    )
    overlap_widths = {
        item.host_semantic_id: max(x for x, _ in item.overlap_loop)
        - min(x for x, _ in item.overlap_loop)
        for item in prepared.geometry_input.port_sheet_regions[0].overlaps
    }
    assert all(
        width == pytest.approx(GRID_CELL_UM) for width in overlap_widths.values()
    )
    construction = build_route_construction_plan(
        prepared.geometry_input,
        route="A",
        representation_context=prepared.physical_plan.context,
    )
    assert [
        surface.surface_id
        for surface in construction.surfaces
        if surface.surface_role == "lumped_port"
    ] == ["SURF__LUMPED_PORT__chip_o_junction_lumped"]
