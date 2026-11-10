"""Palace makes every lumped port the central rectangle of its drawn sheet (the shared junction partition).

Each layout goes through ``EigenmodeSim``'s own preparation, and, where a claim is about the built
model, through the SGB geometry build that ``simulation_drc`` and ``mesh()`` share.  No mesher or
solver runs.
"""

from __future__ import annotations

import json
import math

import pytest
from test_simulation_drc import MATERIALS, NO_VACUUM_STACK, two_pads

import scgsim.palace.preparation.workflow as workflow
from scgsim.geometry import GeometryPlan
from scgsim.palace import EigenmodeSim

PAD_RADIUS_UM = 100.0
PAD_VERTICES = 1024
ELECTRODE_X_UM = 120.0
SHEET_HALF_WIDTH_UM = 1.0
GRID_UM = 0.001
# The class A wall-column limit of the meshing stage, 10 grid steps (issue 125).
STEP_LIMIT_UM = 10 * GRID_UM
NETS = {"pad": ("chip/pad",), "electrode": ("chip/electrode",)}


def pad_and_electrode(monkeypatch, *, round_pad: bool, exact_arc: bool = False):
    """A pad and a straight electrode 20 um away, joined by a 2 um lumped-port sheet.

    The round pad is a 1024-sided polygon of radius 100 um; with ``exact_arc`` it also carries
    the authored circle, as the PDK's circular pad does, so SGB builds the exact arc.  The square
    pad has a straight edge at x = 100 um.  The sheet starts inside the pad and ends 1 um inside
    the electrode.
    """
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    if round_pad:
        pad = [
            (
                PAD_RADIUS_UM * math.cos(2 * math.pi * i / PAD_VERTICES),
                PAD_RADIUS_UM * math.sin(2 * math.pi * i / PAD_VERTICES),
            )
            for i in range(PAD_VERTICES)
        ]
        sheet_start = PAD_RADIUS_UM - 0.1
    else:
        pad = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
        sheet_start = 99.0
    component.add_polygon(pad, layer=(1, 0))
    component.add_polygon(
        [(ELECTRODE_X_UM, -30), (150, -30), (150, 30), (ELECTRODE_X_UM, 30)],
        layer=(2, 0),
    )
    sheet_end = ELECTRODE_X_UM + 1.0
    component.add_polygon(
        [
            (sheet_start, -SHEET_HALF_WIDTH_UM),
            (sheet_end, -SHEET_HALF_WIDTH_UM),
            (sheet_end, SHEET_HALF_WIDTH_UM),
            (sheet_start, SHEET_HALF_WIDTH_UM),
        ],
        layer=(9, 0),
    )
    component.add_port(
        name="P1",
        center=((sheet_start + sheet_end) / 2, 0),
        width=2 * SHEET_HALF_WIDTH_UM,
        orientation=0,
        layer=(9, 0),
    )
    semantics = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": "pad",
                "level": "metal",
                "gds_layer": [1, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (0.0, 0.0),
                },
            },
            {
                "semantic_id": "electrode",
                "level": "metal",
                "gds_layer": [2, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (135.0, 0.0),
                },
            },
        ],
        "ports": [{"name": "P1", "layer": [9, 0], "target_layer": "metal"}],
    }
    if exact_arc:
        r = PAD_RADIUS_UM
        semantics["boundary_curves"] = [
            {
                "boundary_id": "PAD_OUTER",
                "entity_id": "pad",
                "selector_point_um": (0.0, 0.0),
                "role": "outer",
                "curves": [
                    {"kind": "circular_arc", "points_um": [[r, 0], [0, r], [-r, 0]]},
                    {"kind": "circular_arc", "points_um": [[-r, 0], [0, -r], [r, 0]]},
                ],
            }
        ]
    component.info["component_semantics"] = semantics
    return component


def eigenmode_sim(component, nets=NETS):
    """An ``EigenmodeSim`` on the component's plan with its one lumped port, as a user sets it up."""
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_nets(nets)
    sim = EigenmodeSim()
    sim.set_plan(plan.prepare())
    sim.add_port("chip/P1", layer="metal", layout_sheet=True, inductance=1e-9)
    return sim


def prepared_port(sim):
    """The prepared geometry input of the simulation, with its one port region and partition record."""
    request = sim._geometry_request(sim.stack)  # noqa: SLF001 -- the preparation mesh() runs
    prepared = request.source["source_geometry_input"]
    (region,) = prepared.port_sheet_regions
    record = prepared.metadata["junction_partitions"][
        region.metadata["rectangular_junction_partition"]
    ]
    return prepared, region, record


def built_sidecar(monkeypatch, component, nets=NETS):
    """The sidecar of the SGB build ``simulation_drc`` makes, read before its folder is removed."""
    sidecars = []
    real = EigenmodeSim._build_geometry  # noqa: SLF001 -- the geometry half the check runs

    def recording(self, output_dir):
        geometry = real(self, output_dir)
        sidecars.append(json.loads(geometry.artifacts.sidecar.read_bytes()))
        return geometry

    monkeypatch.setattr(EigenmodeSim, "_build_geometry", recording)
    assert (
        EigenmodeSim.simulation_drc(
            component,
            layer_stack=NO_VACUUM_STACK,
            material_records=MATERIALS,
            nets=nets,
        )
        is None
    )
    (sidecar,) = sidecars
    return sidecar


def port_group(sidecar):
    """The one lumped-port physical group of a sidecar."""
    (group,) = [
        group
        for name, group in sidecar["physical_groups"].items()
        if name.startswith("PORT::")
    ]
    return group


def assert_rectangle_along_x(exterior):
    """Four corners, two x and two y: its bounding-box length is its length along the port direction."""
    xs = {x for x, _ in exterior}
    ys = {y for _, y in exterior}
    assert len(exterior) == 4
    assert len(xs) == 2
    assert ys == {-SHEET_HALF_WIDTH_UM, SHEET_HALF_WIDTH_UM}
    return min(xs), max(xs)


@pytest.mark.parametrize("exact_arc", [False, True], ids=["polyline", "exact-arc"])
def test_a_port_on_a_round_pad_is_a_rectangle_whose_ends_join_their_conductors(
    monkeypatch, exact_arc
):
    """The port face is the central rectangle; the end piece on the pad is pad metal, the straight end empty."""
    component = pad_and_electrode(monkeypatch, round_pad=True, exact_arc=exact_arc)
    prepared, region, record = prepared_port(eigenmode_sim(component))
    start, end = assert_rectangle_along_x(region.exterior)
    assert end == pytest.approx(ELECTRODE_X_UM)
    assert start > PAD_RADIUS_UM
    entities = {entity.semantic_id: entity for entity in prepared.entities}
    pad_end, electrode_end = record["ends"]
    assert (pad_end["source_entity_id"], pad_end["net_id"]) == ("chip/pad", "pad")
    assert pad_end["polygon_id"] in entities["chip/pad"].polygon_ids
    assert electrode_end["source_entity_id"] == "chip/electrode"
    assert electrode_end["empty_residual"]
    # SGB builds the same rectangle as the port face: its area is the rectangle's.
    group = port_group(built_sidecar(monkeypatch, component))
    assert group["area_um2"] == pytest.approx(
        (end - start) * 2 * SHEET_HALF_WIDTH_UM, abs=1e-9
    )


@pytest.mark.parametrize("exact_arc", [False, True], ids=["polyline", "exact-arc"])
def test_the_cut_on_a_round_pad_is_the_first_whose_steps_reach_the_class_a_limit(
    monkeypatch, exact_arc
):
    """From the largest rectangle the pad-side cut moves one grid step at a time until both side steps are 10 steps."""
    component = pad_and_electrode(monkeypatch, round_pad=True, exact_arc=exact_arc)
    _, region, record = prepared_port(eigenmode_sim(component))
    checks = [
        check
        for check in record["candidate_checks"]
        if check["side"] == "A" and check["valid"]
    ]
    largest, *moved = checks
    assert moved, (
        "the largest rectangle already passed; the case does not test the rule"
    )
    assert all(step < STEP_LIMIT_UM for step in largest["side_steps_um"])
    chosen = checks[-1]
    assert all(step >= STEP_LIMIT_UM - 1e-12 for step in chosen["side_steps_um"])
    assert all(
        any(step < STEP_LIMIT_UM - 1e-12 for step in check["side_steps_um"])
        for check in checks[:-1]
    )
    assert [check["index"] for check in checks] == list(
        range(largest["index"], chosen["index"] + 1)
    )
    assert record["terminal_step_limit_um"] == pytest.approx(STEP_LIMIT_UM)
    start, _ = assert_rectangle_along_x(region.exterior)
    assert start == pytest.approx(
        record["local_origin_projection_um"][0] + chosen["actual_s_um"]
    )
    if exact_arc:
        # Measured where SGB builds the arc: it meets y = +-1 um at sqrt(100^2 - 1), on the grid at 99.995 um.
        assert start == pytest.approx(99.995 + STEP_LIMIT_UM)


def test_a_port_between_straight_edges_keeps_its_face_and_conductors(monkeypatch):
    """On straight edges the cuts are the conductor edges: the face and the conductors are those built before."""
    component = pad_and_electrode(monkeypatch, round_pad=False)
    _, region, record = prepared_port(eigenmode_sim(component))
    assert assert_rectangle_along_x(region.exterior) == pytest.approx(
        (100.0, ELECTRODE_X_UM)
    )
    assert [end["empty_residual"] for end in record["ends"]] == [True, True]
    partitioned = built_sidecar(monkeypatch, component)
    monkeypatch.undo()
    # Preparation as it was before every port was partitioned: the sheet whole, its landings cut by SGB.
    monkeypatch.setattr(
        workflow, "with_rectangular_ports", lambda build_input: build_input
    )
    component = pad_and_electrode(monkeypatch, round_pad=False)
    whole = built_sidecar(monkeypatch, component)

    def conductors_and_port(sidecar):
        return {
            name: (group.get("area_um2"), group.get("entities"), group.get("face_kind"))
            for name, group in sidecar["physical_groups"].items()
        }

    assert conductors_and_port(partitioned) == conductors_and_port(whole)


def test_a_port_on_conductors_without_a_net_is_refused(monkeypatch):
    """The end pieces join their terminal's Net, so a port between conductors without Nets is refused."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    with pytest.raises(ValueError, match=r"lumped port 'chip/P1' lands on .* no Net"):
        EigenmodeSim.simulation_drc(
            component, layer_stack=NO_VACUUM_STACK, material_records=MATERIALS
        )
