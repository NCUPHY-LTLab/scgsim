# ruff: noqa: PLC0415 -- gdsfactory and the PDK are imported where the layout is built: the PDK is not in SCGSim's own lock, and the public suite collects this module without it
"""The launcher-to-launcher layout of issue 187, composed from OrcaPen SC PDK cells, and the plan a Driven study gives it.

Two PDK ``launcher`` cells joined by a 900 um ``cpw_6_7_6`` straight on a 1870 x 900 um ``D0_SUBSTRATE_AREA``
die, with no dicing frame (the plan's run of fact 9).  Each launcher is a 150 x 150 um pad tapering over
150 um to the CPW, 85 um of gap behind and beside the pad.  The parent authors ``component_semantics`` v2:
``SIGNAL`` (the one DRAW polygon the launchers and the line flatten into), ``GROUND`` (the die face less the
ground mask) and the two port sheets, ``o_lumped_left`` and ``o_lumped_right``.

The launcher's own sheet abuts the pad and the ground with no overlap, so it lands on no conductor and cannot
be a port (orcapen-sc-pdk issue 67); the launcher's sheet is therefore drawn on an unused layer and each port
sheet is drawn here, 85 x 150 um behind its pad, landing ``LANDING_UM`` (10 grid steps) into the pad and the
ground, the landing the meshing stage accepts (fact 5).  The PDK draws both sheets from the ground to the pad;
``right_sense="reversed"`` draws the right one from the pad to the ground.

A ``pdk`` test needs the PDK environment of CI's pdk-tests job.

:func:`mesh_without_gmsh` stands in for the one stage of ``mesh()`` the Driven config tests leave out, Gmsh:
``mesh()`` still runs its preparation, the SGB build and the meshing-stage check for real, and
``write_config()`` builds from the physical groups of that real SGB model, the groups the mesher would carry.
"""

from __future__ import annotations

from typing import Any, Literal

from scgsim.contracts.palace_run_folder import write_json
from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.palace.mesh.build import MeshBuildResult, RouteGeometry
from scgsim.palace.mesh.controls import MeshControls
from scgsim.palace.mesh.manifest import build_mesh_manifest
from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.simulation import core

LINE_UM = 900.0
DIE_UM = (1870.0, 900.0)
PAD_UM = 150.0
TAPER_UM = 150.0
BACK_GAP_UM = 85.0
LANDING_UM = 0.01
PORT_IMPEDANCE_OHM = 50.0
# (x, y, z) padding in um of the plan's vacuum: none in the plane, 1000 um below and above (issue 182).
VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)
NETS = {"signal": ("chip/SIGNAL",), "ground": ("chip/GROUND",)}
PORT_SHEETS = ("chip/o_lumped_left", "chip/o_lumped_right")
# A GDS layer no stack level reads: the launcher's own, zero-overlap sheet goes there and is not modeled.
_UNUSED_LAYER = (250, 0)

Sense = Literal["pdk", "reversed"]


def launcher_to_launcher(*, right_sense: Sense = "pdk") -> Any:
    """The launcher-to-launcher component; ``right_sense`` is the sense the right port sheet is drawn in."""
    import gdsfactory as gf
    from orcapen_sc_pdk import activate
    from orcapen_sc_pdk.ports import add_driven_lumped_port
    from orcapen_sc_pdk.tech import LAYER

    activate()
    component = gf.Component()
    launcher = gf.get_component(
        "launcher",
        cpw_xs="cpw_6_7_6",
        sim_boundary_layer=_UNUSED_LAYER,
        curve_sampling="simulation",
    )
    left = component << launcher
    left.move(origin=left.ports["o_neck"].center, destination=(-LINE_UM / 2, 0.0))
    right = component << launcher
    right.rotate(180)
    right.move(origin=right.ports["o_neck"].center, destination=(LINE_UM / 2, 0.0))
    line = component << gf.components.straight(
        length=LINE_UM, cross_section=gf.get_cross_section("cpw_6_7_6")
    )
    line.move(origin=line.ports["o1"].center, destination=left.ports["o_neck"].center)
    component << gf.components.rectangle(
        size=DIE_UM, centered=True, layer=LAYER.D0_SUBSTRATE_AREA
    )
    pad_back_um = LINE_UM / 2 + TAPER_UM + PAD_UM
    for name, side in (("o_lumped_left", -1.0), ("o_lumped_right", 1.0)):
        x0, x1 = sorted(
            (
                side * (pad_back_um - LANDING_UM),
                side * (pad_back_um + BACK_GAP_UM + LANDING_UM),
            )
        )
        half = PAD_UM / 2
        component.add_polygon(
            [(x0, -half), (x1, -half), (x1, half), (x0, half)],
            layer=LAYER.D0_TOP_SIM_BOUNDARY,
        )
        # The orientation points along the sheet from its start conductor to its end conductor.
        toward_pad = 0.0 if side < 0 else 180.0
        drawn_reversed = name == "o_lumped_right" and right_sense == "reversed"
        add_driven_lumped_port(
            component,
            name=name,
            center=(side * (pad_back_um + BACK_GAP_UM / 2), 0.0),
            orientation=(toward_pad + 180.0) % 360.0 if drawn_reversed else toward_pad,
            layer=LAYER.D0_TOP_SIM_BOUNDARY,
        )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": "SIGNAL",
                "level": "D0_TOP_M1",
                "gds_layer": tuple(LAYER.D0_TOP_M1_DRAW),
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (0.0, 0.0),
                },
            },
            {
                "semantic_id": "GROUND",
                "level": "D0_TOP_M1",
                "gds_layer": tuple(LAYER.D0_TOP_GROUND_MASK),
            },
        ],
        "ports": [
            {
                "name": name,
                "layer": tuple(LAYER.D0_TOP_SIM_BOUNDARY),
                "target_layer": "D0_TOP_M1",
            }
            for name in ("o_lumped_left", "o_lumped_right")
        ],
    }
    return component


def launcher_plan(
    *,
    right_sense: Sense = "pdk",
    surface_epr: bool = False,
    bands: bool = False,
    lumped_elements: Any = None,
) -> Any:
    """The ``GeometryPlan`` of :func:`launcher_to_launcher`: Nets, vacuum, and the two sheets as 50 ohm ports.

    The ports are declared left then right, so the left launcher is port 1.  ``surface_epr`` sets the PDK's
    literature-central MA, MS and SA presets, cut as bands at 0.5 and 1.0 um with ``bands``;
    ``lumped_elements`` replaces the two ports.
    """
    from orcapen_sc_pdk.materials import get_material_records
    from orcapen_sc_pdk.tech import get_single_die_layer_stack

    from scgsim.geometry import GeometryPlan, LumpedPort

    plan = GeometryPlan(
        launcher_to_launcher(right_sense=right_sense),
        root_id="chip",
        layer_stack=get_single_die_layer_stack(),
        material_records=get_material_records(),
        coupon_padding_um=0.0,
    )
    plan.set_nets(NETS)
    plan.set_vacuum_region(VACUUM_PADDING_UM)
    if surface_epr:
        plan.set_surface_epr(literature_specs(bands=bands))
    plan.set_lumped_elements(
        [
            LumpedPort(sheet, reference_net="ground", impedance_ohm=PORT_IMPEDANCE_OHM)
            for sheet in PORT_SHEETS
        ]
        if lumped_elements is None
        else lumped_elements
    )
    return plan


def literature_specs(*, bands: bool) -> dict[str, dict[str, Any]]:
    """MA, MS and SA from the PDK's literature-central presets; with ``bands``, margins 0.5 and 1.0 um."""
    from orcapen_sc_pdk import (
        get_interface_preset_records,
        validate_interface_preset_records,
    )

    records = validate_interface_preset_records(get_interface_preset_records())
    specs = {}
    for kind in ("MA", "MS", "SA"):
        record = records[f"LiteratureCentral_{kind}"]
        specs[kind] = {
            key: record[key]
            for key in ("thickness", "permittivity", "loss_tangent", "source")
        }
        if bands:
            specs[kind]["inset_margins_um"] = [0.5, 1.0]
    return specs


def mesh_without_gmsh(monkeypatch: Any) -> None:
    """Make ``mesh()`` keep its real SGB build and write a placeholder ``palace.msh`` in place of Gmsh's.

    The manifest and the route provenance are written from the real sidecar, as the mesher writes them, so
    ``write_config()`` and ``prepare_handoff()`` read the run folder they would read after a real mesh.
    """

    def mesh(built: RouteGeometry, controls: MeshControls) -> MeshBuildResult:
        paths, artifacts = built.paths, built.artifacts
        sidecar = read_sidecar(artifacts.sidecar)
        groups = groups_from_sidecar(sidecar)
        paths.mesh_path.write_text("placeholder: Gmsh did not run\n", encoding="utf-8")
        write_json(
            paths.manifest_path,
            build_mesh_manifest(
                groups=groups,
                sidecar=sidecar,
                sidecar_file={"path": paths.relative(artifacts.sidecar), "sha256": ""},
                mesh_name=paths.mesh_path.name,
                route=built.route,
                route_a_thin_film=built.route_a_thin_film,
                meshing={},
            ),
        )
        thin_film = built.route_a_thin_film
        write_json(
            paths.provenance_path,
            {
                "route": built.route,
                **({} if thin_film is None else {"route_a_thin_film": thin_film}),
            },
        )
        return MeshBuildResult(
            output_dir=paths.run_dir,
            mesh_path=paths.mesh_path,
            groups=groups,
            mesh_manifest_path=paths.manifest_path,
            provenance_path=paths.provenance_path,
            stack_path=paths.stack_path,
            xao_path=artifacts.xao,
            sidecar_path=artifacts.sidecar,
            quality_report_path=paths.mesh_dir / "quality.json",
        )

    monkeypatch.setattr(core, "mesh_route_geometry", mesh)
