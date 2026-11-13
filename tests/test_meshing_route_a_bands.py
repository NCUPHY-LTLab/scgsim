"""Surface-EPR bands on Route A: the default recipe meshes the public SimplePad with bands clean.

A Route A sheet has no film wall, so no film-wall size reaches its bands; before band edges were sized,
this model stalled with every remaining bad element on a band boundary curve.  The test builds, meshes and
lowers the mesh for Palace (several seconds of Gmsh), so it is slow, and it needs the public PDK.
"""

import pytest

from scgsim.geometry import ApproximationProfile, GeometryPlan
from scgsim.meshing import FeatureRecipe
from scgsim.palace import EigenmodeSim
from scgsim.palace.mesh.controls import DEFAULT_QUALITY_THRESHOLD

MARGINS_UM = [0.5, 1.0]
INTERFACE_KINDS = ("MA", "MS", "SA")  # metal-air, metal-substrate, substrate-air
# The plan vacuum of the PDK's single-die Palace notebooks.
NOTEBOOK_VACUUM_UM = (250.0, 250.0, 1000.0)
PORT_INDUCTANCE_H = 10e-9


@pytest.mark.pdk
@pytest.mark.slow
def test_the_default_recipe_meshes_simplepad_on_route_a_with_bands_clean(tmp_path):
    """SimplePad on Route A with bands 0.5 and 1 um on MA, MS and SA meshes clean with the default recipe."""
    pytest.importorskip("gmsh", reason="meshing needs Gmsh (extra: palace)")
    pytest.importorskip(
        "orcapen_sc_pdk", reason="public OrcaPen SC PDK is not installed"
    )
    from orcapen_sc_pdk import (  # noqa: PLC0415 -- optional PDK, imported only when installed.
        activate,
        get_interface_preset_records,
        validate_interface_preset_records,
    )
    from orcapen_sc_pdk.cells.simulation.scgsim import (  # noqa: PLC0415 -- optional PDK, imported only when installed.
        square_pad_capacitor,
    )
    from orcapen_sc_pdk.materials import get_material_records  # noqa: PLC0415 -- optional PDK, imported only when installed.
    from orcapen_sc_pdk.tech import get_single_die_layer_stack  # noqa: PLC0415 -- optional PDK, imported only when installed.

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
    plan.set_vacuum_region(padding=NOTEBOOK_VACUUM_UM)
    presets = validate_interface_preset_records(get_interface_preset_records())
    specs = {
        kind: {
            **{
                key: presets[f"LiteratureCentral_{kind}"][key]
                for key in ("thickness", "permittivity", "loss_tangent", "source")
            },
            "inset_margins_um": MARGINS_UM,
        }
        for kind in INTERFACE_KINDS
    }
    sim = EigenmodeSim()
    sim.set_plan(plan.prepare())
    sim.set_output_dir(tmp_path / "run")
    sim.add_port(
        "chip/o_junction_lumped",
        layer="D0_TOP_M1",
        layout_sheet=True,
        inductance=PORT_INDUCTANCE_H,
    )
    sim.set_route(
        "A", approximation_profile=ApproximationProfile("thin_film", "v1", {})
    )
    sim.set_surface_epr(specs)

    sim.mesh()

    meshing = sim.mesh_summary()["meshing"]
    assert meshing["recipe"]["version"] == FeatureRecipe.version
    assert meshing["stop"] == "clean"
    assert meshing["linear_mesh"]["min_quality"] >= DEFAULT_QUALITY_THRESHOLD
    # Route A has no film walls and the outline rule is off: every sized curve is a band edge.
    assert meshing["linear_mesh"]["sized_curves"] > 0
