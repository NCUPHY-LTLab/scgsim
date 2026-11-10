"""The cases of the band-edge table (issue 136): how each model is built once, and meshed with a recipe.

Every case is built once (:func:`build_case`) into its own folder; every candidate then meshes the same XAO
and sidecar with one ``mesh_sgb`` call (:func:`mesh_built`): the threshold 0.01, the default repair loop and
budgets, 3D Delaunay, linear elements, one Gmsh thread.  This is the setup of PR 132.

* The pad-in-ground chips of the PR 132 benchmark (1, 2 and 4 mm), from the test input builders, with and
  without bands.
* The OrcaPen SC PDK layouts: the flip-chip Disk Qubit as PR 132 assembled it, with and without bands, and
  SimplePad on Route A with bands.  Their geometry is the one ``EigenmodeSim.mesh()`` builds before it
  meshes (its ``_build_geometry`` step, the path ``simulation_drc`` uses too).  ``mesh()`` itself cannot
  be the harness: after a clean mesh it lowers the mesh for Palace, and on the Disk Qubit of PR 132 that
  step refuses the lumped-port sheet ``P1`` as non-conformal, so no recipe would get a row.

Every case has the vacuum of PR 132 (XY 0, +-Z 1000 um from the geometry bounds) except
``simplepad_A_bands_nb``, which has the (250, 250, 1000) um plan vacuum of the PDK notebooks, as in the Palace
run of issue 84 that found the Route A failure.

The PDK is imported at module scope: this module is loaded only by the runs, which need it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import gdsfactory as gf
import gdstk
import orcapen_sc_pdk
from orcapen_sc_pdk import (
    get_interface_preset_records,
    get_material_records,
    validate_interface_preset_records,
)
from orcapen_sc_pdk.helpers.assembly import author_common_flip_chip_ground_planes
from orcapen_sc_pdk.tech import LAYER_STACK_FLIP_CHIP, get_single_die_layer_stack

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY / "tests")]

from sgb_inputs import EffectiveInput, gds_pad_in_ground  # noqa: E402
from test_sgb_band_labels import PORT_CASES, all_interface_insets  # noqa: E402

from scgsim.geometry import ApproximationProfile, GeometryPlan  # noqa: E402
from scgsim.geometry.models.input import GdsBoundaryReconstructionSpec  # noqa: E402
from scgsim.geometry.sgb.build import build_sgb_geometry  # noqa: E402
from scgsim.meshing import MeshRepairError, SizingRecipe, mesh_sgb  # noqa: E402
from scgsim.palace import EigenmodeSim  # noqa: E402

# Decision C (quality measure and threshold) is the Human's; 0.01 is the provisional bound of run.py and
# the Palace default.
THRESHOLD = 0.01
# Surface-EPR bands (inset margins) on MA, MS and SA, as in PR 132 and issue 84.
MARGINS_UM = (0.5, 1.0)
INTERFACE_KINDS = ("MA", "MS", "SA")  # metal-air, metal-substrate, substrate-air
PRESET_PREFIX = "LiteratureCentral"
# The vacuum of PR 132: no lateral padding, 1000 um below and above the geometry.
VACUUM_UM = {"z_minus_um": 1000.0, "z_plus_um": 1000.0}
# The plan vacuum of the PDK's single-die Palace notebooks, used by the issue 84 Palace run.
NOTEBOOK_VACUUM_UM = (250.0, 250.0, 1000.0)
PORT_INDUCTANCE_H = 10e-9
PAD_RADIUS_UM = 10.0
CHIP_DIES_UM = {"B_1mm": 1000.0, "circB_2mm": 2000.0, "circB_4mm": 4000.0}
DISK_SITE_BOUNDS_UM = (-500.0, -500.0, 500.0, 500.0)  # one Disk Qubit site


def build_case(name: str, directory: Path) -> None:
    """Build the XAO and sidecar of case ``name`` into the empty ``directory`` and record their paths."""
    if name in CHIP_CASES:
        xao, sidecar = _build_chip(name, directory)
    else:
        sim = LAYOUT_CASES[name]()
        # The geometry half of mesh() (see the module docstring for why mesh() is not called).
        built = sim._build_geometry(directory)  # noqa: SLF001
        xao, sidecar = built.artifacts.xao, built.artifacts.sidecar
    record = {
        "xao": str(Path(xao).relative_to(directory)),
        "sidecar": str(Path(sidecar).relative_to(directory)),
    }
    (directory / "case.json").write_text(json.dumps(record) + "\n", encoding="utf-8")


def case_geometry(directory: Path) -> tuple[Path, Path]:
    """The XAO and sidecar :func:`build_case` wrote into ``directory``."""
    record = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    return directory / record["xao"], directory / record["sidecar"]


def mesh_built(
    xao: Path, sidecar: Path, recipe: SizingRecipe, directory: Path
) -> dict[str, Any]:
    """The quality report of one ``mesh_sgb`` call with ``recipe``; a rejected mesh is a result, not an error."""
    try:
        result = mesh_sgb(xao, sidecar, directory, threshold=THRESHOLD, recipe=recipe)
    except MeshRepairError as failure:
        return dict(failure.report)
    return dict(result.report)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, to show which geometry a run meshed."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- the chips


def _build_chip(name: str, directory: Path) -> tuple[Path, Path]:
    die_um = CHIP_DIES_UM[name.removesuffix("_bands").removesuffix("_nobands")]
    effective = _chip_input(directory, die_um, circular=name.startswith("circ"))
    insets = (
        all_interface_insets(effective, "B", PORT_CASES["B"][1])
        if name.endswith("_bands")
        else ()
    )
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route="B",
        insets=insets,
        output_dir=directory,
    )
    return artifacts.xao, artifacts.sidecar


def _chip_input(directory: Path, die_um: float, *, circular: bool) -> EffectiveInput:
    """The pad-in-ground input of the PR 132 benchmark; the circular chips rebuild the pad as an exact circle."""
    centre = (die_um / 2, die_um / 2)
    if not circular:
        return gds_pad_in_ground(directory, die_um=die_um, vacuum_padding=VACUUM_UM)
    effective = gds_pad_in_ground(
        directory,
        die_um=die_um,
        vacuum_padding=VACUUM_UM,
        pad_polygon=gdstk.ellipse(centre, PAD_RADIUS_UM, tolerance=0.01),
    )
    (polygon,) = [p for p in effective.build_input.polygons if p.object_name == "pad"]
    count = len(polygon.exterior)
    reconstruction = GdsBoundaryReconstructionSpec(
        "pad_circle",
        "pad",
        centre,
        "circular_arc",
        closed=True,
        segment_indices=tuple(range(count)),
        anchor_indices=(0, count // 3, 2 * count // 3),
    )
    curved = replace(effective.build_input, boundary_reconstruction=(reconstruction,))
    return EffectiveInput(curved, effective.representation)


CHIP_CASES = tuple(
    f"{chip}_{bands}" for chip in CHIP_DIES_UM for bands in ("nobands", "bands")
)


# ---------------------------------------------------------------------------------- the PDK layouts


def _surface_epr_specs(*, bands: bool) -> dict[str, dict[str, Any]]:
    """The MA, MS and SA film specs of the PDK presets; with ``bands`` each carries the margins 0.5 and 1 um."""
    records = validate_interface_preset_records(get_interface_preset_records())
    specs = {}
    for kind in INTERFACE_KINDS:
        preset = records[f"{PRESET_PREFIX}_{kind}"]
        specs[kind] = {
            key: preset[key]
            for key in ("thickness", "permittivity", "loss_tangent", "source")
        }
        if bands:
            specs[kind]["inset_margins_um"] = list(MARGINS_UM)
    return specs


def _disk_qubit(
    *, bands: bool, plane_bounds_um: tuple[float, ...] | None = None
) -> Callable[[], EigenmodeSim]:
    def build() -> EigenmodeSim:
        """The flip-chip Disk Qubit of PR 132: one Ground net, every authored port at 10 nH, Route B.

        ``plane_bounds_um`` bounds the common ground planes; ``None`` lets the PDK take the layout's extent,
        as the PR 132 pin does (it has no bounds argument).
        """
        orcapen_sc_pdk.activate()
        gf.clear_cache()
        qubit = gf.get_component(
            "disk_flip_chip_qubit",
            include_ground=False,
            junction_lead_span=0.0,
            tc_grounded_sides=(),
            curve_sampling="simulation",
        )
        parent = gf.Component()
        device = parent << qubit
        device.name = "device"
        parent.add_ports(device.ports)
        bumps = set(qubit.info.get("bump_instance_names", ()))
        instances = {"chip/device": device}
        instances |= {
            f"chip/device/{ref.name}": ref for ref in qubit.insts if ref.name in bumps
        }
        if plane_bounds_um is None:  # the PR 132 pin has no bounds argument
            author_common_flip_chip_ground_planes(parent)
        else:
            author_common_flip_chip_ground_planes(
                parent, plane_bounds_um=plane_bounds_um
            )
        plan = _plan(parent, LAYER_STACK_FLIP_CHIP)
        for path, reference in instances.items():
            plan.register_instance(path, reference)
        ground = ("chip/D0_TOP_GROUND_PLANE", "chip/D1_BOTTOM_GROUND_PLANE")
        bump_nets = tuple(
            f"{path}/D0_D1_INDIUM_BUMP" for path in instances if path != "chip/device"
        )
        plan.set_nets({"Ground": (*ground, *bump_nets)})
        snapshot = plan.prepare()
        sim = EigenmodeSim()
        sim.set_plan(snapshot)
        sim.set_vacuum_region(VACUUM_UM)
        for port in snapshot.stack["metadata"].get("port_sheet_source_layers", ()):
            sim.add_port(
                port["name"],
                layer=port["target_layer"],
                layout_sheet=True,
                inductance=PORT_INDUCTANCE_H,
            )
        sim.set_surface_epr(representation="B", specs=_surface_epr_specs(bands=bands))
        return sim

    return build


def _simplepad_route_a(*, notebook_vacuum: bool) -> Callable[[], EigenmodeSim]:
    def build() -> EigenmodeSim:
        """SimplePad of issue 84 on Route A (pad and ground lowered to sheets) with bands 0.5 and 1 um."""
        orcapen_sc_pdk.activate()
        gf.clear_cache()
        component = gf.get_component(
            "square_pad_capacitor",
            pad_side_um=200.0,
            gap_um=20.0,
            die_side_um=1000.0,
            junction_width_um=2.0,
        )
        plan = _plan(component, get_single_die_layer_stack())
        plan.set_nets({"pad": ("chip/PAD",), "ground": ("chip/GROUND",)})
        if notebook_vacuum:
            plan.set_vacuum_region(padding=NOTEBOOK_VACUUM_UM)
        sim = EigenmodeSim()
        sim.set_plan(plan.prepare())
        if not notebook_vacuum:
            sim.set_vacuum_region(VACUUM_UM)
        sim.add_port(
            "chip/o_junction_lumped",
            layer="D0_TOP_M1",
            layout_sheet=True,
            inductance=PORT_INDUCTANCE_H,
        )
        # Route A lowers the pad and ground to sheets; the thin-film profile says so.
        sim.set_surface_epr(
            representation="A",
            specs=_surface_epr_specs(bands=True),
            approximation_profile=ApproximationProfile("thin_film", "v1", {}),
        )
        return sim

    return build


def _plan(component: Any, layer_stack: Any) -> GeometryPlan:
    return GeometryPlan(
        component=component,
        root_id="chip",
        layer_stack=layer_stack,
        material_records=get_material_records(),
        coupon_padding_um=0.0,
    )


LAYOUT_CASES: dict[str, Callable[[], EigenmodeSim]] = {
    "disk_nobands": _disk_qubit(bands=False),
    "disk_bands": _disk_qubit(bands=True),
    # The same at the repository's current PDK pin (49e539f), which refuses planes that a ground mask
    # reaches and asks for explicit bounds: the site extent, as orcapen-sc-pdk#64 builds include_ground=False.
    "disk_site_nobands": _disk_qubit(bands=False, plane_bounds_um=DISK_SITE_BOUNDS_UM),
    "disk_site_bands": _disk_qubit(bands=True, plane_bounds_um=DISK_SITE_BOUNDS_UM),
    "simplepad_A_bands": _simplepad_route_a(notebook_vacuum=False),
    "simplepad_A_bands_nb": _simplepad_route_a(notebook_vacuum=True),
}
CASES = (*LAYOUT_CASES, *CHIP_CASES)
