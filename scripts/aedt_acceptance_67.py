"""The Human's acceptance run of issue #67: an HFSS Eigenmode EPR project built from the SGB STEP, then solved.

The model is the Route C realization of a ``solid`` request (issue #67 decision 7): two pads on a silicon
substrate, one junction across their gap with its Lumped RLC, and Surface EPR for ``MA``, ``MS`` and
``SA`` with inset bands.  Everything goes through the public flow: ``GeometryPlan``, ``EigenmodeSim``, the
handoff's ``--prepare-only`` run, the prepared-native batch, ``collect_native_batch`` and ``analyze_epr``.

Two commands, one at a time, each into the same scratch folder outside the repository:

``build OUT``
    Prepares the handoff, runs it with ``--prepare-only`` (AEDT imports the STEP, assigns materials, PEC,
    the Region, its outer boundary, the junction's Lumped RLC and the Surface-EPR selections, creates the
    setup and saves the project, without solving), prints where the project is and what to inspect in the
    GUI, and opens the saved project in a graphical AEDT session it leaves open.  With ``--close`` it opens
    no session and leaves no AEDT behind.

``solve OUT``
    Packages the saved project, solves it with the AEDT batch solver, collects the adaptive EPR result,
    then integrates the saved field twice with ``analyze_epr``: every Surface-EPR group read on its faces
    (``surface``), and the same with one group read just off them (``adjacent``).  It prints the
    frequencies and that group's rows from both.  Close the inspection session first.

Example::

    python scripts/aedt_acceptance_67.py build /scratch/acceptance-67 --cores 4 --ram-limit-percent 50
    python scripts/aedt_acceptance_67.py solve /scratch/acceptance-67 --cores 4 --ram-limit-percent 50 \\
        --solver-executable /path/to/AnsysEM/v242/Linux64/ansysedt

It needs ``scgsim[aedt-epr]``, gdsfactory, AEDT 2024.2 and a license on this host.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scgsim.aedt import (
    AedtResources,
    EigenmodeRunControl,
    EigenmodeSim,
    EprAnalysisRequest,
    EprResult,
    HandoffPlan,
    HfssEprSpec,
    PlanarJunction,
    analyze_epr,
    collect_native_batch,
    parse_aedt_spec,
    planar_junction_from_port,
    prepare_native_batch,
    resolve_saved_solution,
    run_native_batch,
)
from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry import (
    GeometryPlan,
    GeometryPlanSnapshot,
    JunctionTerminalBinding,
    RectangularJunctionSpec,
)
from scgsim.geometry.models.physical import PhysicalLayerDeclaration

AEDT_VERSION = "2024.2"
PROJECT_NAME = "acceptance_67"
DESIGN_NAME = "PadPairEpr"
JUNCTION_ID = "jj"
PORT_NAME = "chip/J1"

# The acceptance geometry, in um: two 100 um square pads 20 um apart, and a 10 um wide junction sheet
# reaching 5 um into each pad.
PAD_A = ((0, 0), (100, 0), (100, 100), (0, 100))
PAD_B = ((120, 0), (220, 0), (220, 100), (120, 100))
JUNCTION_SHEET = ((95, 45), (125, 45), (125, 55), (95, 55))
VACUUM_PADDING_UM = (100.0, 100.0, 300.0)
MATERIALS = {
    "PEC": {"kind": "conductor"},
    "vacuum": {"kind": "vacuum"},
    "Si": {"kind": "dielectric", "permittivity": 11.45},
}
FILM = {"film_thickness_m": 3e-9, "loss_tangent": 1e-3}
SURFACE_EPR_SPECS = {
    "MA": {**FILM, "permittivity": 10.0, "inset_margins_um": [2.0]},
    "MS": {**FILM, "permittivity": 11.4, "inset_margins_um": [2.0]},
    "SA": {**FILM, "permittivity": 4.0, "inset_margins_um": [2.0]},
}
RUN_CONTROL = EigenmodeRunControl(
    setup_name="Setup1",
    minimum_frequency_ghz=1.0,
    num_modes=1,
    maximum_passes=3,
    maximum_delta_frequency_percent=1.0,
)
# Mode 1, every Surface-EPR group, the junction and every bulk domain.
EPR_REQUEST = EprAnalysisRequest(
    mode_indices=(1,),
    surface_epr_groups=None,
    junction_ids=(JUNCTION_ID,),
    bulk_domain_ids=None,
)
INSPECTION = (
    "the object list against the guard's solids (model/model.step.guard.json): names, PEC on the pads, "
    "Si on the substrate",
    "the Region box and its material against the plan's VACUUM_REGION (model/ sidecar) and its vacuum",
    "the outer boundary: Perfect H on the six Region faces",
    "the junction's Lumped RLC on its port faces, with its current line",
    "one selection sheet per Surface-EPR row, on the faces the row names",
    "the setup: Setup1, its minimum frequency, modes and passes",
)


def main(argv: list[str] | None = None) -> int:
    """Run the ``build`` or ``solve`` command."""
    args = _arguments(argv)
    out = Path(args.out).expanduser().resolve()
    if Path(__file__).resolve().parents[1] in (out, *out.parents):
        raise SystemExit(
            f"{out} is inside the repository; give a scratch folder outside it"
        )
    resources = AedtResources(args.cores, args.ram_limit_percent)
    if args.command == "build":
        build(out, resources, close=args.close)
    else:
        solve(
            out,
            resources,
            solver_executable=args.solver_executable,
            group=args.adjacent_group,
        )
    return 0


def build(out: Path, resources: AedtResources, *, close: bool) -> None:
    """Prepare, import and set up the project, save it, say what to inspect, and leave it open unless ``close``."""
    snapshot, junction = _plan()
    handoff = _handoff(out, snapshot, junction, resources)
    # The handoff's script runs ``python``: this interpreter, the one with SCGSim and PyAEDT.
    path = os.pathsep.join(
        (str(Path(sys.executable).parent), os.environ.get("PATH", ""))
    )
    subprocess.run(
        ["bash", str(handoff.script_path), "--prepare-only"],
        check=True,
        env={**os.environ, "PATH": path},
    )
    project = handoff.run_dir / f"{PROJECT_NAME}.aedt"
    _print_inspection(handoff, project)
    if not close:
        _open_for_inspection(project)


def solve(
    out: Path, resources: AedtResources, *, solver_executable: str, group: str | None
) -> None:
    """Solve the saved project, collect its EPR, and integrate it again with one group ``adjacent``."""
    run_dir = out / "run"
    package = prepare_native_batch(
        run_dir,
        output_dir=out / "package",
        solver_executable=solver_executable,
        resources=resources,
    )
    attempt = out / "attempt"
    run_native_batch(package.root, output_dir=attempt)
    collected = out / "collected"
    run = collect_native_batch(package.root, attempt, output_dir=collected)
    adaptive = run.epr_result()
    if adaptive is None:
        raise RuntimeError(f"the collected run {collected} has no EPR result")
    print("adaptive EPR:", json.dumps(_frequencies(adaptive), indent=2))
    spec = parse_aedt_spec(
        json.loads((collected / "aedt_spec.json").read_text(encoding="utf-8")),
        base_dir=collected,
    )
    if not isinstance(spec, HfssEprSpec):
        raise TypeError(
            f"{collected} holds a {type(spec).__name__}, not an HFSS EPR request"
        )
    group = group or _first_sa_group(spec.model.path("sidecar"))
    saved = resolve_saved_solution(collected / "saved-solution-manifest.json")
    results = {
        selector: analyze_epr(
            saved_solution=saved,
            aedt_version=AEDT_VERSION,
            model=spec.model,
            project_name=spec.project_name,
            design_name=spec.design_name,
            run_control=RUN_CONTROL,
            epr_request=EPR_REQUEST,
            output_dir=out / f"epr-{selector}",
            surface_field_selectors=None
            if selector == "surface"
            else {group: "adjacent"},
        )
        for selector in ("surface", "adjacent")
    }
    for selector, result in results.items():
        print(f"{selector}:", json.dumps(_frequencies(result), indent=2))
    print(f"group {group}, surface then adjacent:")
    print(json.dumps({s: _group_rows(r, group) for s, r in results.items()}, indent=2))


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser(
        "build", help="prepare, import, set up and save the project"
    )
    build_command.add_argument(
        "out", help="scratch folder outside the repository; must not exist"
    )
    build_command.add_argument("--cores", type=int, required=True)
    build_command.add_argument("--ram-limit-percent", type=int, required=True)
    build_command.add_argument(
        "--close",
        action="store_true",
        help="close AEDT when done instead of leaving the project open",
    )
    solve_command = commands.add_parser(
        "solve", help="solve the saved project and analyze its EPR"
    )
    solve_command.add_argument("out", help="the folder build wrote")
    solve_command.add_argument("--cores", type=int, required=True)
    solve_command.add_argument("--ram-limit-percent", type=int, required=True)
    solve_command.add_argument(
        "--solver-executable",
        required=True,
        help="the AEDT executable the batch solve runs",
    )
    solve_command.add_argument(
        "--adjacent-group",
        help="the Surface-EPR group read adjacent; the first SA group when omitted",
    )
    return parser.parse_args(argv)


def _plan() -> tuple[GeometryPlanSnapshot, PlanarJunction]:
    """The plan of the pad pair with its junction declared, and the junction bound to its port sheet."""
    plan = GeometryPlan(
        _component(),
        root_id="chip",
        layer_stack=_layer_stack(),
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_nets({"A": ("chip/pad_a",), "B": ("chip/pad_b",)})
    plan.set_vacuum_region(VACUUM_PADDING_UM)
    (region,) = plan.prepare().geometry_input.port_sheet_regions
    hosts = {
        relation["host_entity_id"].split("/")[1].rsplit("_", 1)[0]: relation[
            "host_entity_id"
        ]
        for relation in region.metadata["geometry_grid_contact_relations"]
    }
    plan.set_rectangular_junctions(
        (
            RectangularJunctionSpec(
                junction_id=JUNCTION_ID,
                port_sheet_id=region.port_sheet_id,
                source_polygon_id=region.source_polygon_id,
                source_occurrence_path="chip",
                terminal_a=JunctionTerminalBinding(hosts["pad_a"], "chip", "A"),
                terminal_b=JunctionTerminalBinding(hosts["pad_b"], "chip", "B"),
            ),
        )
    )
    snapshot = plan.prepare()
    junction = planar_junction_from_port(
        snapshot.geometry_input,
        port_name=PORT_NAME,
        junction_id=JUNCTION_ID,
        terminal_a_net="A",
        terminal_b_net="B",
        inductance_h=10e-9,
        capacitance_f=0.0,
    )
    return snapshot, junction


def _handoff(
    out: Path,
    snapshot: GeometryPlanSnapshot,
    junction: PlanarJunction,
    resources: AedtResources,
) -> HandoffPlan:
    """The HFSS Eigenmode EPR handoff of ``snapshot`` on the Route C model, into ``out/run``."""
    sim = EigenmodeSim(modeling="solid", aedt_version=AEDT_VERSION)
    sim.set_plan(snapshot)
    sim.set_output_dir(out / "run")
    sim.set_eigenmode(
        project_name=PROJECT_NAME,
        design_name=DESIGN_NAME,
        run_control=RUN_CONTROL,
        resources=resources,
    )
    sim.set_surface_epr(SURFACE_EPR_SPECS)
    sim.add_junction(junction)
    sim.set_epr_request(EPR_REQUEST)
    return sim.prepare_handoff()


def _component() -> Any:
    import gdsfactory as gf  # noqa: PLC0415 -- the layout library, needed only to draw the acceptance model

    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(PAD_A, layer=(1, 0))
    component.add_polygon(PAD_B, layer=(2, 0))
    component.add_polygon(JUNCTION_SHEET, layer=(9, 0))
    component.add_port(
        name="J1", center=(110, 50), width=10, orientation=0, layer=(9, 0)
    )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "pad_a", "level": "metal", "gds_layer": [1, 0]},
            {"semantic_id": "pad_b", "level": "metal", "gds_layer": [2, 0]},
        ],
        "ports": [{"name": "J1", "layer": [9, 0], "target_layer": "metal"}],
    }
    return component


def _layer_stack() -> SimpleNamespace:
    """A silicon substrate 100 um thick under a 0.2 um PEC film attached by its lower face."""
    return SimpleNamespace(
        layers={
            "substrate": _level("substrate", "Si", (-100.0, 0.0), "solution_region"),
            "metal": _level("metal", "PEC", (0.0, 0.2), "conductor"),
        }
    )


def _level(
    name: str, material: str, interval: tuple[float, float], role: str
) -> SimpleNamespace:
    film = role == "conductor"
    span = interval[1] - interval[0]
    declaration = PhysicalLayerDeclaration(
        name,
        name,
        {"level": name},
        interval,
        material,
        "stack_film" if film else "finite_volume",
        {"normal_thickness_um": span} if film else {"axial_span_um": span},
        {"layer_attachment_face": "lower"} if film else {},
        {},
    )
    return SimpleNamespace(
        material=material,
        zmin=interval[0],
        thickness=span,
        mesh_order=1,
        info={
            "simulation_role": role,
            "include_in_component_simulation": True,
            "part_role": "face_metal",
            "physical_layer": declaration.to_payload(),
        },
    )


def _print_inspection(handoff: HandoffPlan, project: Path) -> None:
    receipt = json.loads(handoff.receipt_path.read_text(encoding="utf-8"))
    print(f"project: {project}")
    print(f"receipt: {handoff.receipt_path} (status {receipt['status']})")
    print("inspect in the AEDT GUI:")
    for item in INSPECTION:
        print(f"  - {item}")


def _open_for_inspection(project: Path) -> None:
    """Open ``project`` in a new graphical AEDT session and leave that session running."""
    from ansys.aedt.core import Hfss  # noqa: PLC0415 -- AEDT automation, needed only for the GUI session

    app = Hfss(
        project=str(project),
        design=DESIGN_NAME,
        version=AEDT_VERSION,
        non_graphical=False,
        new_desktop=True,
        close_on_exit=False,
    )
    print(
        f"AEDT is open with {project.name} (gRPC port {app.desktop_class.port}); close it when done"
    )
    app.release_desktop(close_projects=False, close_desktop=False)


def _first_sa_group(sidecar_path: Path) -> str:
    rows = read_sidecar(sidecar_path)["surface_epr_row"].values()
    return min(row["group"] for row in rows if row["interface_type"] == "SA")


def _frequencies(result: EprResult) -> list[dict[str, Any]]:
    return [
        {
            "mode": row["mode"],
            "native_pass": row.get("native_pass"),
            "frequency_hz": row.get("frequency_hz"),
        }
        for row in result.rows
    ]


def _group_rows(result: EprResult, group: str) -> list[dict[str, Any]]:
    """Each Surface-EPR contribution with a member of ``group``: its integrals, energy and participation."""
    return [
        {
            "mode": row["mode"],
            "contribution": surface["group_id"],
            "normal_integral_v2": surface["normal_integral_v2"],
            "tangential_integral_v2": surface["tangential_integral_v2"],
            "energy_j": surface["energy_j"],
            "participation": surface["participation"],
        }
        for row in result.rows
        for surface in row["surface_contributions"]
        if any(member["group"] == group for member in surface["members"])
    ]


if __name__ == "__main__":
    sys.exit(main())
