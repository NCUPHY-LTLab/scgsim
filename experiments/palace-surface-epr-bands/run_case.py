"""Run one roster layout with Surface-EPR inset bands through the public ``EigenmodeSim`` API (issue #84).

The script owns the four checks of #84 for one case: it builds the layout from the public OrcaPen SC PDK,
runs ``simulation_drc``, meshes, writes the Palace config, prepares a ``direct-local`` handoff, runs the
handoff's own script, reads the result back and writes ``report.json`` with what each check found.  It
records numbers and judges nothing: no threshold here is an acceptance value.

Stages, each a subcommand, run in order on one output folder:

* ``prepare``  ``simulation_drc``, ``mesh()``, ``write_config()`` and ``prepare_handoff()``.
* ``execute``  ``bash run_palace.sh``, the script the handoff wrote, with a wall-time limit.
* ``readback`` ``resolve_palace_result``, ``epr_result()`` and ``show_epr()``, then the four checks.

The Palace wrapper and the folder holding the MPI launcher are given by ``--palace`` and ``--mpi-bin``
(or ``PALACE_EXECUTABLE`` and ``PALACE_MPI_BIN``); nothing host-specific lives in this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gdsfactory as gf
import orcapen_sc_pdk
from orcapen_sc_pdk import (
    get_interface_preset_records,
    get_material_records,
    validate_interface_preset_records,
)
from orcapen_sc_pdk.tech import get_single_die_layer_stack

from scgsim.geometry import ApproximationProfile, GeometryPlan
from scgsim.meshing import UniformBoundaryRecipe
from scgsim.palace import EigenmodeSim, resolve_palace_result, show_epr

# MA: metal-air, MS: metal-substrate, SA: substrate-air interface films.
INTERFACE_KINDS = ("MA", "MS", "SA")
MARGINS_UM = (0.5, 1.0)
PRESET_PREFIX = "LiteratureCentral"
PORT_INDUCTANCE_H = 10e-9
VACUUM_PADDING_UM = (250.0, 250.0, 1000.0)
# The PDK's flip-chip Palace notebooks pad the coupon by 1000 um in z only.
FLIP_CHIP_VACUUM_UM = (0.0, 0.0, 1000.0)
# The flip-chip cases are built as the #149 baseline builds them, with simulation sampling of curves.
CURVE_SAMPLING = "simulation"
# Pipeline settings from #84: mode 1, order 2, a few AMR passes.  They check the pipeline, not convergence.
NUM_MODES = 1
SEARCH_FREQUENCY_HZ = 1e9
EIGENMODE_TOLERANCE = 1e-6
FEM_ORDER = 2
AMR_MAX_PASSES = 2
AMR_TOLERANCE = 0.02
AMR_UPDATE_FRACTION = 0.7
WALL_LIMIT_S = 30 * 60
# How long the stopped run gets to end after SIGTERM before it is killed.
TERMINATION_GRACE_S = 60
DIELECTRIC_SECTION = "Boundaries.Postprocessing.Dielectric"


@dataclass
class Case:
    """One roster layout: its component, stack, Nets and authored lumped ports."""

    component: Any
    layer_stack: Any
    nets: dict[str, tuple[str, ...]]
    ports: tuple[str, ...]
    port_layer: str = "D0_TOP_M1"
    # Child occurrences that carry their own semantics, keyed by plan path, as GeometryPlan.register_instance takes them.
    instances: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    # The plan vacuum of the PDK's Palace notebooks: (x, y, z) padding in um.
    vacuum_padding_um: tuple[float, float, float] = VACUUM_PADDING_UM
    # Bind every port the plan's layout authors (as simulation_drc does) instead of ``ports``.
    all_authored_ports: bool = False


def main() -> None:
    """Parse the stage and run it on one case and route."""
    args = _parse_arguments()
    output = Path(args.output_dir).resolve()
    if args.stage == "prepare":
        prepare(args.case, args.route, output, args)
    elif args.stage == "execute":
        execute(output, args.wall_limit_s)
    else:
        readback(output)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("prepare", "execute", "readback"))
    parser.add_argument("output_dir")
    parser.add_argument(
        "--case",
        choices=(
            "simplepad",
            "circular_pad",
            "airbridge",
            "flip_chip_pad",
            "disk_qubit",
        ),
        default="simplepad",
    )
    parser.add_argument(
        "--wall-limit-s",
        type=float,
        default=WALL_LIMIT_S,
        help="execute: stop the Palace run after this many seconds",
    )
    parser.add_argument("--route", choices=("A", "B"), default="B")
    parser.add_argument("--palace", default=os.environ.get("PALACE_EXECUTABLE"))
    parser.add_argument("--mpi-bin", default=os.environ.get("PALACE_MPI_BIN"))
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument(
        "--mesh-recipe",
        choices=("default", "uniform-boundary"),
        default="default",
        help="default: the shared FeatureRecipe(); uniform-boundary: UniformBoundaryRecipe(), a diagnostic only",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- prepare


def prepare(case_name: str, route: str, output: Path, args: argparse.Namespace) -> None:
    """Check the layout, mesh it with bands, write the config and package the handoff."""
    if not args.palace or not args.mpi_bin:
        raise SystemExit(
            "--palace and --mpi-bin (or PALACE_EXECUTABLE, PALACE_MPI_BIN) are required"
        )
    output.mkdir(parents=True, exist_ok=True)
    case = build_case(case_name)
    material_records = get_material_records()
    specs = surface_epr_specs()
    record: dict[str, Any] = {"case": case_name, "route": route, "notes": case.notes}
    record["simulation_drc"] = _timed_drc(case, material_records, specs)
    sim = configure_simulation(case, route, material_records, specs, output / "run")
    if args.mesh_recipe == "uniform-boundary":
        sim.set_mesh(recipe=UniformBoundaryRecipe())
    record["mesh_recipe"] = args.mesh_recipe
    record["mesh"] = _timed_mesh(sim)
    if record["mesh"]["status"] == "ok":
        sim.write_config()
        handoff = sim.prepare_handoff(
            profile="direct-local",
            executable=args.palace,
            resources={
                "processes": args.processes,
                "threads": 1,
                "command_style": "wrapper",
            },
            # The stock Palace wrapper finds its MPI launcher on PATH.
            setup_commands=(f'export PATH={shlex.quote(args.mpi_bin)}:"$PATH"',),
        )
        record["handoff"] = {
            "run_dir": str(handoff.run_dir),
            "handoff_id": handoff.handoff_id,
            "script": str(handoff.script_path),
            "processes": args.processes,
        }
    _write(output / "prepare.json", record)
    print(json.dumps(_prepare_digest(record), indent=2, default=str))


def _prepare_digest(record: dict[str, Any]) -> dict[str, Any]:
    """The few numbers of a prepare record worth printing."""
    mesh = record["mesh"]
    digest = {"simulation_drc": record["simulation_drc"], "mesh_status": mesh["status"]}
    if mesh["status"] == "ok":
        meshing = mesh["summary"]["meshing"]
        digest |= {
            "mesh_seconds": mesh["seconds"],
            "rounds": meshing["rounds"],
            "stop": meshing["stop"],
            "tetrahedra": meshing["linear_mesh"]["tetrahedra"],
            "min_sicn": meshing["linear_mesh"]["min_quality"],
        }
    else:
        digest["error"] = mesh["error"]
    return digest


def build_case(name: str) -> Case:
    """Build one roster layout from the public PDK with its final Nets."""
    orcapen_sc_pdk.activate()
    gf.clear_cache()
    if name == "simplepad":
        component = gf.get_component(
            "square_pad_capacitor",
            pad_side_um=200.0,
            gap_um=20.0,
            die_side_um=1000.0,
            junction_width_um=2.0,
        )
        return Case(
            component,
            get_single_die_layer_stack(),
            {"pad": ("chip/PAD",), "ground": ("chip/GROUND",)},
            ("chip/o_junction_lumped",),
        )
    if name == "circular_pad":
        component = gf.get_component(
            "circular_pad_capacitor",
            outer_radius_um=100.0,
            inner_radius_um=0.0,
            gap_um=20.0,
            die_side_um=1000.0,
            junction_width_um=2.0,
        )
        return Case(
            component,
            get_single_die_layer_stack(),
            {"pad": ("chip/PAD",), "ground": ("chip/GROUND",)},
            ("chip/o_junction_lumped",),
        )
    if name == "flip_chip_pad":
        return flip_chip_pad_case()
    if name == "disk_qubit":
        return disk_qubit_case()
    component = gf.get_component("cpw_airbridge_coupon")
    stack = get_single_die_layer_stack(include_airbridges=True)
    instances = {
        f"chip/{reference.name}": reference
        for reference in component.insts
        if reference.name == "AIRBRIDGE"
    }
    entities = _entity_ids(component, stack, instances)
    ground = tuple(e for e in entities if e != "chip/SIGNAL")
    return Case(
        component,
        stack,
        {"signal": ("chip/SIGNAL",), "ground": ground},
        (),
        instances=instances,
        notes={"entities": entities, "ports": "the coupon authors no port"},
    )


def flip_chip_pad_case() -> Case:
    """The flip-chip pad coupon and its four ground bumps, built as the #149 baseline builds it.

    One Ground net holds the lower ground, the upper plane and every bump.  The baseline gives this coupon
    no port (the lower die's locator is not forwarded), so none is bound here either.
    """
    # Imported here for the same reason as in disk_qubit_case: the flip-chip cases run at a later PDK pin.
    from orcapen_sc_pdk.tech import get_two_die_flip_chip_layer_stack  # noqa: PLC0415

    component = gf.get_component("flip_chip_pad_coupon", curve_sampling=CURVE_SAMPLING)
    references = {reference.name: reference for reference in component.insts}
    bumps = ("BUMP_MM", "BUMP_MP", "BUMP_PM", "BUMP_PP")
    nets = {
        "pad": ("chip/LOWER_PAD_DIE/PAD",),
        "ground": (
            "chip/LOWER_PAD_DIE/GROUND",
            "chip/UPPER_GROUND",
            *(f"chip/{name}/D0_D1_INDIUM_BUMP" for name in bumps),
        ),
    }
    return Case(
        component,
        get_two_die_flip_chip_layer_stack(),
        nets,
        (),
        instances={
            f"chip/{name}": references[name] for name in ("LOWER_PAD_DIE", *bumps)
        },
        notes={"ports": "none, as in the #149 baseline"},
        vacuum_padding_um=FLIP_CHIP_VACUUM_UM,
    )


def disk_qubit_case() -> Case:
    """``disk_flip_chip_qubit`` at its defaults as ``chip/device`` under the common ground planes (#149).

    Every authored port is bound with the 10 nH inductance, as ``simulation_drc`` and the baseline bind them.
    """
    # Imported here: the common ground planes exist only at PDK revisions holding orcapen-sc-pdk#64, and the
    # single-die cases still run at the earlier pin.
    from orcapen_sc_pdk.helpers.assembly import author_common_flip_chip_ground_planes  # noqa: PLC0415
    from orcapen_sc_pdk.tech import LAYER_STACK_FLIP_CHIP  # noqa: PLC0415 -- same reason as the line above

    qubit = gf.get_component("disk_flip_chip_qubit", curve_sampling=CURVE_SAMPLING)
    parent = gf.Component()
    device = parent << qubit
    device.name = "device"
    parent.add_ports(device.ports)
    author_common_flip_chip_ground_planes(parent, plane_bounds_um=None)
    references = {reference.name: reference for reference in qubit.insts}
    bumps = sorted(qubit.info["bump_instance_names"])
    nets = {
        "Ground": (
            "chip/D0_TOP_GROUND_PLANE",
            "chip/D1_BOTTOM_GROUND_PLANE",
            *(f"chip/device/{name}/D0_D1_INDIUM_BUMP" for name in bumps),
        )
    }
    return Case(
        parent,
        LAYER_STACK_FLIP_CHIP,
        nets,
        (),
        instances={
            "chip/device": device,
            **{f"chip/device/{name}": references[name] for name in bumps},
        },
        notes={"bumps": bumps},
        vacuum_padding_um=FLIP_CHIP_VACUUM_UM,
        all_authored_ports=True,
    )


def _entity_ids(
    component: Any, stack: Any, instances: dict[str, Any]
) -> tuple[str, ...]:
    """The conductor Entity IDs a plan of ``component`` lists, for assigning Nets."""
    plan = GeometryPlan(
        component=component,
        root_id="chip",
        layer_stack=stack,
        material_records=get_material_records(),
        coupon_padding_um=0.0,
    )
    for path, reference in instances.items():
        plan.register_instance(path, reference)
    return tuple(
        entity.semantic_id
        for entity in plan.entities
        if getattr(entity, "entity_kind", "conductor") == "conductor"
    )


def surface_epr_specs() -> dict[str, dict[str, Any]]:
    """The MA, MS and SA film specs from the PDK presets, each with margins 0.5 and 1 um."""
    records = validate_interface_preset_records(get_interface_preset_records())
    specs = {}
    for kind in INTERFACE_KINDS:
        preset_name = f"{PRESET_PREFIX}_{kind}"
        preset = records[preset_name]
        specs[kind] = {
            "thickness": preset["thickness"],
            "permittivity": preset["permittivity"],
            "loss_tangent": preset["loss_tangent"],
            "source": preset["source"],
            "preset": preset_name,
            "inset_margins_um": list(MARGINS_UM),
        }
    return specs


def _timed_drc(case: Case, material_records: Any, specs: Any) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        EigenmodeSim.simulation_drc(
            case.component,
            layer_stack=case.layer_stack,
            material_records=material_records,
            instances=case.instances or None,
            nets=case.nets,
            surface_epr_specs=specs,
        )
    except Exception as error:  # noqa: BLE001 -- the check's own error is the recorded result
        return {
            "status": "refused",
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - start,
        }
    return {"status": "pass", "seconds": time.perf_counter() - start}


def configure_simulation(
    case: Case, route: str, material_records: Any, specs: Any, run_dir: Path
) -> EigenmodeSim:
    """An ``EigenmodeSim`` for the case: plan, bands, ports, mode 1, order 2 and a few AMR passes."""
    plan = GeometryPlan(
        component=case.component,
        root_id="chip",
        layer_stack=case.layer_stack,
        material_records=material_records,
        coupon_padding_um=0.0,
    )
    for path, reference in case.instances.items():
        plan.register_instance(path, reference)
    plan.set_nets(case.nets)
    plan.set_vacuum_region(padding=case.vacuum_padding_um)
    snapshot = plan.prepare()
    sim = EigenmodeSim()
    sim.set_plan(snapshot)
    sim.set_output_dir(run_dir)
    if route == "A":
        # Route A lowers the pad and Ground to sheets; the thin-film profile says so (see the PDK test of Route A).
        sim.set_route(
            "A", approximation_profile=ApproximationProfile("thin_film", "v1", {})
        )
    sim.set_surface_epr(specs)
    ports = [(port, case.port_layer) for port in case.ports]
    if case.all_authored_ports:
        ports = [
            (record["name"], record["target_layer"])
            for record in snapshot.stack["metadata"].get("port_sheet_source_layers", ())
        ]
    for port, layer in ports:
        sim.add_port(port, layer=layer, layout_sheet=True, inductance=PORT_INDUCTANCE_H)
    sim.set_epr_request(
        surface_interfaces=INTERFACE_KINDS,
        bulk_domain_ids=None,
        port_names=None,
    )
    sim.set_eigenmode(
        num_modes=NUM_MODES,
        target=SEARCH_FREQUENCY_HZ,
        tolerance=EIGENMODE_TOLERANCE,
        save=0,
    )
    sim.set_numerical(
        order=FEM_ORDER,
        tolerance=EIGENMODE_TOLERANCE,
        max_iterations=2000,
        amr_max_passes=AMR_MAX_PASSES,
        amr_tolerance=AMR_TOLERANCE,
        amr_update_fraction=AMR_UPDATE_FRACTION,
        output_paraview=False,
        output_grid_function=False,
    )
    return sim


def _timed_mesh(sim: EigenmodeSim) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        mesh_path = sim.mesh()
    except Exception as error:  # noqa: BLE001 -- the stage's own error is the recorded result
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - start,
        }
    # The summary carries the meshing audit: minSICN, bad elements, repair rounds and stop.
    return {
        "status": "ok",
        "seconds": time.perf_counter() - start,
        "mesh_path": str(mesh_path),
        "summary": sim.mesh_summary(),
    }


# --------------------------------------------------------------------------- execute


def execute(output: Path, wall_limit_s: float) -> None:
    """Run the handoff's own script with the wall-time limit and record the outcome."""
    handoff = json.loads((output / "prepare.json").read_text(encoding="utf-8"))[
        "handoff"
    ]
    start = time.perf_counter()
    # A session of its own, so a timeout stops mpirun and every Palace rank with the script.
    process = subprocess.Popen(
        ["bash", handoff["script"]], cwd=handoff["run_dir"], start_new_session=True
    )
    try:
        record: dict[str, Any] = {"exit_code": process.wait(timeout=wall_limit_s)}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=TERMINATION_GRACE_S)
        except subprocess.TimeoutExpired:
            # A wrapper or rank that ignores SIGTERM must not hold the run past its limit.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record = {"exit_code": None, "error": f"stopped after {wall_limit_s} s"}
    record["wall_limit_s"] = wall_limit_s
    record["wall_seconds"] = time.perf_counter() - start
    _write(output / "execute.json", record)
    print(json.dumps(record, indent=2))


# --------------------------------------------------------------------------- readback


def readback(output: Path) -> None:
    """Resolve the returned run, draw the EPR rows and write the four checks to ``report.json``."""
    prepared = json.loads((output / "prepare.json").read_text(encoding="utf-8"))
    run_dir = Path(prepared["handoff"]["run_dir"])
    report: dict[str, Any] = {
        "geometry": geometry_check(run_dir),
        "configuration": configuration_check(run_dir),
        "palace_output": palace_output_check(run_dir),
    }
    try:
        report["readback"] = readback_check(run_dir, prepared["handoff"]["handoff_id"])
    except Exception as error:  # noqa: BLE001 -- the reader's own refusal is the recorded result
        report["readback"] = {"error": f"{type(error).__name__}: {error}"}
    _write(output / "report.json", report)
    print(json.dumps(report, indent=2, default=str))


def geometry_check(run_dir: Path) -> dict[str, Any]:
    """Band groups per banded face, the faces SGB skipped with their reasons, and every surface group.

    ``surface_groups`` lists each MA, MS and SA physical group with its face kind and band, so a face of a
    requested type that is neither banded nor skipped shows up as a group with ``band`` null and no
    matching ``banded`` or ``skipped`` entry.
    """
    (sidecar_path,) = sorted((run_dir / "geometry").glob("*.sidecar.json"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    requests = []
    for inset in sidecar["insets"]:
        expansion = inset["expansion"]
        requests.append(
            {
                "request_id": inset["request_id"],
                "margins_um": inset["margins_um"],
                "banded": [
                    {
                        "support": item["support"],
                        "band_count": item["band_count"],
                        "support_area_um2": item["support_area_um2"],
                        "bands": item["bands"],
                    }
                    for item in expansion["banded"]
                ],
                "skipped": expansion.get("skipped", []),
            }
        )
    surface_groups = {
        name: {
            "tag": group["tag"],
            "face_kind": group.get("face_kind"),
            "band": group.get("band"),
            "area_um2": group.get("area_um2"),
            "entities": group.get("entities"),
        }
        for name, group in sidecar["physical_groups"].items()
        if group.get("interface_type") in INTERFACE_KINDS
    }
    return {
        "sidecar": sidecar_path.name,
        "requests": requests,
        "surface_groups": surface_groups,
        "slivers": sidecar.get("slivers"),
    }


def configuration_check(run_dir: Path) -> dict[str, Any]:
    """The Dielectric rows of the config against the index-map entries of that section.

    A row whose ``band`` is null in every index-map row it covers is a whole-surface row; one with bands
    is a margin row, whose margin is the lowest band it integrates (band ``k`` lies beyond margin ``k``).
    """
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = config["Boundaries"]["Postprocessing"]["Dielectric"]
    index_map = json.loads(
        (run_dir / "metadata" / "palace_index_map.json").read_text(encoding="utf-8")
    )
    entries = {
        entry["index"]: entry
        for entry in index_map["entries"]
        if entry["section"] == DIELECTRIC_SECTION
    }
    summary = []
    for row in rows:
        entry = entries.get(row["Index"])
        summary.append(
            {
                "index": row["Index"],
                "type": row["Type"],
                "attributes": row["Attributes"],
                "entry_name": None if entry is None else entry["entry_name"],
                "index_map_attributes": None if entry is None else entry["attributes"],
                "attributes_match": entry is not None
                and sorted(entry["attributes"]) == sorted(row["Attributes"]),
                "groups": None
                if entry is None
                else [item["group"] for item in entry["metadata"]["rows"]],
                "bands": None
                if entry is None
                else sorted(
                    {item["band"] for item in entry["metadata"]["rows"]}, key=str
                ),
            }
        )
    return {
        "dielectric_rows": rows,
        "index_map_schema": index_map.get("schema_version"),
        "rows": summary,
        "config_indices": sorted(row["Index"] for row in rows),
        "index_map_indices": sorted(entries),
        "crack_internal_boundary_elements": config["Model"].get(
            "CrackInternalBoundaryElements"
        ),
    }


def palace_output_check(run_dir: Path) -> dict[str, Any]:
    """The columns of every ``surface-Q.csv`` Palace wrote, as indices, with the eigenfrequency table."""
    tables = sorted((run_dir / "results").rglob("surface-Q.csv"))
    eig = sorted((run_dir / "results").rglob("eig.csv"))
    return {
        "surface_q": {
            str(path.relative_to(run_dir)): {
                "header": (header := _csv_header(path)),
                "indices": sorted(
                    {
                        int(match)
                        for cell in header
                        for match in re.findall(r"\[(\d+)\]", cell)
                    }
                ),
            }
            for path in tables
        },
        "eig": {str(path.relative_to(run_dir)): _csv_rows(path) for path in eig},
    }


def readback_check(run_dir: Path, handoff_id: str) -> dict[str, Any]:
    """Resolve the run, list every surface row with its participation and loss, and what ``show_epr`` draws."""
    resolved = resolve_palace_result(run_dir, expected_handoff_id=handoff_id)
    result = resolved.epr_result()
    rows = [_plain(row) for row in result.rows]
    surfaces = [
        {
            "mode": row["mode"],
            "index": item["index"],
            "baseline_index": item["baseline_index"],
            "interface_kind": item["interface_kind"],
            "name": item["name"],
            "evaluation_kind": item["evaluation_kind"],
            "margin_um": item.get("margin_um"),
            "bands": item.get("bands"),
            "participation": item["participation"],
            "inverse_q": item["inverse_q"],
            "finite": _finite(item["participation"]) and _finite(item["inverse_q"]),
        }
        for row in rows
        for item in row["surface"]
    ]
    figure = show_epr(result, mode=1)
    return {
        "modes": [
            {"mode": row["mode"], "frequency_hz": row["frequency_hz"]} for row in rows
        ],
        "surfaces": surfaces,
        "bulk": [item for row in rows for item in row["bulk"]],
        "junction": [item for row in rows for item in row["junction"]],
        "drawn": _drawn_categories(figure),
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _drawn_categories(figure: Any) -> list[dict[str, Any]]:
    """Every trace of the figure with the category labels it draws."""
    traces = []
    for trace in figure.data:
        labels = trace.y if getattr(trace, "orientation", None) == "h" else trace.x
        traces.append(
            {"name": trace.name, "type": trace.type, "labels": _plain(labels)}
        )
    return traces


# --------------------------------------------------------------------------- helpers


def _csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [cell.strip() for cell in next(csv.reader(handle))]


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [[cell.strip() for cell in row] for row in csv.reader(handle)]


def _plain(value: Any) -> Any:
    """A JSON-friendly copy of a report object."""
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _plain(getattr(value, name)) for name in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
