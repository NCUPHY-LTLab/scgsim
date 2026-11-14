"""Simulations at the meshed stage over a real Route B SGB build, for the convergence declaration tests.

The groups and the mesh manifest come from SGB's own sidecar of the pad-in-ground layout, banded at the
``MA`` and ``MS`` margins 0.5 and 1.0 um (``SA`` whole); ``palace.msh`` is a stand-in file, because
``write_config()`` and ``prepare_handoff()`` read only its name and bytes.  No Gmsh and no solver run.
"""

from __future__ import annotations

import json
from pathlib import Path

from palace_sgb_support import MATERIALS, effective_input, sidecar_of, surface_epr_specs

from scgsim.contracts.palace_run_folder import MESH, MESH_MANIFEST, route_provenance
from scgsim.geometry.surface_epr.insets import inset_requests
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.mesh.build import MeshBuildResult
from scgsim.palace.mesh.manifest import build_mesh_manifest
from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.simulation.lifecycle import Stage
from scgsim.palace.source.ports import LayoutPortBinding
from scgsim.palace.source.surface_epr import SurfaceEprSpecs

MARGINS = {"MA": [0.5, 1.0], "MS": [0.5, 1.0], "SA": None}
PORT = LayoutPortBinding(1, "P1", "metal", "9/0", (1.0, 0.0, 0.0), {"L": 1e-9})
MESHING = {"recipe": {"name": "feature", "version": "1"}, "statistics": {}}


def _banded_build(directory: Path) -> tuple[dict, dict]:
    """The banded sidecar of the Route B pad in ground, and its groups."""
    directory.mkdir(parents=True, exist_ok=True)
    effective = effective_input("B", directory)
    unbanded = sidecar_of(effective, "B", directory)
    specs = surface_epr_specs(**MARGINS)
    sidecar = sidecar_of(
        effective, "B", directory / "banded", inset_requests(specs, unbanded)
    )
    return sidecar, groups_from_sidecar(sidecar)


def meshed(sim, directory: Path, *, surface_epr: bool = True):
    """``sim`` as ``mesh()`` leaves it on the banded build, with a run folder ``directory / "run"``."""
    sidecar, groups = _banded_build(directory)
    run = directory / "run"
    (run / "metadata").mkdir(parents=True)
    (run / MESH).write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    manifest = build_mesh_manifest(
        groups=groups,
        sidecar=sidecar,
        sidecar_file={"path": "geometry/sidecar.json", "sha256": "0" * 64},
        mesh_name=MESH,
        route="B",
        route_a_thin_film=None,
        meshing=MESHING,
    )
    (run / MESH_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    (run / route_provenance("B")).write_text('{"route": "B"}\n', encoding="utf-8")
    sim.output_dir = run
    sim._materials = MATERIALS  # noqa: SLF001 -- stands for set_plan()
    if surface_epr:
        sim.surface_epr_specs = SurfaceEprSpecs.normalized(
            {
                kind: {
                    "film_thickness_m": 2e-9,
                    "permittivity": 4,
                    "loss_tangent": 0.0,
                    **({"inset_margins_um": margins} if margins else {}),
                }
                for kind, margins in MARGINS.items()
            },
            materials=MATERIALS,
        )
    if isinstance(sim, EigenmodeSim):
        sim._resolved_ports = [PORT]  # noqa: SLF001 -- stands for mesh()
    sim._mesh_result = MeshBuildResult(  # noqa: SLF001 -- stands for mesh()
        output_dir=run,
        mesh_path=run / MESH,
        groups=groups,
        mesh_manifest_path=run / MESH_MANIFEST,
        provenance_path=run / route_provenance("B"),
        stack_path=run / "geometry/design.stack.json",
        xao_path=run / "geometry.xao",
        sidecar_path=run / "geometry/sidecar.json",
        quality_report_path=run / "mesh/quality.json",
    )
    sim._stage = Stage.MESHED  # noqa: SLF001 -- stands for set_plan() and mesh()
    return sim


def electrostatic(directory: Path, *, surface_epr: bool = True) -> ElectrostaticSim:
    """A meshed Electrostatic simulation with terminals ``pad`` (Net P) and ``plane`` (Net G), in that order."""
    sim = meshed(ElectrostaticSim(), directory, surface_epr=surface_epr)
    sim.add_terminal("pad", net_id="P")
    sim.add_terminal("plane", net_id="G")
    return sim


def eigenmode(directory: Path, *, num_modes: int = 2) -> EigenmodeSim:
    """A meshed Eigenmode simulation with port ``P1`` and ``num_modes`` modes."""
    sim = meshed(EigenmodeSim(), directory)
    sim.set_eigenmode(num_modes=num_modes, target=5e9)
    return sim


# ---- Pass tables in Palace's formats ---------------------------------------------------------------------


def _matrix(path: Path, prefix: str, unit: str, matrix: list[list[float]]) -> None:
    columns = ",".join(f"{prefix}{j}] ({unit})" for j in range(1, len(matrix) + 1))
    rows = "".join(
        f"{i},{','.join(f'{value:+.12e}' for value in row)}\n"
        for i, row in enumerate(matrix, start=1)
    )
    path.write_text(f"i,{columns}\n{rows}", encoding="utf-8")


def _surface_table(
    path: Path, index: str, rows: int, surfaces: int, number: int
) -> None:
    """``surface-Q.csv``: each participation grows by a factor 1 + 0.01 / pass, so its change settles."""
    headers = "".join(f",p_surf[{k}],Q_surf[{k}]" for k in range(1, surfaces + 1))
    path.write_text(
        f"{index}{headers}\n"
        + "".join(
            f"{row}"
            + "".join(
                f",{1e-3 * k * (1 + 0.01 / number) * row:.12e},+inf"
                for k in range(1, surfaces + 1)
            )
            + "\n"
            for row in range(1, rows + 1)
        ),
        encoding="utf-8",
    )


def _domain_table(path: Path, index: str, rows: int, domains: int) -> None:
    headers = "".join(
        f",E_elec[{k}] (J),p_elec[{k}],E_mag[{k}] (J),p_mag[{k}]"
        for k in range(1, domains + 1)
    )
    cells = ",1e-20,0.5,0.0,0.0" * domains
    path.write_text(
        f"{index},E_elec (J),E_mag (J),E_cap (J),E_ind (J){headers}\n"
        + "".join(f"{row},1e-20,0.0,0.0,0.0{cells}\n" for row in range(1, rows + 1)),
        encoding="utf-8",
    )


def _indicators(path: Path, norm: float) -> None:
    path.write_text(
        f"Norm,Minimum,Maximum,Mean\n{norm},1e-9,{norm},{norm / 10}\n",
        encoding="utf-8",
    )


def write_palace_json(folder: Path, number: int) -> None:
    """The ``palace.json`` Palace writes beside the tables of pass ``number`` (1000 unknowns per pass)."""
    memory = {"Average": 10, "Max": 10, "Min": 10, "Total": 10}
    (folder / "palace.json").write_text(
        json.dumps(
            {
                "Problem": {
                    "DegreesOfFreedom": 1000 * number,
                    "MeshElements": 500 * number,
                    "MPISize": 1,
                    "OpenMPThreads": 1,
                    "Iteration": number,
                },
                "ElapsedTime": {
                    "Counts": {"Total": 1},
                    "Durations": {"Total": 2.0 * number, "Solve": 1.0 * number},
                },
                "PeakMemoryMegabytes": memory,
                "PeakNodeMemoryMegabytes": memory,
                "LinearSolver": {"TotalSolves": 2},
                "GitTag": "v0.18.1",
            }
        ),
        encoding="utf-8",
    )


def write_electrostatic_pass(
    folder: Path,
    number: int,
    counts: dict[str, int],
    diagonal_ff: tuple[float, float],
    norm: float,
) -> None:
    """Every Electrostatic table of pass ``number`` for terminals ``pad`` and ``plane``, in Palace's headers."""
    folder.mkdir(parents=True, exist_ok=True)
    c11, c22 = (value * 1e-15 for value in diagonal_ff)
    c12 = -5e-15
    maxwell = [[c11, c12], [c12, c22]]
    _matrix(folder / "terminal-C.csv", "C[i][", "F", maxwell)
    _matrix(folder / "terminal-Cm.csv", "C_m[i][", "F", maxwell)
    _matrix(
        folder / "terminal-Cinv.csv", "C⁻¹[i][", "1/F", [[1e13, 1e11], [1e11, 3e12]]
    )
    (folder / "terminal-V.csv").write_text(
        "i,V_inc[i] (V)\n1,1.0\n2,1.0\n", encoding="utf-8"
    )
    _domain_table(folder / "domain-E.csv", "i", 2, counts["domain"])
    _surface_table(folder / "surface-Q.csv", "i", 2, counts["surface"], number)
    _indicators(folder / "error-indicators.csv", norm)
    write_palace_json(folder, number)


def write_eigenmode_pass(
    folder: Path,
    number: int,
    counts: dict[str, int],
    frequencies_ghz: tuple[float, ...],
    norm: float,
) -> None:
    """Every Eigenmode table of pass ``number`` for the lumped ports, one row per mode, in Palace's headers."""
    folder.mkdir(parents=True, exist_ok=True)
    modes = len(frequencies_ghz)
    (folder / "eig.csv").write_text(
        "m,Re{f} (GHz),Im{f} (GHz),Q,Error (Bkwd.),Error (Abs.)\n"
        + "".join(
            f"{m},{f:+.12e},{f * 1e-6:+.12e},{5e5:+.12e},1e-12,1e-12\n"
            for m, f in enumerate(frequencies_ghz, start=1)
        ),
        encoding="utf-8",
    )
    ports = counts["lumped_port"]
    (folder / "port-EPR.csv").write_text(
        "m"
        + "".join(f",p[{k}]" for k in range(1, ports + 1))
        + "\n"
        + "".join(f"{m}" + ",0.9" * ports + "\n" for m in range(1, modes + 1)),
        encoding="utf-8",
    )
    for kind, unit in (("I", "A"), ("V", "V")):
        (folder / f"port-{kind}.csv").write_text(
            "m"
            + "".join(
                f",Re{{{kind}[{k}]}} ({unit}),Im{{{kind}[{k}]}} ({unit})"
                for k in range(1, ports + 1)
            )
            + "\n"
            + "".join(f"{m}" + ",1e-3,0.0" * ports + "\n" for m in range(1, modes + 1)),
            encoding="utf-8",
        )
    _domain_table(folder / "domain-E.csv", "m", modes, counts["domain"])
    _surface_table(folder / "surface-Q.csv", "m", modes, counts["surface"], number)
    _indicators(folder / "error-indicators.csv", norm)
    write_palace_json(folder, number)
