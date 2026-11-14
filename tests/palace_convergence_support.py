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
