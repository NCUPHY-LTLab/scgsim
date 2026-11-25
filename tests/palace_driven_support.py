# ruff: noqa: PLC0415 -- gdsfactory and the PDK are imported where the layout is built: the PDK is not in SCGSim's own lock, and the public suite collects this module without it
"""The launcher-to-launcher plan a Driven test gives ``DrivenSim``, and the Gmsh stand-in the config tests use.

The layout is the roster's (``scripts/roster_cases.py``): two OrcaPen SC PDK ``launcher`` cells joined by a
900 um ``cpw_6_7_6`` straight on a 1870 x 900 um die, its two port sheets landing 10 grid steps into the pad
and the ground, declared as 50 ohm ports to ground, left first.  The nightly roster check meshes the same
layout, and the Palace runs recorded on issue 187 and in ``experiments/palace-driven`` solved it.

A ``pdk`` test needs the PDK environment of CI's pdk-tests job.

:func:`mesh_without_gmsh` stands in for the one stage of ``mesh()`` the Driven config tests leave out, Gmsh:
``mesh()`` still runs its preparation, the SGB build and the meshing-stage check for real, and
``write_config()`` builds from the physical groups of that real SGB model, the groups the mesher would carry.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import write_json
from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.palace.mesh.build import MeshBuildResult, RouteGeometry
from scgsim.palace.mesh.controls import MeshControls
from scgsim.palace.mesh.manifest import build_mesh_manifest
from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.simulation import core

# The roster lives in scripts/; CI's pdk-tests job runs `python -I`, which puts neither it nor the checkout on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from roster_cases import (  # noqa: E402
    Sense,
    geometry_plan,
    launcher_pair,
    surface_epr_specs,
)


def launcher_plan(
    *,
    right_sense: Sense = "pdk",
    surface_epr: bool = False,
    bands: bool = False,
    lumped_elements: Any = None,
) -> Any:
    """The roster's launcher-to-launcher ``GeometryPlan``: Nets, vacuum, and the two sheets as 50 ohm ports.

    The ports are declared left then right, so the left launcher is port 1.  ``right_sense`` is the sense the
    right sheet is drawn in; ``surface_epr`` sets the PDK's literature-central MA, MS and SA presets, cut as
    bands at 0.5 and 1.0 um with ``bands``; ``lumped_elements`` replaces the two ports.
    """
    from orcapen_sc_pdk.materials import get_material_records

    layout = launcher_pair(right_sense=right_sense)
    if lumped_elements is not None:
        layout = replace(layout, lumped_elements=lumped_elements)
    specs = surface_epr_specs(bands=bands) if surface_epr else None
    return geometry_plan(layout, get_material_records(), specs)


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
