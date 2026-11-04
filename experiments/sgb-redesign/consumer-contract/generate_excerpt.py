"""Build a tiny SGB model with the sgb-core package and write its five files (source of the consumer-contract excerpts).

Run with the sgb-core environment:
  <sgb-core>/.artifacts/env/bin/python generate_excerpt.py <sgb-core>/tests <out_dir>
It uses the sgb-core test helper ``sgb_inputs.gds_pad_in_ground`` (100 um die, ground plane, 20 um pad, one lumped port).
"""
import sys
import tempfile
from pathlib import Path

tests, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(tests))

from sgb_inputs import gds_pad_in_ground, thin_film_pair  # noqa: E402

from scgsim.geometry.sgb import build_sgb_geometry  # noqa: E402
from scgsim.geometry.sgb.cutfamilies.requests import (  # noqa: E402
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
)

with tempfile.TemporaryDirectory() as tmp:
    effective = gds_pad_in_ground(Path(tmp))
    for route in "BC":
        artifacts = build_sgb_geometry(
            effective.build_input,
            effective.representation,
            route=route,
            insets=(InsetRequest("pad_top", ConductorSurface("pad", "top"), (0.5, 1.0)),),
            refinements=(RefinementRequest("ref", 0.2, ((45.0, 45.0), (55.0, 45.0))),),
            output_dir=out / f"route_{route.lower()}",
        )
        print(route, artifacts.counts)


# Route A: two Thin Film sheets joined by a lumped port (the sheet group carries the MA and the MS row)
films = thin_film_pair()
artifacts = build_sgb_geometry(
    films.build_input,
    films.representation,
    route="A",
    insets=(InsetRequest("band_left", ConductorSurface("chip/left", "sheet"), (0.5, 1.0)),),
    output_dir=out / "route_a",
)
print("A", artifacts.counts)

# ---------------------------------------------------------------- what a consumer then does with the files
import json  # noqa: E402

import gmsh  # noqa: E402

from scgsim.geometry.sgb.exchange.step import export_aedt_step  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.verification.loaded import (  # noqa: E402
    apply_sidecar_embeds,
    verify_loaded,
)

runs: dict[str, object] = {}
for route in "BC":
    folder = out / f"route_{route.lower()}"
    sidecar = read_sidecar(folder / f"semantic_geometry_route_{route.lower()}.sidecar.json")
    # the Palace-side sequence: open, synchronize, verify, embed, mesh
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
    gmsh.open(str(folder / f"semantic_geometry_route_{route.lower()}.xao"))
    gmsh.model.occ.synchronize()
    report = verify_loaded(sidecar)
    embedded = apply_sidecar_embeds(sidecar)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 60.0)
    gmsh.model.mesh.generate(3)
    mesh_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    # the AEDT-side export and its guard
    step = export_aedt_step(
        folder / f"semantic_geometry_route_{route.lower()}.brep", sidecar, folder / "aedt_model.step"
    )
    runs[route] = {
        "verify_loaded": {
            "ok": report.ok,
            "counts": report.counts,
            "max_vertex_offset_um": report.max_vertex_offset_um,
            "boundaries_mismatching": report.boundaries_mismatching,
            "max_relative_face_area_error": report.max_relative_face_area_error,
            "max_relative_volume_error": report.max_relative_volume_error,
            "groups": report.groups,
        },
        "apply_sidecar_embeds": embedded,
        "mesh_nodes": mesh_nodes,
        "step_export": {
            "unit": step.unit,
            "object_names": sorted(step.object_names),
            "faces_per_solid": {s.aedt_name: len(s.faces) for s in step.solids},
            "first_guard_rows": {s.aedt_name: s.faces[:2] for s in step.solids},
        },
    }
(out / "consumer_runs.json").write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n")
print(json.dumps({r: {k: v for k, v in d.items() if k != "step_export"} for r, d in runs.items()}, indent=1, sort_keys=True)[:3000])
