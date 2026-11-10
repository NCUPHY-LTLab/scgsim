"""Build a tiny SGB model with SGB and write its five files (source of the consumer-contract excerpts).

Run from the repository root (gmsh comes with the ``palace`` extra, OCP with the ``occt`` group):
  uv run --extra palace --group dev --group occt python experiments/sgb-redesign/consumer-contract/generate_excerpt.py \
      tests experiments/sgb-redesign/consumer-contract/out
It uses the test helper ``sgb_inputs.gds_pad_in_ground`` (100 um die, ground plane, 20 um pad, one lumped port).
"""

import sys
import tempfile
from pathlib import Path

tests, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(tests))

from sgb_gmsh_oracle import mesh_quality, mesh_with_cheap_recipe  # noqa: E402
from sgb_inputs import gds_pad_in_ground, thin_film_pair  # noqa: E402

from scgsim.geometry.sgb import build_sgb_geometry  # noqa: E402
from scgsim.geometry.sgb.cutfamilies.requests import (  # noqa: E402
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
    SolutionInterface,
)

with tempfile.TemporaryDirectory() as tmp:
    effective = gds_pad_in_ground(Path(tmp))
    for route in "BC":
        artifacts = build_sgb_geometry(
            effective.build_input,
            effective.representation,
            route=route,
            insets=(
                InsetRequest("pad_top", ConductorSurface("pad", "top"), (0.5, 1.0)),
            ),
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
    insets=(
        InsetRequest("band_left", ConductorSurface("chip/left", "sheet"), (0.5, 1.0)),
    ),
    output_dir=out / "route_a",
)
print("A", artifacts.counts)

# ---------------------------------------------------------------- what a consumer then does with the files
import json  # noqa: E402

import gmsh  # noqa: E402

from scgsim.geometry.sgb.exchange.guard import read_step_guard  # noqa: E402
from scgsim.geometry.sgb.exchange.step import export_aedt_step  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.verification.loaded import (  # noqa: E402
    apply_sidecar_embeds,
    verify_loaded,
)

runs: dict[str, object] = {}
for route in "BC":
    folder = out / f"route_{route.lower()}"
    sidecar = read_sidecar(
        folder / f"semantic_geometry_route_{route.lower()}.sidecar.json"
    )
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
        folder / f"semantic_geometry_route_{route.lower()}.brep",
        folder / f"semantic_geometry_route_{route.lower()}.sidecar.json",
        folder / "aedt_model.step",
    )
    guard = read_step_guard(step.guard_path)  # what the importer does first
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
            "guard_file": step.guard_path.name,
            "faces_per_solid": {
                s["aedt_name"]: len(s["faces"]) for s in guard["solids"]
            },
            "first_guard_rows": {
                s["aedt_name"]: s["faces"][:2] for s in guard["solids"]
            },
            "free_faces": [f["face"] for f in guard["free_faces"]],
        },
    }
(out / "consumer_runs.json").write_text(
    json.dumps(runs, indent=1, sort_keys=True) + "\n"
)
print(
    json.dumps(
        {
            r: {k: v for k, v in d.items() if k != "step_export"}
            for r, d in runs.items()
        },
        indent=1,
        sort_keys=True,
    )[:3000]
)

# ---------------------------------------------------------------- every entity, every surface, plus SA
# One extra set in ``<out>/all_interfaces/route_{a,b,c}/``: each entity gets inset bands on every supported
# surface (finite conductors: top and bottom; Route A sheets: the sheet) and on the substrate-vacuum (SA)
# interface, margins (0.5, 1.0).  The lumped port lies on the hosts' support plane (the substrate top), so it
# occupies part of the SA plane: SA band faces exclude it while every band origin stays the metal edge.
ALL_MARGINS = (0.5, 1.0)


def conductor_ids(effective):
    """Semantic ids of the conductor entities of an effective input, in input order."""
    return [
        e.semantic_id
        for e in effective.build_input.entities
        if e.material_kind == "conductor"
    ]


def all_interface_insets(effective, route, sa_pair):
    """Bands on every supported surface of every conductor and on the SA interface at the port plane."""
    sides = ("sheet",) if route == "A" else ("top", "bottom")
    insets = [
        InsetRequest(
            f"band_{entity}_{side}".replace("/", "_"),
            ConductorSurface(entity, side),
            ALL_MARGINS,
        )
        for entity in conductor_ids(effective)
        for side in sides
    ]
    first, second = sa_pair
    insets.append(
        InsetRequest("band_sa", SolutionInterface(first, second, z_um=0.0), ALL_MARGINS)
    )
    return tuple(insets)


def summarise(folder, route):
    """Open an .xao with gmsh: counts, group names, the port plane, SA/PORT areas, and a mesh from the cheap recipe."""
    stem = f"semantic_geometry_route_{route.lower()}"
    sidecar = read_sidecar(folder / f"{stem}.sidecar.json")
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
    gmsh.open(str(folder / f"{stem}.xao"))
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(3)
    surfaces = gmsh.model.getEntities(2)
    groups = {
        gmsh.model.getPhysicalName(d, t): (d, t)
        for d, t in gmsh.model.getPhysicalGroups()
    }

    def group_area(name):
        dim, tag = groups[name]
        return sum(
            gmsh.model.occ.getMass(dim, e)
            for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        )

    # the facts come from the sidecar fields, not from parsing the group names
    sa_bands = {
        n: group_area(n)
        for n, g in sorted(sidecar["physical_groups"].items())
        if g["interface_type"] == "SA" and g["band"] is not None
    }
    port_group = sidecar["ports"][0]["group"]
    port_area = group_area(port_group)
    (port,) = sidecar["ports"]
    apply_sidecar_embeds(sidecar)
    algorithm = mesh_with_cheap_recipe(folder / f"{stem}.xao", sidecar)
    quality = mesh_quality()
    nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return {
        "xao": str(folder / f"{stem}.xao"),
        "volumes": len(volumes),
        "surfaces": len(surfaces),
        "physical_group_names": sorted(groups),
        "port_group": port_group,
        "port_z_um": port["z_um"],
        "port_adjacent_volumes": len(port["adjacent_volumes"]),
        "embedded_faces": len(sidecar["embedded_faces"]),
        "port_group_area_um2": port_area,
        "sa_band_areas_um2": sa_bands,
        "mesh_feature_kinds": sorted({f["kind"] for f in sidecar["mesh_features"]}),
        "mesh": {
            "recipe": "HXT, MeshSizeMax = bounding-box diagonal / 20, no size field",
            "algorithm3d_used": algorithm,
            "nodes": nodes,
            "tets": quality.get("tets"),
            "minSICN_min": quality.get("minSICN_min"),
        },
    }


all_dir = out / "all_interfaces"
summaries: dict[str, object] = {}
with tempfile.TemporaryDirectory() as tmp:
    pad = gds_pad_in_ground(Path(tmp))
    films = thin_film_pair()
    cases = {
        "B": (pad, ("substrate", "VACUUM_REGION")),
        "C": (pad, ("substrate", "VACUUM_REGION")),
        "A": (films, ("substrate", "upper_air")),
    }
    for route, (effective, sa_pair) in cases.items():
        folder = all_dir / f"route_{route.lower()}"
        artifacts = build_sgb_geometry(
            effective.build_input,
            effective.representation,
            route=route,
            insets=all_interface_insets(effective, route, sa_pair),
            output_dir=folder,
        )
        print("all_interfaces", route, artifacts.counts)
        summaries[route] = summarise(folder, route)
(all_dir / "summary.json").write_text(
    json.dumps(summaries, indent=1, sort_keys=True) + "\n"
)
print(json.dumps(summaries, indent=1, sort_keys=True))
