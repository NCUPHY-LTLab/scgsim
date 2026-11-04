"""Experiment E3: direct BREP/XAO serialisation instead of gmsh-API emission.

Default run: build the three running examples, write ``model.brep`` and ``model.xao`` with
``sgbproto.brep_writer``, load them back into Gmsh and verify (counts, boundaries, groups, volumes, mesh
conformity), try BREP -> STEP, and write everything under ``outputs/e3_brep/<case>/``.

``--timing``: write / load cost of the gmsh-API emission versus the direct writer on the random layout
generator (every N in its own subprocess; every load in a fresh Gmsh process) -> ``outputs/e3_brep/timing.json``.

Usage:  python examples/brep_emit.py [--timing [--ns 1000 5000 20000] [--workers 1 8 16]]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "e3_brep"
MARGINS = (0.1, 0.5, 2.0)  # inset band margins (um), as in run_inset_bands.py
REFINE = {"refine_families": ("band_pad_top", "band_gap"), "refine_size": 1.0, "refine_distance": (1.0, 20.0)}
CASES = {  # name -> (mesh size, mesh size min); the band cases use MeshSizeMin 0 (a 2 um minimum cannot resolve a 0.1 um band)
    "pad_in_ground_A": (40.0, 0.0),
    "pad_in_ground_B": (60.0, 2.0),
    "airbridge_B": (60.0, 2.0),
    "pad_in_ground_A_bands": (40.0, 0.0),
    "pad_in_ground_B_bands": (60.0, 0.0),
}


def case_spec(name: str):
    from sgbproto import scenes

    if name.startswith("pad_in_ground"):
        bands = MARGINS if name.endswith("_bands") else None
        return scenes.pad_in_ground(name.split("_")[3], bands_um=bands)
    return scenes.airbridge()


# --------------------------------------------------------------------------- default run: three cases


def load_model(path: Path, autofix: int = 0) -> None:
    import gmsh

    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCAutoFix", autofix)
    (gmsh.open if path.suffix == ".xao" else gmsh.merge)(str(path))
    gmsh.model.occ.synchronize()


def run_case(name: str) -> dict:
    import gmsh

    from sgbproto.brep_writer import verify_loaded, write_brep, write_xao
    from sgbproto.emit import emit
    from sgbproto.pipeline import build

    size, size_min = CASES[name]
    spec = case_spec(name)
    banded = name.endswith("_bands")
    mesh_kwargs = {"supports": {f.family_id: f.support for f in spec.families}, **REFINE} if banded else {}
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    topo = build(spec).topology
    summary: dict = {"case": name, "planned": topo.counts()}
    brep = write_brep(topo, out / "model.brep")
    xao = write_xao(topo, out / "model.xao", name="model")
    summary["files"] = {"model.brep": brep.bytes_written, "model.xao": xao.bytes_written}
    summary["plan_counts"] = brep.plan.counts
    summary["write_s"] = {"brep": round(brep.timings["total"], 4), "xao": round(xao.timings["total"], 4)}
    summary["write_s"]["sidecar"] = round(xao.timings["sidecar"], 4)
    for kind, result in (("brep", brep), ("xao", xao)):
        gmsh.initialize(["-noenv"])
        try:
            load_model(result.path)
            report = verify_loaded(topo, result.plan, mesh_size=size, mesh_size_min=size_min, groups=kind == "xao", **mesh_kwargs)
            if kind == "brep":
                summary["step"] = try_step(out)
        finally:
            gmsh.finalize()
        summary[f"verify_{kind}"] = report
        if name == "pad_in_ground_B":
            summary[f"analytic_volumes_{kind}"] = analytic_volume_check(topo, result.plan, report["volumes_um3"])
    # reference: the gmsh-API emission of the same topology, loaded and compared by group content
    reference = emit(topo, out / "_gmsh_api", "model")
    summary["group_content_vs_gmsh_api_xao"] = compare_group_content(out / "model.xao", reference.xao_path)
    summary["sidecar_vs_gmsh_api_sidecar"] = compare_sidecars(out / "model.sidecar.json", reference.sidecar_path)
    shutil.rmtree(out / "_gmsh_api")
    (out / "validation.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print(name, "ok" if summary["verify_brep"]["ok"] and summary["verify_xao"]["ok"] else "FAILED",
          summary["plan_counts"], summary["write_s"], summary["group_content_vs_gmsh_api_xao"]["equal"],
          summary["sidecar_vs_gmsh_api_sidecar"]["equal"])
    return summary


def compare_sidecars(direct: Path, oracle: Path) -> dict:
    """Groups, unions, embeds and curved-wall list of the direct sidecar against the gmsh-API sidecar."""
    mine, ref = json.loads(direct.read_text()), json.loads(oracle.read_text())
    strip = lambda groups: {n: (g["dim"], g["physical_tag"], g["ids"]) for n, g in groups.items()}
    checks = {
        "physical_groups": strip(mine["physical_groups"]) == strip(ref["physical_groups"]),
        "unions": mine["unions"] == ref["unions"],
        "embedded": mine["embedded"] == ref["embedded"],
        "curved_wall_faces": set(mine["curved_wall_faces"]) == set(ref["curved_wall_faces"]),
        "planned_ids_per_dimension": all(
            set(ids) == {e["id"] for e in ref["construction_order"] if e["dim"] == int(dim)}
            for dim, ids in mine["ids_by_tag"].items()),
    }
    return {"equal": all(checks.values()), **checks}


def analytic_volume_check(topo, plan, masses: list[float]) -> dict:
    """pad_in_ground_B closed form: chip 300 x 300 x 100; ground 0.2 thick; pad = polygon + corner segment - circle."""
    polygon = [(-45, -30), (-10, -30), (-10, -15), (10, -15), (10, -30), (50, -30), (50, 18), (38, 30), (-50, 30), (-50, -25)]
    area = sum(polygon[i][0] * polygon[(i + 1) % 10][1] - polygon[(i + 1) % 10][0] * polygon[i][1] for i in range(10)) / 2
    area += 25 * (math.pi / 2 - 1) / 2 - math.pi * 64  # 5 um rounded corner (outward bulge), r = 8 um hole
    expected = {"substrate": 300 * 300 * 100.0, "vacuum": 300 * 300 * 100.0 - (300 * 300 - 160 * 120) * 0.2 - area * 0.2}
    out = {}
    for vid, mass in zip(plan.solid_ids, masses):
        name = topo.volumes[vid].domain.name
        out[name] = {"expected": expected[name], "loaded": mass, "relative_error": abs(mass - expected[name]) / expected[name]}
    return out


def group_content(path: Path) -> dict:
    """Per physical group: dimension and the sorted bounding boxes of its entities (independent of tags and orientation)."""
    import gmsh

    gmsh.initialize(["-noenv"])
    try:
        load_model(path)
        out = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            members = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            boxes = sorted(tuple(round(x + 0.0, 3) for x in gmsh.model.getBoundingBox(dim, int(t))) for t in members)
            out[gmsh.model.getPhysicalName(dim, tag)] = (dim, len(members), boxes)
        return out
    finally:
        gmsh.finalize()


def compare_group_content(direct: Path, reference: Path) -> dict:
    a, b = group_content(direct), group_content(reference)
    return {"equal": a == b, "groups": len(a), "differences": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}


@contextlib.contextmanager
def _silenced_stdout():
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def try_step(out: Path) -> dict:
    """BREP -> STEP through OCCT (the file currently merged in Gmsh) and back: entity counts."""
    import gmsh

    path = out / "model.step"
    t0 = time.perf_counter()
    with _silenced_stdout():  # the OCCT STEP writer prints its transfer statistics to fd 1
        gmsh.write(str(path))
    seconds = time.perf_counter() - t0
    before = [len(gmsh.model.getEntities(d)) for d in range(4)]
    gmsh.model.remove()
    gmsh.model.add("step")
    with _silenced_stdout():
        gmsh.merge(str(path))
    gmsh.model.occ.synchronize()
    after = [len(gmsh.model.getEntities(d)) for d in range(4)]
    volume = [round(gmsh.model.occ.getMass(3, t), 6) for _, t in gmsh.model.getEntities(3)]
    return {"bytes": path.stat().st_size, "write_s": round(seconds, 4), "entities_before": before,
            "entities_after_reload": after, "volumes_after_reload": volume}


# --------------------------------------------------------------------------- timing run


def load_in_fresh_process(path: Path, autofix: int = 0) -> dict:
    """merge/open + synchronize in a new Python process (Gmsh start-up excluded)."""
    code = (
        "import sys,time,json,gmsh\n"
        "from pathlib import Path\n"
        "p=Path(sys.argv[1])\n"
        "gmsh.initialize(['-noenv']);gmsh.option.setNumber('General.Terminal',0)\n"
        f"gmsh.option.setNumber('Geometry.OCCAutoFix',{autofix})\n"
        "t=time.perf_counter();(gmsh.open if p.suffix=='.xao' else gmsh.merge)(str(p));a=time.perf_counter()-t\n"
        "t=time.perf_counter();gmsh.model.occ.synchronize();b=time.perf_counter()-t\n"
        "print(json.dumps({'read_s':a,'synchronize_s':b,'entities':[len(gmsh.model.getEntities(d)) for d in range(4)],"
        "'groups':len(gmsh.model.getPhysicalGroups())}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"}
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["load_s"] = row["read_s"] + row["synchronize_s"]
    return row


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()[:16]


def timing_child(n: int, workers: list[int]) -> dict:
    from sgbproto.brep_writer import write_brep, write_xao
    from sgbproto.emit import emit
    from sgbproto.pipeline import build
    from sgbproto.scenes import random_layout

    tmp = OUT / f"_tmp_n{n}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.perf_counter()
    topo = build(random_layout(n, seed=1, route="B")).topology
    row: dict = {"n_features": n, "build_s": round(time.perf_counter() - t0, 2), "counts": topo.counts(), "direct": {}}
    digests = set()
    for w in workers:  # direct writes first: the forked pool must not inherit an initialised Gmsh
        xao = write_xao(topo, tmp / f"direct_w{w}.xao", workers=w, name="model", sidecar=False)  # the XAO embeds its name
        digests.add(sha(xao.path))
        row["direct"][f"xao_w{w}"] = _timings(xao)
        if w != workers[0]:
            xao.path.unlink()
    row["direct"]["plan_counts"] = xao.plan.counts
    row["direct"]["xao_bytes"] = xao.bytes_written
    row["direct"]["xao_identical_for_every_worker_count"] = len(digests) == 1
    for w in (workers[0], workers[-1]):
        brep = write_brep(topo, tmp / f"direct_w{w}.brep", workers=w, sidecar=False)
        row["direct"][f"brep_w{w}"] = _timings(brep)
    row["direct"]["brep_bytes"] = brep.bytes_written
    nopc = write_brep(topo, tmp / "direct_nopc.brep", workers=workers[-1], pcurves=False, sidecar=False)
    row["direct"][f"brep_nopcurves_w{workers[-1]}"] = _timings(nopc)
    row["direct"]["brep_nopcurves_bytes"] = nopc.bytes_written
    t0 = time.perf_counter()
    emitted = emit(topo, tmp / "api", "model", write_xao=True, orientation_attempts=0)  # timing: no sewing (quadratic at scale)
    row["gmsh_api"] = {"total_s": round(time.perf_counter() - t0, 3),
                       "stages_s": {k: round(v, 3) for k, v in emitted.timings.items()},
                       "xao_bytes": emitted.xao_path.stat().st_size}
    row["load"] = {
        "gmsh_api_xao": load_in_fresh_process(emitted.xao_path),
        "direct_xao": load_in_fresh_process(tmp / f"direct_w{workers[0]}.xao"),
        "direct_brep": load_in_fresh_process(tmp / f"direct_w{workers[0]}.brep"),
        "direct_brep_nopcurves": load_in_fresh_process(tmp / "direct_nopc.brep"),
        "direct_xao_autofix_default": load_in_fresh_process(tmp / f"direct_w{workers[0]}.xao", autofix=1),
        "gmsh_api_xao_autofix_default": load_in_fresh_process(emitted.xao_path, autofix=1),
    }
    shutil.rmtree(tmp, ignore_errors=True)
    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return row


def _timings(result) -> dict:
    return {**{k: round(v, 3) for k, v in result.timings.items()},
            "plan_walk_s": round(result.plan.timings["walk"], 3),
            "plan_arrays_s": round(result.plan.timings["arrays_and_checks"], 3)}


def timing_parent(args: argparse.Namespace) -> None:
    rows = []
    for n in args.ns:
        proc = subprocess.run([sys.executable, __file__, "--child", json.dumps([n, args.workers])],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append({"n_features": n, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed"})
        else:
            rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1])[:400], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    print_table(rows)


def print_table(rows: list[dict]) -> None:
    print("\nN | gmsh API write | gmsh API load | direct w1 | direct w8 | direct w16 | direct load")
    for r in rows:
        if "error" in r:
            print(r)
            continue
        d, api, ld = r["direct"], r["gmsh_api"], r["load"]
        ws = [k for k in d if k.startswith("xao_w")]
        print(r["n_features"], f"{api['total_s']:.1f}", f"{ld['gmsh_api_xao'].get('load_s', float('nan')):.1f}",
              *[f"{d[k]['total']:.1f}" for k in ws], f"{ld['direct_xao'].get('load_s', float('nan')):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--ns", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--child", help="internal: JSON [n, workers]")
    args = parser.parse_args()
    if args.child:
        n, workers = json.loads(args.child)
        print(json.dumps(timing_child(n, workers)))
    elif args.timing:
        timing_parent(args)
    else:
        for case in CASES:
            run_case(case)
