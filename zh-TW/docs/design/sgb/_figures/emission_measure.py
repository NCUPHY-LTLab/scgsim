"""Measuring children of ``emission.py``: one layout through the Gmsh-API oracle and the direct writer, and one validation run."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmsh
from emission_common import DATA, HEAD_SIZES, RANDOM_SEED, SHARED, VALIDATOR_SIZES
from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.verification.loaded import verify_loaded  # noqa: E402
from sgb_gmsh_oracle import OracleResult, gmsh_session  # noqa: E402
from sgb_gmsh_oracle import _build_entities as oracle_build_entities  # noqa: E402  # private-import: the timing figure needs the oracle's entity build apart from its synchronize and physical-group steps, which emit_with_gmsh_api does not expose
from sgb_scenes import random_layout  # noqa: E402

# the children run through ``emission.py`` itself, so each starts with every import of the full script, as a figure run does
ENTRY_SCRIPT = Path(__file__).resolve().with_name("emission.py")
LOAD_SCRIPT = (
    "import sys, time, gmsh\n"
    "gmsh.initialize(['-noenv']); gmsh.option.setNumber('General.Terminal', 0); gmsh.option.setNumber('Geometry.OCCAutoFix', 0)\n"
    "t = time.perf_counter()\n"
    "(gmsh.open if sys.argv[1].endswith('.xao') else gmsh.merge)(sys.argv[1])\n"
    "gmsh.model.occ.synchronize()\n"
    "print(time.perf_counter() - t)\n"
)


# --------------------------------------------------------------------------- timings


def _load_seconds(path: Path) -> float:
    """Seconds Gmsh needs to open (XAO) or merge (BREP) a file and synchronize, in a fresh process."""
    out = subprocess.run(
        [sys.executable, "-c", LOAD_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def head_one(n: int) -> dict:
    """One random Route B layout through the Gmsh-API oracle, stage by stage, and through the direct writer (16 workers), then both loads."""

    model = build_model_from_catalogue(
        random_layout(n, seed=RANDOM_SEED, route="B"), "B"
    )
    row: dict = {"n": n, "counts": model.counts, "groups": len(model.routed.groups)}
    with tempfile.TemporaryDirectory(prefix=f"emission-head-{n}-") as scratch:
        folder = Path(scratch)
        started = time.perf_counter()
        plan = plan_brep(model.topology, model.routed)
        plan_s = time.perf_counter() - started
        brep, xao = folder / "direct.brep", folder / "direct.xao"
        render_brep = write_brep_file(plan, brep, workers=16)
        render_xao = write_xao_file(plan, xao, name="direct", workers=16, brep=brep)
        row["direct"] = {
            "plan_s": plan_s,
            "render_brep_s": render_brep,
            "render_xao_s": render_xao,
            "plan_counts": plan.counts,
            "xao_bytes": xao.stat().st_size,
            "brep_bytes": brep.stat().st_size,
        }
        with gmsh_session():
            result = OracleResult()
            t0 = time.perf_counter()
            oracle_build_entities(model, result)
            t1 = time.perf_counter()
            gmsh.model.occ.synchronize()
            t2 = time.perf_counter()
            for name, group in model.routed.groups.items():
                tags = [result.tags[i][1] for i in group.ids if i in result.tags]
                if tags:
                    gmsh.model.addPhysicalGroup(group.dim, tags, name=name)
            t3 = time.perf_counter()
            api_xao = folder / "api.xao"
            gmsh.write(str(api_xao))
            t4 = time.perf_counter()
        row["api"] = {
            "occ_build_s": t1 - t0,
            "synchronize_s": t2 - t1,
            "physical_groups_s": t3 - t2,
            "write_xao_s": t4 - t3,
            "total_s": t4 - t0,
            "xao_bytes": api_xao.stat().st_size,
        }
        row["load"] = {
            "api_xao_s": _load_seconds(api_xao),
            "direct_xao_s": _load_seconds(xao),
            "direct_brep_s": _load_seconds(brep),
        }
    return row


def validator_one(n: int) -> dict:
    """Build one layout, write all its files, open the XAO and run ``verify_loaded``: the time of every check, structural and load-side."""

    started = time.perf_counter()
    model = build_model_from_catalogue(
        random_layout(n, seed=RANDOM_SEED, route="B"), "B"
    )
    build_s = time.perf_counter() - started
    with tempfile.TemporaryDirectory(prefix=f"emission-validator-{n}-") as scratch:
        artifacts = emit_model(model, Path(scratch), workers=16)
        sidecar = read_sidecar(artifacts.sidecar)
        with gmsh_session():
            started = time.perf_counter()
            gmsh.open(str(artifacts.xao))
            gmsh.model.occ.synchronize()
            load_s = time.perf_counter() - started
            started = time.perf_counter()
            report = verify_loaded(sidecar)
            verify_s = time.perf_counter() - started
        assert report.ok, report.problems[:3]
    return {
        "n": n,
        "counts": model.counts,
        "written": {
            k: artifacts.counts[k] for k in ("vertices", "edges", "faces", "solids")
        },
        "build_s": build_s,
        "stage_s": model.timings,
        "validator_s": model.report.timings,
        "load_s": load_s,
        "verify_s": verify_s,
        "verify_phases_s": report.timings_s,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def measure() -> None:
    """Run the two measurements of this page, one subprocess per N (a clean peak memory), and store them."""
    out: dict = {
        "layout": f"random_layout(n, seed={RANDOM_SEED}, route='B')",
        "cpu_count": os.cpu_count(),
        "load_average_at_start": os.getloadavg(),
        "head": [],
        "validator": [],
    }
    for key, flag, sizes in (
        ("head", "--head-one", HEAD_SIZES),
        ("validator", "--validator-one", VALIDATOR_SIZES),
    ):
        for n in sizes:
            started = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(ENTRY_SCRIPT), flag, str(n)],
                capture_output=True,
                text=True,
                check=True,
            )
            out[key].append(json.loads(proc.stdout.strip().splitlines()[-1]))
            print(f"{flag} N = {n}: {time.perf_counter() - started:.0f} s", flush=True)
            DATA.write_text(json.dumps(out, indent=1))
    out["load_average_at_end"] = os.getloadavg()
    DATA.write_text(json.dumps(out, indent=1))


def shared_rows() -> dict[int, dict]:
    """The one-tile, one-worker runs of ``figures/why-scaling-measured.json`` (production build, files, Gmsh load and ``verify_loaded``), by N."""
    rows = json.loads(SHARED.read_text())["rows"]
    return {
        r["n_features"]: r
        for r in rows
        if r["tiles_per_side"] == 1 and r["workers"] == 1
    }
