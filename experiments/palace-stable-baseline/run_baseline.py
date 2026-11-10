# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""Run every case of the baseline, each in its own process, and write one record per case.

    python run_baseline.py RECORDS_DIR [--jobs 8] [--timeout 900] [--only CASE ...]

First every layout with bands and every supported scene; then, for every layout that stopped with bands,
the same layout without bands; then, for every layout whose banded mesh() failed, the same layout with bands
and the vacuum the PDK's Palace notebooks give the plan (``--notebook-vacuum``) instead of the 1 um shell.  A case still running after ``--timeout`` seconds is stopped with its whole
process group and recorded as ``timeout``.  The interpreter is this one: run it from the environment that
has SCGSim and the OrcaPen SC PDK revision under test.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_case import write_record  # noqa: E402


def main() -> None:
    """Run the three passes of the baseline."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.records.mkdir(parents=True, exist_ok=True)

    from cases import LAYOUTS, scene_builders

    sys.path.insert(0, str(HERE.parents[1] / "tests"))
    layouts = [name for name in LAYOUTS if not args.only or name in args.only]
    scenes = [name for name in scene_builders() if not args.only or name in args.only]
    first = [(name, True, False, False) for name in layouts] + [
        (name, True, True, False) for name in scenes
    ]
    _run_all(first, args)
    banded = {
        name: json.loads((args.records / record_file(name, True, False)).read_text())
        for name in layouts
    }
    _run_all(
        [
            (name, False, False, False)
            for name, record in banded.items()
            if _stopped(record)
        ],
        args,
    )
    mesh_failed = [
        name
        for name, record in banded.items()
        if record["stages"].get("mesh", {}).get("result") not in (None, "pass")
    ]
    _run_all([(name, True, False, True) for name in mesh_failed], args)


def record_file(
    name: str, bands: bool, scene: bool, notebook_vacuum: bool = False
) -> str:
    """The file name of one case's record."""
    variant = ("bands" if bands else "nobands") + (
        "_notebookvacuum" if notebook_vacuum else ""
    )
    return f"{'scene' if scene else 'layout'}__{name}__{variant}.json"


def _stopped(record) -> bool:
    return (
        bool(record.get("exit_code"))
        or any(stage["result"] != "pass" for stage in record["stages"].values())
        or ("mesh" not in record["stages"])
    )


def _run_all(jobs, args) -> None:
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(lambda job: run_one(*job, args), jobs):
            print(line, flush=True)


def run_one(name: str, bands: bool, scene: bool, notebook_vacuum: bool, args) -> str:
    """Run one case in its own process group, write its record to ``args.records`` and say how it ended."""
    out = args.records / record_file(name, bands, scene, notebook_vacuum)
    command = [sys.executable, str(HERE / "run_case.py"), name, str(out)]
    command += ["--scene"] if scene else ([] if bands else ["--no-bands"])
    command += ["--notebook-vacuum"] if notebook_vacuum else []
    # A rerun into the same folder must not show an earlier run's stages: a case that stops before its first
    # record (an import error, say) would otherwise leave the old file in place.
    out.unlink(missing_ok=True)
    log = out.with_suffix(".log").open("w")
    started = time.perf_counter()
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        code = "timeout"
    seconds = round(time.perf_counter() - started, 1)
    if code != 0:
        # The case writes its record around every stage, so a stopped case keeps the stages it finished and
        # names the one that was running.  The file was removed above, so a record here is this run's.
        record = stopped_record(
            out,
            {
                "case": name,
                "bands": bands,
                "scene": scene,
                "notebook_vacuum": notebook_vacuum,
                "stages": {},
            },
        )
        for stage in record["stages"].values():
            if stage["result"] == "running":
                stage.update(
                    result="timeout" if code == "timeout" else "crashed", seconds=None
                )
        record.update(timeout=code == "timeout", exit_code=code, seconds=seconds)
        write_record(out, record)
    return f"{out.name}: exit {code}, {seconds} s"


def stopped_record(out: Path, empty: dict) -> dict:
    """The record a stopped case left at ``out``, or ``empty`` when it left none or none that can be read.

    ``run_case`` writes whole records only, so an unreadable file should not occur; if one does, the run's
    stages are unknown and the record says none rather than stopping the whole baseline.
    """
    try:
        return json.loads(out.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return empty


if __name__ == "__main__":
    main()
