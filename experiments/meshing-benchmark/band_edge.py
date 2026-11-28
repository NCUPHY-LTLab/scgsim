"""Band-edge sizing candidates of issue 136: which a priori size near Surface-EPR band edges meshes clean, and how fast.

Usage (from the repository root; the dev group synced; an OrcaPen SC PDK checkout at the pin of PR 132)::

    uv run --locked --no-sync --with-editable PDK_CHECKOUT python experiments/meshing-benchmark/band_edge.py \\
        run OUTPUT_DIR [--cases NAME ...] [--candidates NAME ...] [--label first]
    uv run --locked --no-sync python experiments/meshing-benchmark/band_edge.py \\
        table OUTPUT_DIR [OUTPUT_DIR ...] [--write-json experiments/meshing-benchmark/band_edge.json]

``run`` builds every selected case once and meshes it with every selected candidate (:data:`CANDIDATES`),
each pair in a fresh process of its own, one after another, and writes one record per run to
``OUTPUT_DIR/runs/LABEL/CASE/CANDIDATE.json``: the numbers of the quality report, the sizes the recipe
asked for on that case, the hashes of the geometry, the time, ``/proc/loadavg`` and the Palace processes
on the machine before and after, and the SCGSim and PDK revisions.  ``table`` prints the tables of
``band_edge.md`` from those records, one per pass, and writes them, without local paths, to one JSON file.

The table in ``band_edge.md`` was measured before ``FeatureRecipe`` had its band-edge rule: its ``default``
rows are ``scgsim.meshing.features/1`` (no size on band edges), and its band rows came from an experiment
recipe (``experiments.meshing-benchmark.band-edge/1``) that asked for the sizes ``FeatureRecipe`` now asks
for with ``band_mesh_size_multiple``.  ``band4`` became the default (``features/2``), so here ``default``
and ``band4`` are the same recipe, and ``features/1`` is no longer a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

from scgsim.meshing import FeatureRecipe, SizingRecipe, UniformBoundaryRecipe
from scgsim.meshing.features import load_features

HERE = Path(__file__).resolve().parent
# Bad elements named per stopped run in the table; the records keep every element the report lists.
LISTED_PLACES = 3

# The two references and four band-edge candidates; band_edge.md states each rule in one sentence.
CANDIDATES: dict[str, SizingRecipe] = {
    "default": FeatureRecipe(),
    "uniform": UniformBoundaryRecipe(),
    "band2": FeatureRecipe(band_mesh_size_multiple=2.0),
    "band4": FeatureRecipe(band_mesh_size_multiple=4.0),
    "band8": FeatureRecipe(band_mesh_size_multiple=8.0),
    "wall100_band8": FeatureRecipe(
        film_sidewall_thickness_multiple=100.0, band_mesh_size_multiple=8.0
    ),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``run``, ``one`` (a single run, in the process ``run`` starts) and ``table``."""
    args = _parse_arguments(argv)
    if args.command == "run":
        return _run_matrix(args)
    if args.command == "one":
        _run_one(args.output_dir, args.case, args.candidate, args.label)
        return 0
    return _print_table(args.output_dirs, args.write_json)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="mesh every case with every candidate")
    run.add_argument("output_dir", type=Path)
    run.add_argument("--cases", nargs="+")
    run.add_argument("--candidates", nargs="+", choices=list(CANDIDATES))
    run.add_argument("--label", default="first", help="name of this pass of runs")
    one = commands.add_parser("one", help="one case and candidate in this process")
    for name in ("output_dir", "case", "candidate", "label"):
        one.add_argument(name, type=Path if name == "output_dir" else str)
    table = commands.add_parser("table", help="print the table of the records")
    table.add_argument("output_dirs", type=Path, nargs="+")
    table.add_argument("--write-json", type=Path)
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------ runs


def _run_matrix(args: argparse.Namespace) -> int:
    """Build the chip geometry once, then run each case and candidate in a fresh process."""
    import band_edge_cases as cases  # noqa: PLC0415 -- needs the PDK, which `table` does not

    selected = args.cases or list(cases.CASES)
    unknown = sorted(set(selected) - set(cases.CASES))
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; known: {', '.join(cases.CASES)}")
    for name in selected:
        directory = args.output_dir / "cases" / name
        if not (directory / "case.json").is_file():
            directory.mkdir(parents=True, exist_ok=True)
            cases.build_case(name, directory)
    failures = 0
    for name in selected:
        for candidate in args.candidates or list(CANDIDATES):
            print(f"== {args.label} {name} {candidate}", flush=True)
            command = [sys.executable, __file__, "one", str(args.output_dir)]
            finished = subprocess.run(
                [*command, name, candidate, args.label], check=False
            )
            failures += finished.returncode != 0
    return 1 if failures else 0


def _run_one(output_dir: Path, case: str, candidate: str, label: str) -> None:
    """Mesh ``case`` with ``candidate`` and write the run record."""
    import band_edge_cases as cases  # noqa: PLC0415 -- needs the PDK, which `table` does not

    recipe = CANDIDATES[candidate]
    directory = output_dir / "meshes" / label / case / candidate
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f"{directory} is not empty; use another --label")
    record: dict[str, Any] = {"case": case, "candidate": candidate, "label": label}
    record["started_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["load_before"] = _loadavg()
    record["palace_before"] = _palace_processes()
    xao, sidecar = cases.case_geometry(output_dir / "cases" / case)
    report = cases.mesh_built(xao, sidecar, recipe, directory)
    record["load_after"] = _loadavg()
    record["palace_after"] = _palace_processes()
    record["geometry"] = {
        "xao_sha256": cases.file_sha256(xao),
        "sidecar_sha256": cases.file_sha256(sidecar),
    }
    record["sizes"] = _requested_sizes(recipe, xao, sidecar)
    record["environment"] = _environment()
    record |= _report_numbers(report)
    target = output_dir / "runs" / label / case / f"{candidate}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(_row(record), flush=True)


def _loadavg() -> list[float]:
    """The 1-, 5- and 15-minute load averages of the machine."""
    fields = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    return [float(value) for value in fields[:3]]


def _palace_processes() -> int:
    """Processes whose executable name starts with ``palace`` (a Palace run of another agent)."""
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "comm").read_text(encoding="utf-8").strip()
        except OSError:  # the process ended while the table was read
            continue
        count += command.startswith("palace")
    return count


def _requested_sizes(recipe: SizingRecipe, xao: Path, sidecar: Path) -> dict[str, Any]:
    """The coarse size and the curve sizes ``recipe`` asks for on this model, as ``mesh_sgb`` asked for them."""
    # Gmsh reads the extent here, after the run, so the timed call loads it only in its own workers.
    from scgsim.meshing.native import bounding_box_of  # noqa: PLC0415

    requests = recipe.size_requests(load_features(sidecar), bounding_box_of(xao))
    curves = Counter(requests.curve_sizes.values())
    return {
        "max_size_um": requests.max_size,
        "curves_by_size_um": {f"{size:.4g}": n for size, n in sorted(curves.items())},
        "fields": [
            {"size_um": f.size, "distance_um": f.distance, "curves": len(f.curves)}
            for f in requests.fields
        ],
    }


def _environment() -> dict[str, str]:
    """Package versions, the SCGSim and PDK revisions, and the processor count; no local paths."""
    packages = {}
    for name in ("gmsh", "scgsim", "orcapen-sc-pdk", "gdsfactory"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    packages["scgsim_revision"] = _git_revision(HERE)
    packages["orcapen_sc_pdk_revision"] = _pdk_revision()
    packages["python"] = sys.version.split()[0]
    packages["cpus"] = str(os.cpu_count())
    return packages


def _pdk_revision() -> str:
    """The PDK commit: from a git install's ``direct_url.json``, or from the git checkout an editable install points at."""
    try:
        direct = distribution("orcapen-sc-pdk").read_text("direct_url.json")
    except PackageNotFoundError:
        return "not installed"
    info = json.loads(direct) if direct else {}
    if "vcs_info" in info:
        return str(info["vcs_info"]["commit_id"])
    url = str(info.get("url", ""))
    if url.startswith("file://"):
        return _git_revision(Path(url.removeprefix("file://")))
    return "unknown"


def _git_revision(directory: Path) -> str:
    """``HEAD`` of the checkout at ``directory``, with the tracked files changed in its working tree."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True
    )
    if head.returncode:
        return "unknown"
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=directory,
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    names = sorted(line[3:] for line in changed if line.strip())
    return head.stdout.strip() + (f" + changes in {', '.join(names)}" if names else "")


def _report_numbers(report: dict[str, Any]) -> dict[str, Any]:
    """What the table reads from a quality report: outcome, rounds, timing and where the bad elements are."""
    mesh = report["mesh"] or {}
    remaining = report["remaining_bad"]
    return {
        "recipe": report["recipe"],
        "stop": report["stop"],
        "rounds": report["rounds"],
        "tetrahedra": mesh.get("tetrahedra"),
        "min_quality": mesh.get("min_quality"),
        "bad_count": mesh.get("bad_count"),
        "sized_curves": mesh.get("sized_curves"),
        "local_sizes": mesh.get("local_sizes"),
        "smallest_local_size_um": mesh.get("smallest_local_size_um"),
        "history": report["history"],
        "time_budget": report["time_budget"],
        "seconds": report["timing"]["total_seconds"],
        "round_seconds": report["timing"]["round_seconds"],
        "remaining_bad": {
            "count": remaining["count"],
            "by_location": remaining["by_location"],
            "listed": [_place(element) for element in remaining["listed"]],
        },
    }


def _place(element: dict[str, Any]) -> dict[str, Any]:
    keep = ("quality", "centroid", "edge_min", "edge_max", "location")
    return {
        **{key: element[key] for key in keep},
        "groups": element["groups"],
        "feature_kinds": element["feature_kinds"],
    }


# ----------------------------------------------------------------------------------------------- table


def _print_table(output_dirs: list[Path], write_json: Path | None) -> int:
    """One table per pass (label), rows by case and candidate; all records to ``write_json``."""
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for output_dir in output_dirs
        for path in sorted((output_dir / "runs").glob("*/*/*.json"))
    ]
    records.sort(key=_order)
    for label in dict.fromkeys(record["label"] for record in records):
        rows = [_row(record) for record in records if record["label"] == label]
        print(f"### {label}\n\n{TABLE_HEADER}" + "\n".join(rows) + "\n")
    if write_json is not None:
        write_json.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    return 0


# Passes, cases and candidates in the order the table lists them.
PASS_ORDER = ("first", "quiet", "pin49", "pin49-quiet", "features2")
CASE_ORDER = (
    "disk_nobands",
    "disk_bands",
    "B_1mm_nobands",
    "B_1mm_bands",
    "circB_2mm_nobands",
    "circB_2mm_bands",
    "circB_4mm_nobands",
    "circB_4mm_bands",
    "simplepad_A_bands",
    "simplepad_A_bands_nb",
    "simplepad_B_bands",
    "simplepad_B_bands_nb",
    "disk_site_nobands",
    "disk_site_bands",
)


def _order(record: dict[str, Any]) -> tuple[int, int, int, str]:
    def rank(value: str, order: tuple[str, ...]) -> int:
        return order.index(value) if value in order else len(order)

    return (
        rank(record["label"], PASS_ORDER),
        rank(record["case"], CASE_ORDER),
        rank(record["candidate"], tuple(CANDIDATES)),
        record["started_utc"],
    )


TABLE_HEADER = (
    "| case | candidate | sizes µm (coarse; curves) | s | tets (round 0 → last) | min SICN "
    "| bad per round | rounds | stop | where (stopped runs) | load 1 min (before–after) | Palace | pass |\n"
    "|---|---|---|---:|---|---:|---|---:|---|---|---|---|---|\n"
)


def _row(record: dict[str, Any]) -> str:
    history = [
        round_ for round_ in record["history"] if round_["tetrahedra"] is not None
    ]
    tetrahedra = (
        f"{history[0]['tetrahedra']:,} → {history[-1]['tetrahedra']:,}"
        if len(history) > 1
        else (f"{history[0]['tetrahedra']:,}" if history else "-")
    )
    bad = ", ".join(
        "killed" if round_["timed_out"] else str(round_["bad_count"])
        for round_ in record["history"]
    )
    quality = "-" if record["min_quality"] is None else f"{record['min_quality']:.4g}"
    before, after = record["load_before"][0], record["load_after"][0]
    palace = record["palace_before"] + record["palace_after"]
    return (
        f"| {record['case']} | {record['candidate']} | {_sizes(record['sizes'])} "
        f"| {record['seconds']:.1f} | {tetrahedra} | {quality} | {bad} | {record['rounds']} "
        f"| {record['stop']} | {_where(record)} | {before:.1f}–{after:.1f} "
        f"| {'yes' if palace else 'no'} | {record['label']} |"
    )


def _sizes(sizes: dict[str, Any]) -> str:
    curves = ", ".join(f"{size} ×{n}" for size, n in sizes["curves_by_size_um"].items())
    fields = ", ".join(
        f"{f['size_um']:.4g} ×{f['curves']} over {f['distance_um']:.3g}"
        for f in sizes["fields"]
    )
    parts = [
        f"{sizes['max_size_um']:.4g}",
        curves or "none",
        *([fields] if fields else []),
    ]
    return "; ".join(parts)


def _where(record: dict[str, Any]) -> str:
    """For a stop that is not clean: bad elements by location, then the groups of the worst few."""
    if record["stop"] == "clean":
        return ""
    remaining = record["remaining_bad"]
    by_location = ", ".join(
        f"{n} {where}" for where, n in remaining["by_location"].items()
    )
    places = []
    for element in remaining["listed"][:LISTED_PLACES]:
        x, y, z = (round(value, 1) for value in element["centroid"])
        kinds = "/".join(element["feature_kinds"]) or "-"
        groups = "; ".join(_short_group(group) for group in element["groups"][:2])
        places.append(f"{element['quality']:.2g} at ({x}, {y}, {z}) {kinds} [{groups}]")
    return f"{remaining['count']} bad ({by_location}): " + " · ".join(places)


def _short_group(group: str) -> str:
    """A group name without its entity path (``MS::TC_E__top::band=0``), with ``|`` escaped for the table."""
    kind, _, rest = group.partition("::")
    if rest:
        entity, _, tail = rest.partition("__")
        leaf = entity.rsplit("/", 1)[-1]
        group = f"{kind}::{leaf}__{tail}" if tail else f"{kind}::{leaf}"
    return group.replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
