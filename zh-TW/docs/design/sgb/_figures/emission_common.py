"""Shared inputs of the emission figure scripts: the output paths, palettes, the running example and the small table and file helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from sgb_scenes import airbridge  # noqa: E402

from scgsim.contracts.sgb_sidecar import read_sidecar, views  # noqa: E402
from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.output.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402

FIGURES = kit.FIGURES
DATA = FIGURES / "emission-timing-data.json"
INK = "#1b1b1b"
SHARED = (
    FIGURES / "why-scaling-measured.json"
)  # production stage times, written by why_scaling.py; reused here, not re-measured
HEAD_SIZES = (
    500,
    1000,
    2000,
)  # the Gmsh-API oracle is superlinear in the number of physical groups; it did not finish N = 5000 in 10 minutes
VALIDATOR_SIZES = (1000, 5000, 20000)
RANDOM_SEED = 2  # random_layout(n, seed=2, route="B"): seed 1 at N = 20000 legitimately fails (two shapes of one Net pinch, NonManifoldError)
STAGE_COLOURS = {  # one colour per named stage, shared by the plotly and matplotlib versions
    "OCC build (addPoint ... addVolume)": "#D55E00",
    "synchronize": "#E69F00",
    "physical groups": "#F0E442",
    "write XAO": "#CC79A7",
    "plan (serial)": "#0072B2",
    "render": "#56B4E9",
    "load into Gmsh": "#444444",
}
GROUP_COLOURS = {
    "MA::pad__top": "#0072B2",
    "MA::pad__sidewall": "#7fb8e0",
    "MS::pad__bottom": "#56B4E9",
    "MA::ground__top": "#8C8C8C",
    "MA::ground__sidewall": "#b0b0b0",
    "MS::ground__bottom": "#CFCFCF",
    "SA::substrate|vacuum__interface": "#E69F00",
    "PORT::p1": "#D55E00",
    "EXT::substrate__bottom": "#009E73",
    "EXT::substrate__sidewall": "#4fc3a1",
    "EXT::vacuum__top": "#CC79A7",
    "EXT::vacuum__sidewall": "#e3a8c9",
}
GROUP_OPACITY = {name: 0.12 for name in GROUP_COLOURS if name.startswith("EXT")}


# --------------------------------------------------------------------------- helpers


def seconds_text(x: float) -> str:
    """Seconds with one decimal from ten, two from one and three below."""
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}" if x >= 1 else f"{x:.3f}"


def markdown_table(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> str:
    """A Markdown table; cells escape ``|``, the first column is left-aligned and the others right-aligned unless ``align`` says otherwise."""
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule) + "|"]
    lines += [
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def write_figure_file(name: str, text: str) -> None:
    """Write a text file into the figures folder and report its name."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text(text)
    print("wrote", name)


def case_models() -> dict:
    """The running examples of the production tests (``CASES``), with the pad's lumped port on the support plane."""
    return {
        "pad_in_ground_A": lambda: kit.pad_model("A", None),
        "pad_in_ground_B": lambda: kit.pad_model("B", None),
        "pad_in_ground_C": lambda: build_model_from_catalogue(
            kit.pad_catalogue("B"), "C"
        ),
        "airbridge_B": lambda: build_model_from_catalogue(airbridge(), "B"),
        "pad_in_ground_A_bands": lambda: kit.pad_model("A", kit.MARGINS),
        "pad_in_ground_B_bands": lambda: kit.pad_model("B", kit.MARGINS),
        "pad_in_ground_C_bands": lambda: build_model_from_catalogue(
            kit.pad_catalogue("B"), "C", insets=kit.pad_requests("B")
        ),
    }


def bundle(model, folder: Path):
    """Write a model and return (artifacts, plan, sidecar)."""
    artifacts = emit_model(model, folder)
    return (
        artifacts,
        plan_brep(model.topology, model.routed),
        read_sidecar(artifacts.sidecar),
    )


# --------------------------------------------------------------------------- the running example


def running_example(folder: Path):
    """The pad-in-ground example (Route B), its files and plan, and the check that the plan's numbering equals the files'."""
    model = kit.pad_model("B", None)
    artifacts, plan, sidecar = bundle(model, folder)
    recorded = {
        pid: (int(dim), i + 1)
        for dim, ids in sidecar["binding"]["ids_by_tag"].items()
        for i, pid in enumerate(ids)
    }
    if recorded != plan.tags():
        raise SystemExit("the plan's numbering is not the written sidecar's ids_by_tag")
    return model, plan, sidecar, artifacts


def face_reference(plan, face_index: int) -> int:
    """The number a shell uses for this face: its position counted from the end of the TShapes section."""
    return plan.section_bases()["F"] - face_index


def check_shell_references(plan, brep: Path) -> None:
    """The shell records of the written BREP are the plan's shells with the references this script derives."""
    lines = brep.read_text().split("\n")
    written = []
    for i, line in enumerate(lines):
        if line == "Sh" and lines[i + 1] == "":
            tokens = lines[i + 3].split()
            written.append([int(tokens[k]) for k in range(0, len(tokens) - 1, 2)])
    expected = [
        [(1 if r > 0 else -1) * face_reference(plan, abs(r) - 1) for r in shell]
        for shell in plan.found.shells
    ]
    if written != expected:
        raise SystemExit(
            "the shell records of the BREP file are not what the plan's references say"
        )


def example_numbers(model, plan, sidecar) -> dict:
    """The counts of the running example the emission page quotes: planned and emitted entities, face use by shells, groups and tags."""
    uses: dict[int, int] = {}
    for shell in plan.found.shells:
        for ref in shell:
            uses[abs(ref)] = uses.get(abs(ref), 0) + 1
    members = views.group_members(sidecar)
    numbers = {
        "planned": model.topology.counts(),
        "emitted": plan.counts,
        "section_bases": plan.section_bases(),
        "faces_used_by_two_shells": sum(1 for n in uses.values() if n == 2),
        "faces_used_by_one_shell": sum(1 for n in uses.values() if n == 1),
        "groups": {
            name: {"dim": g["dim"], "tag": g["tag"], "members": len(members[name])}
            for name, g in sidecar["group"].items()
        },
        "ids_by_tag": {
            dim: len(ids) for dim, ids in sidecar["binding"]["ids_by_tag"].items()
        },
        "sidecar_bytes": None,
        "embedded_faces": len(views.embedded_faces(sidecar)),
        "free_faces": plan.counts["free_faces"],
    }
    return numbers
