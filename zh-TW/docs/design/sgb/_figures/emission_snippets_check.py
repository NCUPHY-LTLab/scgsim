"""Check that the file excerpts quoted in ``docs/design/sgb/emission.qmd`` and ``brep-format.qmd`` are real.

Run (from anywhere)::

    python3 docs/design/sgb/_figures/emission_snippets_check.py

Two kinds of quoted block are checked:

* **Prototype blocks** have a header like ``{.text filename="model.brep, lines 899-907"}`` (or ``line 205``).
  The file name selects the prototype output; the lines are compared with the block, ignoring trailing spaces.
* **Production blocks** have a header like ``{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}``.
  The sidecar is generated into a scratch directory by the checkout that contains ``scgsim.geometry.sgb`` (``pad_b`` of
  ``tests/sgb_emit_cases.py``, built with ``emit_model``); the excerpt is parsed as JSON (a ``"…"`` key or
  item stands for omitted content) and every value in it must equal the sidecar's, found under the same key at
  the top level or among the ``physical_groups``. Set ``SGB_CORE`` to that checkout when it is not this one.

Exit status 1 lists every mismatch.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = (HERE.parent / "emission.qmd", HERE.parent / "brep-format.qmd")
OUTPUTS = HERE.parents[4] / "experiments" / "sgb-redesign" / "prototype" / "outputs"
SOURCES = {  # name in the block header -> file of the running example
    "model.brep": OUTPUTS / "e3_brep" / "pad_in_ground_B" / "model.brep",
    "model.xao": OUTPUTS / "e3_brep" / "pad_in_ground_B" / "model.xao",
    "model.sidecar.json": OUTPUTS / "e3_brep" / "pad_in_ground_B" / "model.sidecar.json",
}
SGB_CORE = Path(os.environ.get("SGB_CORE", HERE.parents[4]))  # checkout with scgsim.geometry.sgb
PRODUCTION_BLOCK = re.compile(r'^```\{\.json filename="(?P<name>semantic_geometry_route_b\.sidecar\.json)（節錄）"\}\n(?P<body>.*?)^```', re.S | re.M)
GENERATE = """
import sys
sys.path.insert(0, "tests")
from pathlib import Path
from sgb_emit_cases import CASES
from scgsim.geometry.sgb.build import emit_model
emit_model(CASES["pad_b"](), Path(sys.argv[1]))
"""
BLOCK = re.compile(r'^```\{\.\w+ filename="(?P<name>[\w.]+), lines? (?P<first>\d+)(?:-(?P<last>\d+))?[^"]*"\}\n(?P<body>.*?)^```', re.S | re.M)


def _quoted_blocks(text: str):
    for match in BLOCK.finditer(text):
        first = int(match["first"])
        last = int(match["last"] or first)
        yield match["name"], first, last, match["body"].rstrip("\n").split("\n")


def _production_sidecar() -> dict:
    """Build ``pad_b`` with the sgb-core worktree into a scratch directory and read its sidecar."""
    python = SGB_CORE / ".artifacts" / "env" / "bin" / "python"
    with tempfile.TemporaryDirectory(prefix="sgb-snippets-") as scratch:
        subprocess.run([str(python), "-c", GENERATE, scratch], cwd=SGB_CORE, check=True)
        return json.loads(next(Path(scratch).glob("*.sidecar.json")).read_text())


def _differences(excerpt: object, actual: object, path: str) -> list[str]:
    """Where ``excerpt`` (with ``"…"`` for omitted content) is not contained in ``actual``."""
    if isinstance(excerpt, dict) and isinstance(actual, dict):
        out = []
        for key, value in excerpt.items():
            if key == "…":
                continue
            if key not in actual:
                out.append(f"{path}/{key}: no such key")
            else:
                out += _differences(value, actual[key], f"{path}/{key}")
        return out
    if isinstance(excerpt, list) and isinstance(actual, list):
        out = []
        for i, item in enumerate(x for x in excerpt if x != "…"):
            out += _differences(item, actual[i], f"{path}[{i}]") if i < len(actual) else [f"{path}[{i}]: missing"]
        return out
    return [] if excerpt == actual else [f"{path}: {excerpt!r} != {actual!r}"]


def _check_production(problems: list[str]) -> int:
    blocks = [m for page in PAGES for m in PRODUCTION_BLOCK.finditer(page.read_text())]
    if not blocks:
        return 0
    sidecar = _production_sidecar()
    for match in blocks:
        body = match["body"].strip().rstrip(",")
        excerpt = json.loads("{" + body + "}")
        for key, value in excerpt.items():  # a top-level key, or a group name
            actual = sidecar.get(key, sidecar["physical_groups"].get(key))
            if actual is None:
                problems.append(f"production block: {key!r} is neither a sidecar key nor a physical group")
            else:
                problems += [f"production block {key}{d}" for d in _differences(value, actual, "")]
    return len(blocks)


def main() -> int:
    cache: dict[str, list[str]] = {}
    problems = []
    blocks = [block for page in PAGES for block in _quoted_blocks(page.read_text())]
    for name, first, last, body in blocks:
        lines = cache.setdefault(name, SOURCES[name].read_text().split("\n"))
        expected = [line.rstrip() for line in lines[first - 1:last]]
        if [line.rstrip() for line in body] != expected:
            problems.append(f"{name} lines {first}-{last}: the block differs from the file")
    production = _check_production(problems)
    for problem in problems:
        print(problem)
    print(f"{len(blocks)} prototype and {production} production quoted blocks checked, {len(problems)} mismatching")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
