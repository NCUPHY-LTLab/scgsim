"""Check that the file excerpts quoted in ``docs/design/sgb/emission.qmd`` and ``brep-format.qmd`` are real.

Run from the repository root with the package's environment::

    uv run --extra palace --group dev python zh-TW/docs/design/sgb/_figures/emission_snippets_check.py

The running example (the pad-in-ground scene of ``tests/sgb_scenes.py`` under Route B, its lumped port on the support plane
z = 0; see ``inset_kit.pad_catalogue``) is written by ``emit_model`` into scratch folders, in a child process of the same
interpreter, and every quoted block is compared with what was written.  Two kinds of block are checked:

* **File blocks** have a header like ``{.text filename="semantic_geometry_route_b.brep, lines 1170-1178"}`` (or ``line 205``).
  The file name selects the written file; the lines are compared with the block, ignoring trailing spaces.
* **Sidecar blocks** have a header like ``{.json filename="semantic_geometry_route_b.sidecar.json（節錄）"}``.  The sidecar is
  read from the example with the contract's ``read_sidecar``.  The excerpt is parsed as JSON (a ``"…"`` key or item stands
  for omitted content) and every value in it must equal the sidecar's, found under the same key: a table of
  ``scgsim.sgb.sidecar.v5`` (``group``, ``face``, ``edge`` ...) or another top-level field.

Exit status 1 lists every mismatch.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from scgsim.contracts.sgb_sidecar import read_sidecar

HERE = Path(__file__).resolve().parent
PAGES = (HERE.parent / "emission.qmd", HERE.parent / "brep-format.qmd")
ROOT = HERE.parents[4]
STEM = "semantic_geometry_route_b"
SIDECAR_BLOCK = re.compile(
    rf'^```\{{\.json filename="(?P<name>{STEM}\.sidecar\.json)（節錄）"\}}\n(?P<body>.*?)^```',
    re.S | re.M,
)
FILE_BLOCK = re.compile(
    r'^```\{\.\w+ filename="(?P<name>[\w.]+), lines? (?P<first>\d+)(?:-(?P<last>\d+))?[^"]*"\}\n(?P<body>.*?)^```',
    re.S | re.M,
)
GENERATE = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import inset_kit as kit
from scgsim.geometry.sgb.build import emit_model
emit_model(kit.pad_model("B", None), Path(sys.argv[2]) / "plain")
"""


def _quoted_blocks(text: str):
    for match in FILE_BLOCK.finditer(text):
        first = int(match["first"])
        last = int(match["last"] or first)
        yield match["name"], first, last, match["body"].rstrip("\n").split("\n")


def _write_example(scratch: Path) -> None:
    """Write the example into ``scratch``, in a fresh interpreter."""
    subprocess.run(
        [sys.executable, "-c", GENERATE, str(HERE), str(scratch)],
        cwd=ROOT,
        check=True,
    )


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
            out += (
                _differences(item, actual[i], f"{path}[{i}]")
                if i < len(actual)
                else [f"{path}[{i}]: missing"]
            )
        return out
    return [] if excerpt == actual else [f"{path}: {excerpt!r} != {actual!r}"]


def _check_sidecar_blocks(scratch: Path, problems: list[str]) -> int:
    blocks = [m for page in PAGES for m in SIDECAR_BLOCK.finditer(page.read_text())]
    sidecar = read_sidecar(scratch / "plain" / f"{STEM}.sidecar.json")
    for match in blocks:
        body = match["body"].strip().rstrip(",")
        excerpt = json.loads("{" + body + "}")
        for key, value in excerpt.items():  # a table or another top-level field
            actual = sidecar.get(key)
            if actual is None:
                problems.append(f"sidecar block: {key!r} is not a sidecar field")
            else:
                problems += [
                    f"sidecar block {key}{d}" for d in _differences(value, actual, "")
                ]
    return len(blocks)


def main() -> int:
    """Compare every quoted file excerpt of the emission pages with the written files; the exit status is 1 on a mismatch."""
    problems: list[str] = []
    cache: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="sgb-snippets-") as folder:
        scratch = Path(folder)
        _write_example(scratch)
        blocks = [block for page in PAGES for block in _quoted_blocks(page.read_text())]
        for name, first, last, body in blocks:
            lines = cache.setdefault(
                name, (scratch / "plain" / name).read_text().split("\n")
            )
            expected = [line.rstrip() for line in lines[first - 1 : last]]
            if [line.rstrip() for line in body] != expected:
                problems.append(
                    f"{name} lines {first}-{last}: the block differs from the file"
                )
        sidecar_blocks = _check_sidecar_blocks(scratch, problems)
    for problem in problems:
        print(problem)
    print(
        f"{len(blocks)} file blocks and {sidecar_blocks} sidecar blocks checked, {len(problems)} mismatching"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
