"""Check that the file excerpts quoted in ``docs/design/sgb/emission.qmd`` and ``brep-format.qmd`` are real lines of the prototype outputs.

Run (from anywhere)::

    /home/ili/Githubs/SCQ_Design/packages/scgsim/.venv/bin/python docs/design/sgb/_figures/emission_snippets_check.py

A quoted block is a fenced block whose header looks like ``{.text filename="model.brep, lines 899-907"}``
(or ``line 205``). The file name selects the prototype output; the lines are compared with the
block, ignoring trailing spaces. Exit status 1 lists every mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = (HERE.parent / "emission.qmd", HERE.parent / "brep-format.qmd")
OUTPUTS = HERE.parents[4] / "experiments" / "sgb-redesign" / "prototype" / "outputs"
SOURCES = {  # name in the block header -> file of the running example
    "model.brep": OUTPUTS / "e3_brep" / "pad_in_ground_B" / "model.brep",
    "model.xao": OUTPUTS / "e3_brep" / "pad_in_ground_B" / "model.xao",
    "model.sidecar.json": OUTPUTS / "e3_brep" / "pad_in_ground_B" / "model.sidecar.json",
}
BLOCK = re.compile(r'^```\{\.\w+ filename="(?P<name>[\w.]+), lines? (?P<first>\d+)(?:-(?P<last>\d+))?[^"]*"\}\n(?P<body>.*?)^```', re.S | re.M)


def _quoted_blocks(text: str):
    for match in BLOCK.finditer(text):
        first = int(match["first"])
        last = int(match["last"] or first)
        yield match["name"], first, last, match["body"].rstrip("\n").split("\n")


def main() -> int:
    cache: dict[str, list[str]] = {}
    problems = []
    blocks = [block for page in PAGES for block in _quoted_blocks(page.read_text())]
    for name, first, last, body in blocks:
        lines = cache.setdefault(name, SOURCES[name].read_text().split("\n"))
        expected = [line.rstrip() for line in lines[first - 1:last]]
        if [line.rstrip() for line in body] != expected:
            problems.append(f"{name} lines {first}-{last}: the block differs from the file")
    for problem in problems:
        print(problem)
    print(f"{len(blocks)} quoted blocks checked, {len(problems)} mismatching")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
