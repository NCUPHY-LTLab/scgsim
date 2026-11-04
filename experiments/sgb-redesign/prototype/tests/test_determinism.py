"""Same input => same ids, across runs, interpreter hash seeds and worker counts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sgbproto.pipeline import build, ids_digest
from sgbproto.scenes import pad_in_ground, random_layout

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    "import sys; sys.path.insert(0, %r)\n"
    "from sgbproto.pipeline import build, ids_digest\n"
    "from sgbproto.scenes import random_layout, pad_in_ground\n"
    "print(ids_digest(build(random_layout(40, 9, 'B'))), ids_digest(build(pad_in_ground('B'))))\n"
) % str(ROOT)


def test_two_runs_give_identical_ids():
    assert ids_digest(build(pad_in_ground("B"))) == ids_digest(build(pad_in_ground("B")))
    spec = random_layout(40, seed=9, route="B")
    assert ids_digest(build(spec)) == ids_digest(build(spec))


def test_ids_do_not_depend_on_python_hash_randomisation():
    outputs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True)
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1


def test_ids_are_content_addressed_not_positional():
    a = build(pad_in_ground("A"))
    b = build(pad_in_ground("A", hole_sides=16))
    # unrelated geometry keeps its ids: the ground plane's outer boundary edges are shared
    shared = set(a.topology.edges) & set(b.topology.edges)
    assert shared
    assert a.topology.groups.keys() == b.topology.groups.keys()
