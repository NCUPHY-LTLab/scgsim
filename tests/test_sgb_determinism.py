"""Same input gives the same ids, across runs, interpreter hash seeds and layouts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sgb_scenes import pad_in_ground, random_layout

from scgsim.geometry.sgb.derivation.labelled import derive_from_catalogue

TESTS = Path(__file__).resolve().parent
SCRIPT = (
    f"import sys; sys.path.insert(0, {str(TESTS)!r})\n"
    "from scgsim.geometry.sgb.derivation.labelled import derive_from_catalogue\n"
    "from sgb_scenes import pad_in_ground, random_layout\n"
    "print(derive_from_catalogue(random_layout(40, 9, 'B')).identity_digest(),"
    " derive_from_catalogue(pad_in_ground('B')).identity_digest())\n"
)


def test_two_runs_give_identical_ids():
    """Building one catalogue twice yields one identity."""
    assert (
        derive_from_catalogue(pad_in_ground("B")).identity_digest()
        == derive_from_catalogue(pad_in_ground("B")).identity_digest()
    )
    catalogue = random_layout(40, seed=9, route="B")
    assert (
        derive_from_catalogue(catalogue).identity_digest()
        == derive_from_catalogue(catalogue).identity_digest()
    )


def test_ids_do_not_depend_on_python_hash_randomisation():
    """Set and dict ordering never reaches an id: three hash seeds, one digest."""
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


def test_ids_are_content_addressed_not_positional():
    """Unrelated geometry keeps its ids when another feature of the model changes."""
    full = derive_from_catalogue(pad_in_ground("A"))
    plain = derive_from_catalogue(pad_in_ground("A", rounded_corner=False))
    shared = set(full.level_layers[1].edge_ids) & set(plain.level_layers[1].edge_ids)
    assert shared  # the ground plane's outer boundary and cutout are untouched
    assert set(full.level_layers[1].vertex_ids) & set(plain.level_layers[1].vertex_ids)
