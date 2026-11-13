"""Same input gives the same ids, across runs, interpreter hash seeds and layouts."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from sgb_scenes import pad_in_ground, random_layout, slab_model

from scgsim.geometry.sgb.arrangement.derivation.labelled import derive_from_catalogue
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.exact.grid import Grid

TESTS = Path(__file__).resolve().parent
SCRIPT = (
    f"import sys; sys.path.insert(0, {str(TESTS)!r})\n"
    "from scgsim.geometry.sgb.arrangement.derivation.labelled import derive_from_catalogue\n"
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


MODEL_SCRIPT = (
    f"import sys; sys.path.insert(0, {str(TESTS)!r})\n"
    "from scgsim.geometry.sgb.model import build_model_from_catalogue\n"
    "from sgb_scenes import pad_in_ground, random_layout\n"
    "print(build_model_from_catalogue(pad_in_ground('B'), 'B').identity_digest(),"
    " build_model_from_catalogue(random_layout(30, 5, 'C'), 'C').identity_digest())\n"
)


def test_model_ids_do_not_depend_on_python_hash_randomisation():
    """Vertices, edges, faces, volumes and groups: three hash seeds, one digest."""
    outputs = set()
    for hash_seed in ("0", "3", "999"):
        result = subprocess.run(
            [sys.executable, "-c", MODEL_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


def test_the_model_digest_includes_the_physical_grid():
    """One lattice at two grid sizes has the same entity ids but areas four times apart: the digests differ."""
    catalogue = slab_model("B", [("pad", "P", (-4, -4, 4, 4))], half=10)
    models = {
        size_um: build_model_from_catalogue(replace(catalogue, grid=Grid(size_um)), "B")
        for size_um in (0.001, 0.002)
    }
    fine, coarse = models[0.001], models[0.002]
    assert (
        fine.identity_sets() == coarse.identity_sets()
    )  # the ids alone cannot tell them apart
    assert fine.identity_digest() != coarse.identity_digest()
    again = build_model_from_catalogue(replace(catalogue, grid=Grid(0.001)), "B")
    assert again.identity_digest() == fine.identity_digest()


def test_wall_ids_do_not_depend_on_the_vertical_edges_next_to_them():
    """A wall id hashes its horizontal chains and column end points only, so a port that cuts *other* walls keeps the rest."""
    from sgb_scenes import lumped_port, rect_um  # noqa: PLC0415 -- helpers used by this test only

    from scgsim.geometry.sgb.exact.prisms import PrismCatalogue  # noqa: PLC0415 -- used by this test only

    base = pad_in_ground("B", with_port=False)

    def walls(x0, x1):
        """Walls."""
        catalogue = PrismCatalogue(
            base.grid, (*base.prisms, lumped_port("p1", rect_um(x0, -5, x1, 5), "B"))
        )
        topology = build_model_from_catalogue(catalogue, "B").topology
        return {f.id for f in topology.faces.values() if f.kind == "wall"}, topology

    floating, _ = walls(55, 75)
    touching, topology = walls(50, 80)
    assert (
        len(floating & touching) == len(floating) - 2
    )  # only the two cut walls change their ids
    columns = [e for e in topology.edges.values() if e.kind == "vertical"]
    assert columns  # the vertical edges differ between the two models while the other walls keep their ids
