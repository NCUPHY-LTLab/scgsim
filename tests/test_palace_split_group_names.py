"""A split surface group keeps its split index in ``palace.msh``, and the geometry preview reads the run (needs Gmsh).

A Route A sheet band is split into one solver attribute per Surface-EPR row signature.  On the thin-film pair
with a bump on the left film, the bump covers part of a band of the left sheet: there the sheet has no vacuum
above, so those faces carry only the metal-substrate row and the band is split in two (``...::band=2#1`` and
``...#2``).  The scene meshes in about two seconds.
"""

from __future__ import annotations

import json

import pytest
from palace_sgb_support import mesh_run, surface_epr_specs
from sgb_inputs import thin_film_pair
from test_palace_mesh_end_to_end import config_of, physical_names

from scgsim.contracts.palace_run_folder import CONFIG, INDEX_MAP
from scgsim.palace.simulation.core import index_map

MARGINS = [0.25, 0.5]
SPECS = surface_epr_specs(MA=MARGINS, MS=MARGINS, SA=[0.5])


@pytest.fixture(scope="module")
def split_run(tmp_path_factory):
    """A meshed Route A run with a split band group, with its config and index map written beside the mesh."""
    directory = tmp_path_factory.mktemp("split")
    result = mesh_run(thin_film_pair(bump=True), "A", directory, specs=SPECS)
    config = config_of(result, "A", SPECS)
    run = result.output_dir
    (run / CONFIG).write_text(json.dumps(config.config), encoding="utf-8")
    (run / INDEX_MAP).write_text(
        json.dumps(
            index_map(
                config.index_entries, ground_boundary=config.ground_boundary_resolution
            )
        ),
        encoding="utf-8",
    )
    return result


def manifest_names(result):
    """(dimension, attribute) -> name of every solver attribute the mesh manifest lists."""
    manifest = json.loads(result.mesh_manifest_path.read_text(encoding="utf-8"))
    return {
        (entry["dimension"], entry["attributes"][0]): entry["name"]
        for entry in manifest["groups"]
    }


def test_the_mesh_names_every_attribute_as_the_manifest_does(split_run):
    """The split parts and every attribute numbered after them carry their manifest names in the file."""
    listed = manifest_names(split_run)
    assert {"chip/left::band=2#1", "chip/left::band=2#2"} <= set(listed.values())
    assert physical_names(split_run.mesh_path) == listed


def test_the_geometry_preview_reads_the_run(split_run):
    """``inspect_palace_geometry`` binds the mesh to the manifest and labels each attribute by its name."""
    pytest.importorskip("meshio")
    pytest.importorskip("pyvista")
    from scgsim.visualization.loaders.palace import (  # noqa: PLC0415 -- optional visualization dependency.
        inspect_palace_geometry,
    )

    preview = inspect_palace_geometry(split_run.output_dir)
    mesh_mode = preview._modes["mesh"]  # noqa: SLF001 -- the scene, not a rendered contact sheet
    labels = {part.label.split(" · ")[0] for part in mesh_mode.parts}
    assert labels == set(manifest_names(split_run).values())


def test_a_mesh_named_otherwise_than_its_manifest_is_refused_with_every_difference(
    split_run, tmp_path
):
    """A ``palace.msh`` whose name table was changed after writing: every differing tag is listed."""
    pytest.importorskip("meshio")
    pytest.importorskip("pyvista")
    from scgsim.visualization.loaders.palace import (  # noqa: PLC0415 -- optional visualization dependency.
        inspect_palace_geometry,
    )

    run = tmp_path / "run"
    for path in (split_run.mesh_path, split_run.mesh_manifest_path):
        target = run / path.relative_to(split_run.output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name in (CONFIG, INDEX_MAP):
        (run / name).write_bytes((split_run.output_dir / name).read_bytes())
    text = (run / "palace.msh").read_text(encoding="utf-8")
    # Drop the split index from both parts, as the mesh writer once did.
    (run / "palace.msh").write_text(
        text.replace('"chip/left::band=2#1"', '"chip/left::band=2"').replace(
            '"chip/left::band=2#2"', '"chip/left::band=2x"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as refusal:
        inspect_palace_geometry(run)
    message = str(refusal.value)
    assert message.startswith("internal: palace.msh names its physical tags otherwise")
    assert "['chip/left::band=2'] in palace.msh and ['chip/left::band=2#1']" in message
    assert "['chip/left::band=2x'] in palace.msh and ['chip/left::band=2#2']" in message
    assert "mesh()" not in message
