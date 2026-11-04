"""Figures and tables for ``docs/design/sgb/parallelism.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/parallel.py

The tile ownership of the running example (the z = 0 level of the Route A pad-in-ground with bands) is computed with the production
runtime: the pre-dissolution arrangement of the level (``keep_raw``), the model's tile cut lines (``noding.tiling.tile_cut_lines``) and
the owner of every hot pixel (``tile_index``).  The snap-round counts are read from the arrangements of the seven running examples.

The scaling data are not measured here.  They are read from ``figures/why-scaling-measured.json`` (``why_scaling.py measure``): 27 cases
of random Route B layouts of seed 2 (N = 1000, 5000, 20000; 1 x 1, 4 x 4 and 8 x 8 tiles; 1, 8, 16 and 32 workers), each in its own
interpreter so that the peak memory is that case's own, built with ``build_model_from_catalogue(..., tiles=T, workers=W)``.  The one-tile,
one-worker case of each N also wrote the files with ``emit_model``, opened the XAO in Gmsh and ran ``verify_loaded``; its step times
are the emission and load steps quoted here.  Outputs go to ``docs/design/sgb/figures/`` (prefix ``parallel-``):

* ``parallel-tiles``      between-pixel tile cuts on the z = 0 level of the running example (computed) and a
                          conceptual pixel-level zoom; html + svg
* ``parallel-speedup``    measured speed-up against workers with the Amdahl curve, and the time of every stage of the
                          one-process run at N = 20000 followed by the emission and load steps; html + svg
* ``parallel-tables.md``  every table and range of the data in English, for checking
* ``parallel-table-*.md`` the tables the page includes: ``excerpt``, ``full``, ``stages``, ``workers``, ``endtoend``, ``memory``, ``tiles``
* ``parallel-numbers.json``  the numbers of the text that are not in a table

The work is split over ``parallel_figures`` and ``parallel_tables``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from parallel_figures import figure_speedup, figure_tiles, snap_round_histogram
from parallel_tables import page_tables, scaling_rows, scaling_summary, speedups, tables


def main() -> None:
    """Write the parallelism figures, tables and numbers."""
    kit.ensure_plotly_js()
    model = kit.pad_model("A", kit.MARGINS, keep_raw=True)
    tile_stats = figure_tiles(model)
    scaling = scaling_rows()
    rows = speedups(scaling)
    extra = figure_speedup(scaling, rows)
    (kit.FIGURES / "parallel-tables.md").write_text(
        tables(scaling, rows, tile_stats, extra)
    )
    page_tables(scaling, rows, tile_stats)
    numbers = {
        "snap_rounds": snap_round_histogram(),
        "tiles": tile_stats,
        "scaling": scaling_summary(scaling, rows),
    }
    (kit.FIGURES / "parallel-numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers["snap_rounds"]))
    print("wrote parallel-*")


if __name__ == "__main__":
    main()
