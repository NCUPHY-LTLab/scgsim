"""Figures and tables for ``docs/design/sgb/inset-imprint.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/inset.py

Every geometry figure is computed by the production package: the pad-in-ground running example
(``tests/sgb_scenes.py::pad_in_ground``, its lumped port on the support plane z = 0) is built with
Surface-EPR inset bands through ``build_model_from_catalogue`` and read from the labelled level arrangement,
the assembled topology and the route plan.  The KLayout cross-check uses the test-side oracle
``tests/sgb_klayout.py``; the mesh check meshes the written XAO with Gmsh (no Palace and no AEDT is run).
Outputs go to ``docs/design/sgb/figures/`` with the prefix ``inset-``:

* ``inset-erosion-construction``  corner rule on large margins (computed with margins 1.5, 3, 6 um)
* ``inset-bands-pad``             the z = 0 level arrangement with bands (margins 0.1, 0.5, 2 um); html + svg
* ``inset-arcs-vs-polygon``       exact offset arcs against a 512-gon erosion, radial deviation; html + svg
* ``inset-topology-change``       margins 4, 9, 30 um: a neck splits, a band vanishes; html + svg
* ``inset-mask-vs-imprint``       an element cut by a mask against a mesh aligned to the band edge, and the
                                  area error of both as the element size shrinks (2-D gmsh illustration)
* ``inset-routes-section``        conceptual cross-section of Route A (one sheet) and Route B (top and bottom faces)
* ``inset-tables.md``             markdown tables quoted by the page (band areas, KLayout cross-check, mesh check)

The work is split over ``inset_figures``, ``inset_mask`` and ``inset_tables``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from inset_figures import (
    figure_arcs,
    figure_bands,
    figure_construction,
    figure_topology_change,
)
from inset_kit import MARGINS  # noqa: E402
from inset_mask import figure_mask_vs_imprint, figure_routes_section, mesh_check
from inset_tables import (
    arc_deviation,
    build_tables,
    extreme_band_comparisons,
    noded_edges,
    route_b_comparison,
)
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402


def main() -> None:
    """Write the inset-imprint figures, tables and numbers."""
    kit.ensure_plotly_js()
    model_a = kit.pad_model("A", MARGINS)
    figure_construction()
    figure_bands(model_a)
    figure_arcs(model_a)
    topology_change = figure_topology_change()
    mask_rows = figure_mask_vs_imprint()
    figure_routes_section()
    meshes = {
        "A": mesh_check(model_a, 40.0),
        "B": mesh_check(kit.pad_model("B", MARGINS), 60.0),
        "C": mesh_check(
            build_model_from_catalogue(
                kit.pad_catalogue("B"), "C", insets=kit.pad_requests("B")
            ),
            60.0,
        ),
    }
    build_tables(model_a, topology_change, mask_rows, meshes)
    extra = {
        "arc_deviation": arc_deviation(model_a),
        "level_z0": noded_edges(),
        "extreme_band_comparisons": extreme_band_comparisons(),
        "route_b_comparison": route_b_comparison(),
    }
    (kit.FIGURES / "inset-numbers.json").write_text(json.dumps(extra, indent=1))
    print(json.dumps(extra))


if __name__ == "__main__":
    main()
