"""Figures and tables for ``docs/design/sgb/emission.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/emission.py            # figures and tables
    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/emission.py --measure  # also rerun the timings
    ... emission.py --measure-only                                                                           # only the timings
    ... emission.py --occt-only                                                                              # only the OCCT and catalogue tables

Everything is computed by the production package.  The running example is the pad-in-ground scene of
``tests/sgb_scenes.py`` under Route B, its lumped port on the support plane z = 0 (see ``inset_kit.pad_catalogue``): it is built,
written with ``emit_model`` into a scratch folder, numbered by ``plan_brep`` (whose numbering is checked against the written
sidecar's ``ids_by_tag`` and the written BREP's shell records) and reopened in Gmsh.  The timings are measured by this script
(``--measure``, one subprocess per N so the peak memory is honest) and stored in ``figures/emission-timing-data.json``; the other
runs read that file.  Only what ``figures/why-scaling-measured.json`` (``why_scaling.py``) does not hold is measured here: the Gmsh-API
oracle stage by stage next to the direct writer at N = 500, 1000 and 2000, and the time of every check at N = 1000, 5000 and 20000.
The direct writer's tail (plan, render with 1 and 16 workers, reports, sidecar, Gmsh load, ``verify_loaded``, file sizes) is read from
that shared file.  The OCCT checks use OCP (the ``occt`` group of the SGB core).  Outputs go to ``docs/design/sgb/figures/``
with the prefix ``emission-``:

* ``emission-timing``         write and load cost of the Gmsh-API oracle emitter against the direct writer (html + svg)
* ``emission-shared-face``    one shared face referenced + by a shell and - by the other (svg; schematic plus real references)
* ``emission-groups-3d``      the reopened XAO meshed on its surfaces, coloured by physical group (html + svg)
* ``emission-*-table.md``     markdown tables included by the page (counts, timing, sidecar, tags, groups, validation, OCCT, bands, ...)
* ``emission-example-numbers.json``  the counts of the running example that the page quotes
* ``emission-step-table.md``  is NOT written here: it is the frozen STEP round-trip table of the AEDT experiment, included by ``aedt-handoff.qmd``

The work is split over ``emission_common`` (paths and the running example), ``emission_figures``, ``emission_tables``, ``emission_checks`` and
``emission_measure`` (the measuring children this script starts with ``--head-one`` and ``--validator-one``).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from emission_checks import extreme_tables, occt_tables, validation_table
from emission_common import (
    DATA,
    FIGURES,
    check_shell_references,
    example_numbers,
    running_example,
)
from emission_figures import (
    five_edge_wall_figure,
    groups_3d_figures,
    shared_face_figure,
    timing_figures,
)
from emission_measure import head_one, measure, validator_one
from emission_tables import (
    counts_table,
    tags_and_groups_tables,
    timing_tables,
    validator_tables,
)

# --------------------------------------------------------------------------- main


def main() -> None:
    """Write the emission page's figures, tables and numbers, or run one measuring part when a flag asks for it."""
    for flag, function in (
        ("--head-one", head_one),
        ("--validator-one", validator_one),
    ):
        if flag in sys.argv:
            print(json.dumps(function(int(sys.argv[sys.argv.index(flag) + 1]))))
            return
    if "--measure-only" in sys.argv:
        measure()
        return
    if "--occt-only" in sys.argv:
        print(json.dumps({"examples": occt_tables(), "catalogue": extreme_tables()}))
        return
    kit.ensure_plotly_js()
    if "--measure" in sys.argv or not DATA.exists():
        measure()
    numbers: dict = {}
    with tempfile.TemporaryDirectory(prefix="emission-example-") as scratch:
        model, plan, sidecar, artifacts = running_example(Path(scratch))
        check_shell_references(plan, artifacts.brep)
        numbers = example_numbers(model, plan, sidecar)
        numbers["sidecar_bytes"] = artifacts.sidecar.stat().st_size
        numbers["bytes"] = {
            "brep": artifacts.brep.stat().st_size,
            "xao": artifacts.xao.stat().st_size,
        }
        counts_table(model, plan)
        tags_and_groups_tables(model.topology, plan, sidecar)
        numbers["shared_face"] = shared_face_figure(model.topology, plan)
        numbers["groups_3d"] = groups_3d_figures(plan, artifacts.xao)
    data = json.loads(DATA.read_text())
    timing_figures(data["head"])
    numbers["timing"] = timing_tables(data["head"], data["validator"])
    numbers["validator"] = validator_tables(data["validator"])
    validation_table()
    numbers["occt"] = occt_tables()
    numbers["catalogue"] = extreme_tables()
    five_edge_wall_figure()
    (FIGURES / "emission-example-numbers.json").write_text(
        json.dumps(numbers, indent=1, default=str)
    )
    print("wrote emission-example-numbers.json")


if __name__ == "__main__":
    main()
