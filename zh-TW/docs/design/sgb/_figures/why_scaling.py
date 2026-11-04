"""Figures of why-plan-first.qmd: measured scaling of the production pipeline.

Measures ``scgsim.geometry.sgb`` itself.  Each case runs in its own interpreter (so peak memory is that case's
own) on the Route B random layouts of ``tests/sgb_scenes.random_layout`` (seed 2, pitch 260 um): N = 1000, 5000 and
20000 features, nine schedules (one tile, and 4 x 4 and 8 x 8 tiles with 1, 8, 16 and 32 workers).  The one-tile
case also writes the CAD files with the direct writer (1 and 16 workers), opens the XAO in Gmsh, runs the load-side
check ``verify_loaded``.  A second mode replays the same models through the Gmsh API oracle
(``tests/sgb_gmsh_oracle``, the test-side bottom-up emitter) for the comparison of decision D8; its time grows
much faster than linearly (N = 5000 did not finish in 10 minutes in a trial run), so it is run only up to N = 2000.  Nothing here measures a global 3-D fragment.

Four modes::

    python why_scaling.py measure   # runs the cases (about 15 minutes) and writes figures/why-scaling-measured.json
    python why_scaling.py oracle    # the Gmsh-API oracle at N = 500, 1000, 2000 -> figures/why-scaling-oracle.json
    python why_scaling.py validators  # time of each structural validation check at N = 1000, 5000, 20000 -> figures/why-validator-scaling.json
    python why_scaling.py           # draws the figures and writes figures/why-scaling-numbers.json from those files

Writes ``figures/why-scaling.html`` and ``.svg`` (log-log time and memory against N), ``figures/why-tail.html`` and
``.svg`` (end-to-end stacks at N = 20000), ``figures/why-scaling-numbers.json`` and the raw
``figures/why-scaling-measured.json``.  ``arrangement_guarantees.py`` reads the raw file for the arrangement sizes.

The measuring half is ``why_scaling_measure`` (it also starts each case in its own interpreter); the plotting half is ``why_scaling_figures``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from why_scaling_measure import measure, measure_oracle, measure_validators

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Plot the scaling figures from the stored measurements, or run one measuring mode when an argument asks for it."""
    figures = HERE.parent / "figures"
    measured, oracle_file = (
        figures / "why-scaling-measured.json",
        figures / "why-scaling-oracle.json",
    )
    if len(sys.argv) > 1 and sys.argv[1] == "measure":
        measure(measured)
    elif len(sys.argv) > 1 and sys.argv[1] == "validators":
        measure_validators(figures / "why-validator-scaling.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "oracle":
        measure_oracle(oracle_file)
    else:
        plot_from_files(measured, oracle_file)


def plot_from_files(measured: Path, oracle_file: Path) -> None:
    """Draw the figures and write ``why-scaling-numbers.json`` from the measured and oracle files."""
    # the measuring modes never plot, and the plotting libraries are heavy
    from why_scaling_figures import load_measured, plot  # noqa: PLC0415 -- plotting libraries load only when figures are drawn

    oracle = {
        r["n_features"]: r for r in json.loads(oracle_file.read_text())["rows"]
    }  # N -> row
    plot(
        load_measured(measured),
        oracle,
        json.loads(measured.read_text()).get("machine", {}),
    )


if __name__ == "__main__":
    main()
