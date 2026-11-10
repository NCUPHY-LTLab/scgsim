# Palace stable baseline

These scripts measure, for each OrcaPen SC PDK layout, which stage of the Palace route it reaches and
where it stops: the plan, the SGB build, `simulation_drc` and `mesh()`. They also count the fine
structure SGB records for each built model. The results and how each stop is attributed are in the
experiment issue linked from the pull request that added this folder (issue 141, work step 2).

Measurement only: nothing here changes SCGSim, and no solver runs.

## What runs

| Script | What it does |
|---|---|
| `cases.py` | How each layout is built: component, layer stack, instances, Net assignment, simulation class. Also the supported banded scenes of `tests/sgb_supported_*`. |
| `run_case.py` | One layout through plan, SGB build, `simulation_drc` and `mesh()`, with Surface-EPR bands (Route B, MA/MS/SA at margins 0.5 and 1.0 um by type) or with `--no-bands`. `--scene` builds a test scene and measures its fine structure only. Writes one JSON record. |
| `run_baseline.py` | Every case in its own process, at most `--jobs` at a time (default 8), each stopped after `--timeout` seconds (default 900). Then every layout that stopped with bands again without bands, and every layout whose banded `mesh()` failed again with the vacuum the PDK's Palace notebooks use (`run_case.py --notebook-vacuum`). |
| `fine_structure.py` | Counts short edges, narrow faces and small angles from SGB's sliver report, split by whether a band curve is involved, and the classes A, B and C of issue 125. |
| `check_pdk47.py` | Looks for the drawing artifacts of orcapen-sc-pdk#47 (the eight 1 nm claw jogs and the 9.9 nm lead-fan step) in the Disk Qubit builds. |
| `tables.py` | Prints the stage, fine-structure and class tables of a records folder as Markdown. |

Settings are those `simulation_drc` fixes, so all stages see one model: plan root `chip`, zero coupon
padding, a 1 um vacuum region when the layout has none, every authored port at 10 nH, every layout at
`curve_sampling="simulation"`, and the default mesh controls (`FeatureRecipe()`, minSICN 0.01, the default
time budget). With `--notebook-vacuum` the plan instead gets the vacuum padding of the PDK's Palace
notebooks: (250, 250, 1000) um for a single die (SimplePad, CircularPad), (0, 0, 1000) um for a flip-chip
coupon (Kosen Xmon). `simulation_drc` keeps its own fixed settings in both.

## How to run

The PDK needs dependencies that SCGSim's own lock does not hold (`doroutes`, its gdsfactory pin). Build an
environment the way CI's `pdk-tests` job does: the PDK's frozen dependencies, then the PDK and SCGSim
without dependencies. `PDK_SRC` is a checkout of the PDK revision under test, outside this repository.

```bash
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$PDK_SRC" --frozen \
    --group dev --group palace-notebooks --no-install-project --no-install-package scgsim
uv pip install --python "$VENV/bin/python" --no-deps "$PDK_SRC"
uv pip install --python "$VENV/bin/python" --no-deps -e .
uv pip check --python "$VENV/bin/python"

"$VENV/bin/python" experiments/palace-stable-baseline/run_baseline.py "$RECORDS"
"$VENV/bin/python" experiments/palace-stable-baseline/check_pdk47.py "$RECORDS/pdk47.json"
python experiments/palace-stable-baseline/tables.py "$RECORDS"
```

`--only CASE ...` limits a run to some cases; `run_case.py CASE OUT.json` runs one.

The SGB build stage calls the geometry build that `simulation_drc` and `mesh()` share
(`_build_geometry`), because no public call keeps the built files. `fine_structure.capture` asks SGB's
sliver report for every entry instead of the ten it writes; the files SGB writes are unchanged.
