# Palace stable baseline

These scripts measure, for each OrcaPen SC PDK layout, which stage of the Palace route it reaches and
where it stops: the plan, the SGB build, `simulation_drc` and `mesh()`. They also count the fine
structure SGB records for each built model. They are the scripts behind step 2 ("Baseline") of
[issue 141](https://github.com/NCUPHY-LTLab/scgsim/issues/141).

Measurement only: nothing here changes SCGSim, and no solver runs.

The five roster layouts of issue 141 are also checked every night: the Nightly workflow's `roster` job
runs `scripts/check_roster.py`, which takes each through `simulation_drc` and `mesh()` with bands and
fails when one does not pass. The roster and the run settings live in `scripts/roster_cases.py`, which
these scripts import.

## Where the results are

The stage table, the fine-structure table, the classes of issue 125 and the attribution of every stop
are in [issue 149](https://github.com/NCUPHY-LTLab/scgsim/issues/149); its comments hold the per-case
records. They were measured on develop `1bc64ca` with the PDK at `49e539f`, in a 1 um vacuum shell, and
each failed mesh again in the vacuum of the PDK's Palace notebooks. The scripts now run on a
later develop, where several of those numbers differ; [What differs on develop now](#what-differs-on-develop-now)
says which, and why.

## What runs

| Script | What it does |
|---|---|
| `cases.py` | How each layout is built: component, layer stack, instances, Net assignment, simulation class. The roster comes from `scripts/roster_cases.py`; the other layouts are built here with its helpers. Also the banded scenes of `tests/sgb_supported_*`. |
| `run_case.py` | One layout through plan, SGB build, `simulation_drc` and `mesh()`, with Surface-EPR bands (Route B, MA/MS/SA at margins 0.5 and 1.0 um by type) or with `--no-bands`. `--scene` builds a test scene and measures its fine structure only. Writes one JSON record. |
| `run_baseline.py` | Every case in its own process, at most `--jobs` at a time (default 8), each stopped after `--timeout` seconds (default 900). Then every layout that stopped with bands again without bands. It removes a case's earlier record before the case starts, so a rerun into the same folder never shows stages it did not run. |
| `check_records.py` | Checks that a record says only what its run did: a case that stops before its first record leaves no earlier run's stages, a record read while it is rewritten is always complete, and an unreadable record counts as one with no stage. Needs no PDK. |
| `fine_structure.py` | Counts short edges, narrow faces and small angles from SGB's sliver report, split by whether a band curve is involved, and the classes A, B and C of issue 125. An arc edge, and the column gap of a curved wall, is measured along its arc. |
| `check_pdk47.py` | Looks for the drawing artifacts of orcapen-sc-pdk#47 (the eight 1 nm claw jogs and the 9.9 nm lead-fan step) in the Disk Qubit builds. |
| `tables.py` | Prints the stage, fine-structure and class tables of a records folder as Markdown. |

Every record is written whole, through a temporary file in the same folder, so a case the time limit
stops mid-write leaves its previous complete record. Each stage of a record says `pass` or the exact error
class and message, and its wall time. The
`simulation_drc` stage also keeps the findings of its meshing stage (the fine edges and faces the mesher
cannot use, with their owner and place): the warnings of a pass, or the refusals of the
`SimulationDrcError` it raises. The mesh stage keeps the recipe, the counts, the stop reason and the stop
places of the quality report.

The settings are those of `scripts/roster_cases.py`, the same for every layout: plan root `chip` and zero
coupon padding, as `simulation_drc` builds its plan; the vacuum the Human set for testing layouts
([issue 182](https://github.com/NCUPHY-LTLab/scgsim/issues/182)), padding 0 um in x and y and 1000 um below
and above; every authored port at 10 nH; every layout at `curve_sampling="simulation"`; and the default mesh
controls (`FeatureRecipe()`, minSICN 0.01, the default time budget). The SGB build and `mesh()` use them;
`simulation_drc` sets its own vacuum.

## How to run

The PDK needs dependencies that SCGSim's own lock does not hold (`doroutes`, its gdsfactory pin). Build an
environment the way CI's `pdk-tests` job does: the PDK's frozen dependencies, then the PDK and SCGSim
without dependencies. `PDK_SRC` is a checkout of the PDK revision under test (the one CI's `pdk-tests` job
pins), outside this repository; `VENV` and `RECORDS` are folders outside it too.

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

`--only CASE ...` limits a run to some cases; `run_case.py CASE OUT.json` runs one. Give each running
`run_baseline.py` its own records folder: two at once in one folder would write the same records. The cheapest check
that every stage works is SimplePad and one scene, about 25 s:

```bash
"$VENV/bin/python" experiments/palace-stable-baseline/run_baseline.py "$RECORDS" --only simple_pad cell01_disk
```

The SGB build stage calls the geometry build that `simulation_drc` and `mesh()` share
(`_build_geometry`), because no public call keeps the built files. `fine_structure.capture` asks SGB's
sliver report for every entry instead of the ten it writes; the files SGB writes are unchanged.

## What differs on develop now

The scripts were last run on develop `464ba5b`, through the geometry and meshing stages only, with the
PDK at `49e539f`: SimplePad, every banded scene, `check_pdk47.py`, `bend_euler` and the B-spline strip
polygon. The results on issue 149 stay as measured. A rerun on this develop gives these differences:

| What changed on develop | Effect on a rerun | Measured (on `464ba5b` unless stated) |
|---|---|---|
| The default `FeatureRecipe` is `scgsim.meshing.features/2`: every Surface-EPR band boundary curve is sized at 4 × the narrowest band width (PR 158). A model without bands keeps the features/1 sizes. | Every banded `mesh()` row changes. The `--no-bands` rows do not. | SimplePad: 170,631 tetrahedra, minSICN 0.0122, 0 rounds, 16 s of meshing (issue 149: 107,094, 0.0138, 0 rounds, 9 s). |
| `simulation_drc` adds a meshing stage with the fine-structure limits of issue 125, raises `SimulationDrcError` on its refusal, and `mesh()` runs the same check before the mesher starts (PR 147). | A layout the meshing stage refuses now stops at `simulation_drc`, and its `mesh()` stops before meshing. Issue 149's `simulation_drc` column is the geometry stage alone. | SimplePad: the meshing stage passes with no finding. The B-spline strip polygon: it passes with 3 warnings. |
| The rule of at most 128 points per full circle is gone (PR 148). | The two spline strip polygons with bands are no longer refused at the SGB build. | B-spline strip polygon with bands: builds, and meshes clean (84,100 tetrahedra, minSICN 0.0167, 3 rounds). The interpolation strip polygon was not rerun. |
| Two straight band offsets that meet at a convex junction end at their crossing (PR 148). | Fewer short band edges in the scenes. | Band edges under 25 grid steps, issue 149 → now: `cell14_euler_bend` 36 → 0, `cell15_euler_u_turn` 84 → 0, `cell16_coupled_pair` 72 → 0, `cell18_taper` 24 → 0, `cell18_handoff` 12 → 6, `cell21_bspline_resampled` 45 → 9, `cell21_interpolation_resampled` 120 → 6, `junction_fan`, `junction_fan_noport` and `d1_die_defaults` 7 → 5. Unchanged: `junction_straight` and `d1_die_test_build` (8, with the 0.71 nm face and 0.0007° wedge); the other scenes still have none. |
| `cell24_sheet_circular_pad` left the supported list, because its files cannot be written (PR 155). | `cases.scene_builders` keeps it, so its row is still measured. | Unchanged: 2 source edges under 25 grid steps, 10.1 nm, 5.6°. |
| Every layout is meshed in the vacuum of issue 182, padding (0, 0, 1000) um, instead of the 1 um shell, and the notebook-vacuum rerun is gone (issues 179 and 182). | Every `mesh()` row changes, and an SGB build's fine structure can. The `simulation_drc` rows do not: it sets its own vacuum. | Measured on `8c9ca76` with `scripts/check_roster.py`, the roster with bands, all clean in 0 rounds: SimplePad 117,138 tetrahedra, minSICN 0.0215, 9 s of meshing; circular pad 86,957, 0.0107, 7 s; airbridge coupon 52,444, 0.0105, 4 s; flip-chip pad 89,758, 0.0131, 7 s; Disk Qubit 569,499, 0.0228, 57 s (in the 1 um shell it stopped at the 180 s limit). |

Not rerun on `464ba5b`: the other layouts, the `--no-bands` pass, and the notebook-vacuum pass the scripts
had then. The SGB
build alone (no `simulation_drc`, no mesh) of the Kosen Xmon with bands no longer has its 8 band edges of
2.2 to 2.8 nm: its shortest band edge is 37.4 nm, and none is under 25 grid steps. The Palace readers that
now keep the requested modes (PR 157) are not used here, because no solver runs.

## Correction: lengths along the arc

SGB's sliver report gives every edge as the distance between its end points. Before this correction,
`fine_structure.py` and `check_pdk47.py` counted and listed an arc edge by that chord, and measured a
curved wall's column gap the same way. They now measure along the arc (`2 r asin(chord / 2r)`, as
`tests/sgb_supported_kit.py` does).

No count, class or attribution moves. On develop `464ba5b`, SimplePad, every banded scene and the four
Disk Qubit builds of `check_pdk47.py` give the same counts at 2, 10 and 25 grid steps before and after. So
do the SGB builds with bands of the other layouts in issue 149's fine-structure table. Smallest values
that are arcs grow. In issue 149's fine-structure table, measured by chord → along the arc:

| case | shortest band edge |
|---|---|
| SimplePad | 707 nm → 785 nm (the quarter circle a 0.5 um band draws around a square corner) |
| flip-chip pad | 707 nm → 785 nm (the same corner) |
| scene `cell24_sheet_circular_pad` | 97.4 nm → 97.5 nm (issue 149: 97 nm) |

The other rows of that table do not move at the precision issue 149 gives: the circular pad (981 nm), the
airbridge coupon (500 um), the Disk Qubits (36.8 nm) and the scenes. In the per-case records on issue 149,
every `smallest` edge entry that is an arc grows the same way, by up to 11 % (a quarter circle).
