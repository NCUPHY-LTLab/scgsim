# Meshing benchmark

This benchmark meshes seven fixed SGB cases with `scgsim.meshing`, through its three sizing recipes
(`FeatureRecipe`, `GlobalSizeRecipe`, `UniformBoundaryRecipe`), and records for each run the recipe and
its version, the 2D and 3D algorithms, tetrahedron count, minimum quality, elements below the
threshold, repair rounds, status, time and whether the other 3D algorithm had to be used as a
fallback. Run it again after every change of a recipe, the repair loop or the mesher options and
compare with `baseline.json`.

## Run it

From the repository root, with the test and Palace dependencies synced
(`uv sync --locked --group test --group occt --extra palace`):

```bash
uv run python experiments/meshing-benchmark/run.py OUTPUT_DIR \
    --compare experiments/meshing-benchmark/baseline.json
```

The full run takes about 7 minutes on one workstation, mostly the HXT column on the 2 mm and 4 mm chips.
`results.json` records the Gmsh version once, in its `environment` section; each run lists its recipe
(name and version), `algorithm_2d` (`null` is Gmsh's default) and `algorithm_3d` (the one that made the
mesh, after any fallback).
Useful options:

- `--cases B circB` runs only those cases (`A B C circB B_1mm circB_2mm circB_4mm`).
- `--no-determinism` runs the module once per case instead of twice.
- `--write-baseline PATH` writes the compact numbers of the run, as `baseline.json` was written.

`OUTPUT_DIR` receives `results.json` (with the full quality report of every module run), the built
XAO and sidecar of each case under `cases/`, and the meshes under `meshes/`. `--compare` prints every
difference in tetrahedra, minimum quality, bad count, rounds and stop reason, and exits with 1 if
there is any. Times are not compared; they depend on the machine.

## Cases

All cases carry bands (0.5 and 1.0 um) on every conductor surface and on the substrate-vacuum
interface, and a lumped port.

| case | what it is |
|---|---|
| `A` | Route A: two thin-film sheets joined by a port |
| `B`, `C` | 100 um chip, a 20 um pad in a 40 um cutout of the ground, Route B and Route C |
| `circB` | the same with the pad rebuilt as an exact circle (arc edges) |
| `B_1mm` | the same pad and cutout on a 1 mm chip |
| `circB_2mm`, `circB_4mm` | the circular pad on 2 mm and 4 mm chips |

The metal is 0.2 um thick. The large chips scale only the die, the substrate and the vacuum padding
(`die_um` of `gds_pad_in_ground` in `tests/sgb_inputs.py`), so the film walls stay 0.2 um thin in a
model four orders of magnitude larger. That is where near-flat tetrahedra appear.

## Methods

Every method is one `mesh_sgb` call with the provisional threshold **0.01 on minSICN** (the quality
measure and bound are decision C, which is still open). The recipe chooses only the initial sizes;
repair, the threshold check and the report are the same for all of them.

- **module** is `FeatureRecipe()` (`scgsim.meshing.features/1`: a priori film-wall sizes), the default
  `RepairPolicy` and `Budgets`, and 3D Delaunay.
- **module + conductor-outline** is the same with `FeatureRecipe(conductor_outline=True)`, the optional
  rule that sizes the conductors' outline edges at 5 um, growing to the coarse size over 300 um.
- **module, HXT** is the same with `MesherOptions(algorithm_3d=10)`, the algorithm the module used
  before the timing study below.
- **R8** is `GlobalSizeRecipe()` (`scgsim.meshing.global-size/1`: only the global maximum size,
  bounding-box diagonal / 20, no curve sizes), HXT, and `Budgets(max_rounds=0)`: the cheap reference.
- **RP** is `UniformBoundaryRecipe()` (`scgsim.meshing.uniform-boundary/1`: 5 um on every curve growing
  to 300 um, as Palace does today), 2D algorithm 5, 3D Delaunay, and `Budgets(max_rounds=0)`. It uses
  the module's own Distance/Threshold field, so its element counts differ slightly from the earlier
  prototype's.
- **module, repair only** (`B_1mm` only) is the module with the `film_sidewall` features removed from
  the sidecar, so the repair loop must find the film walls from the bad elements alone.

Every method runs in a fresh worker process.

### How to read "rejected"

`Budgets(max_rounds=0)` turns repair off, not the threshold: R8 and RP are one mesh, and a mesh with a
tetrahedron below the threshold is **rejected** (`MeshRepairError`; the status column shows
`rejected (rounds)`, the stop reason of a loop whose round budget is spent). Their tetrahedra, minimum
SICN and count below the threshold are read from the report the error carries, so a rejected run
still has its numbers. A rejected run is not a failure of the benchmark; it says that the recipe on its own
does not meet the threshold on that case. The earlier table gave the references no status (their stop
reason was `-`); their numbers are unchanged, and only R8 on the three large chips is now rejected.

## Result

Gmsh 4.15.2, threshold 0.01 (provisional). "below" counts tetrahedra under the threshold; "recipe" is
the recipe's version without the `scgsim.meshing.` prefix; "2D / 3D" are the Gmsh algorithms of the mesh
(`default` is Gmsh's own 2D choice). Seconds are the whole `mesh_sgb` call, so the R8 and RP times
include the extent probe and the audit's attribution of bad elements, which the earlier table did not.
The repair loop is the local one of issue 124 (a size around each bad element). Route A (`A`) meshes
differently from the earlier table already on the development branch before that change: its numbers
here are those of the current geometry.

| case | method | recipe | 2D / 3D | tetrahedra | min SICN | below | rounds | status | seconds | fallback |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| A | module | features/1 | default / 1 | 6,688 | 0.2133 | 0 | 0 | accepted | 0.5 | no |
| A | module + conductor-outline | features/1 | default / 1 | 6,688 | 0.2133 | 0 | 0 | accepted | 0.5 | no |
| A | module, HXT | features/1 | default / 10 | 6,838 | 0.3114 | 0 | 0 | accepted | 0.5 | no |
| A | R8 | global-size/1 | default / 10 | 6,838 | 0.3114 | 0 | 0 | accepted | 0.5 | no |
| A | RP | uniform-boundary/1 | 5 / 1 | 821 | 0.05472 | 0 | 0 | accepted | 0.5 | no |
| B | module | features/1 | default / 1 | 13,601 | 0.03635 | 0 | 0 | accepted | 1.0 | no |
| B | module + conductor-outline | features/1 | default / 1 | 40,339 | 0.04522 | 0 | 0 | accepted | 1.4 | no |
| B | module, HXT | features/1 | default / 10 | 10,453 | 0.01916 | 0 | 0 | accepted | 5.5 | no |
| B | R8 | global-size/1 | default / 10 | 7,365 | 0.01982 | 0 | 0 | accepted | 2.3 | no |
| B | RP | uniform-boundary/1 | 5 / 1 | 8,491 | 0.02078 | 0 | 0 | accepted | 0.9 | no |
| C | module | features/1 | default / 1 | 17,208 | 0.02696 | 0 | 0 | accepted | 1.1 | no |
| C | module + conductor-outline | features/1 | default / 1 | 45,056 | 0.04522 | 0 | 0 | accepted | 1.4 | no |
| C | module, HXT | features/1 | default / 10 | 13,959 | 0.01916 | 0 | 0 | accepted | 9.6 | no |
| C | R8 | global-size/1 | default / 10 | 8,737 | 0.01982 | 0 | 0 | accepted | 5.0 | no |
| C | RP | uniform-boundary/1 | 5 / 1 | 10,508 | 0.02078 | 0 | 0 | accepted | 0.9 | no |
| circB | module | features/1 | default / 1 | 13,021 | 0.02769 | 0 | 0 | accepted | 1.0 | no |
| circB | module + conductor-outline | features/1 | default / 1 | 39,585 | 0.04998 | 0 | 0 | accepted | 1.3 | no |
| circB | module, HXT | features/1 | default / 10 | 10,104 | 0.01588 | 0 | 0 | accepted | 5.0 | no |
| circB | R8 | global-size/1 | default / 10 | 7,220 | 0.01991 | 0 | 0 | accepted | 1.6 | no |
| circB | RP | uniform-boundary/1 | 5 / 1 | 8,235 | 0.02134 | 0 | 0 | accepted | 0.9 | no |
| B_1mm | module | features/1 | default / 1 | 52,338 | 0.02563 | 0 | 0 | accepted | 2.9 | no |
| B_1mm | module + conductor-outline | features/1 | default / 1 | 149,218 | 0.02047 | 0 | 0 | accepted | 4.7 | no |
| B_1mm | module, HXT | features/1 | default / 10 | 33,795 | 0.01716 | 0 | 0 | accepted | 24.6 | no |
| B_1mm | R8 | global-size/1 | default / 10 | 9,664 | 0.002063 | 123 | 0 | rejected (rounds) | 2.3 | no |
| B_1mm | RP | uniform-boundary/1 | 5 / 1 | 100,227 | 0.01607 | 0 | 0 | accepted | 4.9 | no |
| B_1mm | module, repair only | features/1 | default / 1 | 26,310 | 0.009641 | 2 | 7 | rejected (stalled) | 6.5 | no |
| circB_2mm | module | features/1 | default / 1 | 98,678 | 0.01878 | 0 | 0 | accepted | 5.2 | no |
| circB_2mm | module + conductor-outline | features/1 | default / 1 | 127,251 | 0.0252 | 0 | 0 | accepted | 6.7 | no |
| circB_2mm | module, HXT | features/1 | default / 10 | 61,787 | 0.01693 | 0 | 0 | accepted | 48.3 | no |
| circB_2mm | R8 | global-size/1 | default / 10 | 9,344 | 0.001036 | 120 | 0 | rejected (rounds) | 2.6 | no |
| circB_2mm | RP | uniform-boundary/1 | 5 / 1 | 205,692 | 0.01501 | 0 | 0 | accepted | 10.8 | no |
| circB_4mm | module | features/1 | default / 1 | 187,568 | 0.01016 | 0 | 0 | accepted | 10.9 | no |
| circB_4mm | module + conductor-outline | features/1 | default / 1 | 187,568 | 0.01016 | 0 | 0 | accepted | 13.2 | no |
| circB_4mm | module, HXT | features/1 | default / 10 | 115,766 | 0.01515 | 0 | 0 | accepted | 111.8 | no |
| circB_4mm | R8 | global-size/1 | default / 10 | 8,579 | 0.0004609 | 122 | 0 | rejected (rounds) | 1.5 | no |
| circB_4mm | RP | uniform-boundary/1 | 5 / 1 | 419,045 | 0.01199 | 0 | 0 | accepted | 24.1 | no |

What the table shows:

- **R8 is not enough on the large chips.** HXT with only a global size leaves 120 to 123 near-flat
  tetrahedra at the film walls on the 1, 2 and 4 mm chips, and none on the 100 um chips. The three
  large-chip runs are the ones marked `rejected (rounds)`: the same numbers the earlier benchmark
  printed for R8, now judged by the threshold.
- **The a priori film rule removes them without any repair round.** Every module run is accepted in round
  0, and takes 2.6 s (1 mm), 5.1 s (2 mm) and 11.1 s (4 mm). On the 4 mm chip the module has 45 % of the
  tetrahedra of RP (187,568 against 419,045) and half of its time.
- **The local repair loop does not replace the film rule.** Without the film facts, the loop alone brings
  `B_1mm` from 123 bad elements to 2 and stops after 7 rounds because the count stopped falling
  (`stalled`). The loop before issue 124 refined each touched film wall along its whole length and
  cleaned this case in 3 rounds; the local sizes refine around each element, and new near-flat elements
  keep appearing further along the 0.2 um walls. With the film facts (the default) no repair is needed.
- **The 4 mm minimum is close to the threshold.** Delaunay gives 0.01016 against 0.01 on `circB_4mm`, where
  HXT gives 0.01515. A stricter bound than 0.01 will make the loop work on the large chips.
- **The module uses more tetrahedra than R8 on the small cases** (1.8 times on `B`), because film walls
  get a size even where R8 was clean. A wall gets no size when 25 times its thickness is not below half
  the coarse size, so `A` and the substrate walls are left alone.
- **The conductor-outline rule costs about three times the tetrahedra on the 100 um and 1 mm chips,** with
  a better minimum quality on the 100 um chips (0.045 against 0.036 on `B`). On the 4 mm chip it changes
  nothing: the film field, which grows by 1 um per um, is already finer than the outline field
  everywhere, so the minimum of the two is the film field. On Route A it does nothing either: the coarse
  size (1 um) is below 5 um, and a Route A sheet outline has no `film_sidewall` feature.
- **No run needed the algorithm fallback.** It is tested with a scripted worker only.

## Where the time goes

HXT took 25 s on the 1 mm chip and 111 s on the 4 mm chip, against 24 s for RP, although it makes fewer
tetrahedra. The study below ran the module's own size field on `B_1mm` and `circB_4mm`, each variant in
a fresh process.

Split of the HXT time (one thread, seconds):

| | 1D | 2D | 3D | of which HXT optimiser | write and audit | start and probe |
|---|---:|---:|---:|---:|---:|---:|
| `B_1mm` | 1.4 | 0.2 | 22.7 | 22.6 | 0.1 | 0.3 |
| `circB_4mm` | 5.1 | 1.0 | 105.1 | - | 0.1 | 0.3 |

The 3D step is 93 to 95 % of the time. Inside it, HXT's boundary recovery and refinement take about 0.1 s on
the 1 mm chip; the rest is its optimiser, which runs on one thread and makes eight global passes, the
first two taking 6 s each.

Variants of the same size field (seconds are the mesh generation):

| variant | `B_1mm` | `circB_4mm` |
|---|---|---|
| HXT, one thread (features/1 before) | 24.9 s, 33,795 tets, 0.0172, 0 below | 114.6 s, 115,766 tets, 0.0151, 0 below |
| Delaunay (3D algorithm 1) | 2.7 s, 52,338 tets, 0.0256, 0 below | 11.7 s, 187,568 tets, 0.0102, 0 below |
| HXT, `MeshSizeExtendFromBoundary=1` | 75.9 s, 847,464 tets, 0.0217, 0 below | not finished after 14 CPU minutes |
| HXT, Distance sampling capped at 200 | 23.2 s, 32,476 tets, 0.0116, 0 below | 59.2 s, 90,810 tets, 0.0117, 0 below |
| HXT, 4 threads | 11.3 s and 10.8 s, 33,758 and 33,761 tets | not run |
| HXT, 16 threads | 8.4 s and 5.8 s, 33,736 and 33,697 tets | not run |

- **Delaunay is the default now.** It is 9 to 10 times faster, clean on all seven cases and
  reproducible, with 1.5 times the tetrahedra of HXT. HXT remains available as
  `MesherOptions(algorithm_3d=10)` and is the retry when Delaunay fails.
- **Extension from the boundary** multiplies the tetrahedra by 25 on the 1 mm chip.
- **The Distance sampling is not the cost.** Field evaluation shows up in the 1D step (1.4 s and 5.1 s)
  and costs nothing in 3D. Capping the sampling at 200 only made the mesh coarser: samples 20 um apart
  on a 4 mm outline overestimate the distance by up to 10 um, so the elements near the walls grow too
  large (min SICN 0.0117).
- **Threads make HXT faster but not reproducible.** Gmsh runs on one thread unless told otherwise; with
  4 or 16 threads two identical runs gave different tetrahedron counts. The module sets
  `General.NumThreads=1`.

## Determinism

Each case is meshed twice by the module in separate directories. In all seven cases the MSH files are
byte-identical and the quality reports (without the timing block) are equal.
