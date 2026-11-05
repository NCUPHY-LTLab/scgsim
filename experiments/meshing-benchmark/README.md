# Meshing benchmark

This benchmark meshes seven fixed SGB cases with `scgsim.meshing` and with two reference recipes,
and records tetrahedron count, minimum quality, elements below the threshold, repair rounds, stop
reason, time and whether the HXT fallback ran. Run it again after every rule change and compare
with `baseline.json`.

## Run it

From the repository root, with the test and Palace dependencies synced
(`uv sync --locked --group test --group occt --extra palace`):

```bash
uv run python experiments/meshing-benchmark/run.py OUTPUT_DIR \
    --compare experiments/meshing-benchmark/baseline.json
```

The full run takes about 13 minutes on one workstation, mostly the 2 mm and 4 mm chips. Useful options:

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

- **module** is `mesh_sgb` with the default `RuleSet` (`scgsim.meshing.rules/1`) and the threshold
  **0.01 on minSICN. The threshold is provisional**: the quality measure and bound are decision C,
  which is still open.
- **module + conductor-outline** is the same with `RuleSet(conductor_outline=True)`, the optional rule
  that sizes the conductors' outline edges at 5 um, growing to the coarse size over 300 um.
- **R8** is HXT with only the global maximum size (bounding-box diagonal / 20): the cheap reference.
- **RP** is the uniform policy Palace applies today, 5 um on every curve growing to 300 um, with 2D
  algorithm 5 and 3D Delaunay. It uses the module's own Distance/Threshold field, so its element
  counts differ slightly from the earlier prototype's.
- **module, repair only** (`B_1mm` only) is the module with the `film_sidewall` features removed from
  the sidecar, so the repair loop must find the film walls from the bad elements alone.

Every method runs in a fresh worker process. R8 and RP are one round with no repair; their bad
elements are counted, not attributed.

## Result

Gmsh 4.15.2, threshold 0.01 (provisional). "below" counts tetrahedra under the threshold.

| case | method | tetrahedra | min SICN | below | rounds | stop | seconds | fallback |
|---|---|---:|---:|---:|---:|---|---:|---|
| A | module | 7,163 | 0.13 | 0 | 0 | clean | 0.5 | no |
| A | module + conductor-outline | 7,163 | 0.13 | 0 | 0 | clean | 0.5 | no |
| A | R8 | 6,887 | 0.3369 | 0 | 0 | - | 0.2 | no |
| A | RP | 825 | 0.1056 | 0 | 0 | - | 0.2 | no |
| B | module | 11,249 | 0.01828 | 0 | 0 | clean | 5.8 | no |
| B | module + conductor-outline | 35,333 | 0.04167 | 0 | 0 | clean | 5.3 | no |
| B | R8 | 7,365 | 0.01982 | 0 | 0 | - | 2.0 | no |
| B | RP | 8,491 | 0.02078 | 0 | 0 | - | 0.7 | no |
| C | module | 14,747 | 0.01828 | 0 | 0 | clean | 8.6 | no |
| C | module + conductor-outline | 39,924 | 0.04167 | 0 | 0 | clean | 11.1 | no |
| C | R8 | 8,737 | 0.01982 | 0 | 0 | - | 4.7 | no |
| C | RP | 10,508 | 0.02078 | 0 | 0 | - | 0.7 | no |
| circB | module | 10,983 | 0.01853 | 0 | 0 | clean | 5.0 | no |
| circB | module + conductor-outline | 35,180 | 0.04111 | 0 | 0 | clean | 4.4 | no |
| circB | R8 | 7,220 | 0.01991 | 0 | 0 | - | 1.3 | no |
| circB | RP | 8,235 | 0.02134 | 0 | 0 | - | 0.7 | no |
| B_1mm | module | 37,260 | 0.01738 | 0 | 0 | clean | 24.1 | no |
| B_1mm | module + conductor-outline | 113,038 | 0.02259 | 0 | 0 | clean | 9.9 | no |
| B_1mm | R8 | 9,664 | 0.002063 | 123 | 0 | - | 2.0 | no |
| B_1mm | RP | 100,227 | 0.01607 | 0 | 0 | - | 4.7 | no |
| B_1mm | module, repair only | 41,154 | 0.01111 | 0 | 7 | clean | 111.4 | no |
| circB_2mm | module | 67,727 | 0.01675 | 0 | 0 | clean | 47.6 | no |
| circB_2mm | module + conductor-outline | 87,015 | 0.01857 | 0 | 0 | clean | 25.4 | no |
| circB_2mm | R8 | 9,344 | 0.001036 | 120 | 0 | - | 2.3 | no |
| circB_2mm | RP | 205,692 | 0.01501 | 0 | 0 | - | 10.5 | no |
| circB_4mm | module | 126,841 | 0.01567 | 0 | 0 | clean | 93.9 | no |
| circB_4mm | module + conductor-outline | 126,836 | 0.01567 | 0 | 0 | clean | 95.2 | no |
| circB_4mm | R8 | 8,579 | 0.0004609 | 122 | 0 | - | 1.2 | no |
| circB_4mm | RP | 419,045 | 0.01199 | 0 | 0 | - | 24.0 | no |

What the table shows:

- **R8 is not enough on the large chips.** HXT with only a global size leaves 120 to 123 near-flat
  tetrahedra at the film walls on the 1, 2 and 4 mm chips, and none on the 100 um chips.
- **The a priori film rule removes them without any repair round.** Every module run is clean in round
  0. On the large chips the module has about 1/3 of the tetrahedra of RP (4 mm: 126,841 against
  419,045).
- **The repair loop works on real meshes.** Without the film facts, the loop alone cleans `B_1mm` in 7
  rounds, to 41,154 tetrahedra, but takes 111 s instead of 24 s.
- **The module costs more than R8 on the small cases** (about 1.5 times the tetrahedra on `B`), because
  film walls get a size even where R8 was clean. `A` also gets a size from the substrate walls, which
  `film_sidewall` covers.
- **The conductor-outline rule costs about three times the tetrahedra on the 100 um and 1 mm chips,**
  with a better minimum quality (0.042 on the 100 um chips) and, on the 1 mm chip, a shorter time
  (9.9 s against 24.1 s). On the 4 mm chip it changes nothing: the film field, which grows by 1 um per um, is
  already finer than the outline field everywhere, so the minimum of the two is the film field. On
  Route A it does nothing either: the coarse size (1 um) is below 5 um, and a Route A sheet outline has
  no `film_sidewall` feature.
- **No run needed the Delaunay fallback.** The fallback is tested with a scripted worker only.

## Determinism

Each case is meshed twice by the module in separate directories. In all seven cases the MSH files are
byte-identical and the quality reports (without the timing block) are equal.
