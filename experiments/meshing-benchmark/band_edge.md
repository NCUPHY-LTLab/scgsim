# Band-edge sizing candidates (issue 136)

The measured table from which the Human picked the default sizing rule for Surface-EPR band edges. It
compares four band-edge rules with the default of the time, `FeatureRecipe()` at
`scgsim.meshing.features/1` (no size on band edges), and `UniformBoundaryRecipe()`, on the cases of PR 132
and on SimplePad on Route A.

**Decision (the Human, 2026-11-11): `band4` is the default**, judged at PDK `49e539f`, and the multiple is
a parameter. `FeatureRecipe` now sizes every band boundary curve at `band_mesh_size_multiple` (default 4)
× the narrowest band width of its face, on the ladder, as the `band4` candidate did; its version is
`scgsim.meshing.features/2`. `band2` and `band8` are `FeatureRecipe(band_mesh_size_multiple=2.0)` and
`(8.0)`. The table below was measured before that change (at `f53cbdf`): its `default` rows are
`features/1`, which no longer exists, and its band rows ran an experiment recipe that asked for the
same sizes. [Check of the new default](#check-of-the-new-default) re-runs it on the current tree.

## Setup

The setup of PR 132, unchanged:

- OrcaPen SC PDK at `8d18025` (the pin of PR 132), layered over the locked environment with
  `uv run --with-editable`. The checkout's source is that commit; its `pyproject.toml` and `uv.lock`
  carry a local edit (an added `gdsfactoryplus` dependency), so the layer also holds what that
  `pyproject.toml` asks for (gdsfactory 9.48.0).
- Vacuum XY 0 and ±Z 1000 µm from the geometry bounds.
- One `mesh_sgb` call per run: threshold 0.01 on minSICN (provisional, decision C), the default repair
  loop and `Budgets()` (3-minute limit up to 4 mm², scaled above it), 3D Delaunay, linear elements,
  one Gmsh thread. Gmsh 4.15.2, SCGSim `develop` at `83f6af1`.
- Each case is built once and every candidate meshes the same XAO and sidecar; every record carries
  their hashes. The PDK layouts are built by the geometry half of `EigenmodeSim.mesh()`; `mesh()`
  itself is not the harness, because after a clean mesh of the PR 132 Disk Qubit its Palace lowering
  refuses the lumped-port sheet `P1` as non-conformal (8 triangles, 2 of them on one tetrahedron only).

Cases:

| case | what it is |
|---|---|
| `disk_nobands`, `disk_bands` | `disk_flip_chip_qubit(include_ground=False, junction_lead_span=0.0, tc_grounded_sides=(), curve_sampling="simulation")` under the common flip-chip ground planes, Route B, every authored port at 10 nH, MA/MS/SA films without and with bands (0.5, 1.0) µm, as in PR 132 |
| `B_1mm_*`, `circB_2mm_*`, `circB_4mm_*` | the 1, 2 and 4 mm pad-in-ground chips of PR 132 (0.2 µm metal), `_bands` with bands (0.5, 1.0) µm on every conductor surface and the SA interface, `_nobands` without |
| `simplepad_A_bands` | SimplePad of issue 84 (`square_pad_capacitor`, 200 µm pad, 20 µm gap, 1 mm die) on Route A (pad and ground as sheets), bands (0.5, 1.0) µm, port 10 nH |
| `simplepad_A_bands_nb` | the same with the (250, 250, 1000) µm plan vacuum of the PDK notebooks: the Palace run of issue 84 that found the Route A failure |
| `disk_site_*` | supplement, outside the PR 132 setup: the Disk Qubit at the repository's current PDK pin `49e539f`, whose ground-plane helper needs explicit bounds (the site, ±500 µm) |

## Candidates

`b` is the `narrowest_band_um` fact of a band boundary feature: 0.5 µm here, since margins (0.5, 1.0)
make both bands 0.5 µm wide. "On the ladder" rounds a size down to the coarse size over a power of two,
as every size of `FeatureRecipe` is; the sizes column of the table gives the result on each case.

| name | rule |
|---|---|
| `default` | `FeatureRecipe()` (`scgsim.meshing.features/1`): a film wall gets 25 × its thickness on the ladder when that is below half the coarse size (bounding-box diagonal / 20); band edges get no size. |
| `uniform` | `UniformBoundaryRecipe()` (`scgsim.meshing.uniform-boundary/1`): 5 µm on every curve, growing to 300 µm over 300 µm. |
| `band2` | the default's film walls, plus every band boundary curve at 2 × `b` on the ladder (target 1 µm: 0.64 µm on the Disk Qubit, 0.56–0.64 µm on SimplePad, 0.60–0.99 µm on the chips). |
| `band4` | the same at 4 × `b` (target 2 µm: 1.28–1.30 µm on the Disk Qubit, 1.12–1.28 µm on SimplePad, 1.2–2.0 µm on the chips). |
| `band8` | the same at 8 × `b` (target 4 µm: 2.55–2.60 µm on the Disk Qubit, 2.24–2.56 µm on SimplePad, 2.4–4.0 µm on the chips). |
| `wall100_band8` | film walls at 100 × their thickness (5.1 µm on the Disk Qubit) and band curves at 8 × `b`: the light candidate. |

The faces between a band and a film wall get no size of their own. They are band strips 0.5 µm wide,
so every point of one lies within 0.25 µm of a boundary curve, and the curve sizes reach it through the
gradation of 1 µm per µm; a face size would add nothing there. A curve covered by a band and a film
wall keeps the finer of the two sizes. A rule tied to the band on each side of one curve (rather than the
narrowest band of the face) is not expressible: the sidecar states the margins and the narrowest band
per banded face, not per curve. With equal bands, as here, it would ask for the same sizes.

At the time of the table the band candidates were an experiment dataclass with `name`, `version` and
`size_requests`, the public recipe interface; it imported `SizeRequests`, `BoundingBox`,
`SidecarFeatures` and `ladder_size` from `scgsim.meshing.sizing` and `scgsim.meshing.features`, since
`scgsim.meshing` does not re-export them.

## How to read the table

- **s** is `timing.total_seconds` of the quality report: the whole `mesh_sgb` call.
- **tets** is round 0 → the last round that made a mesh; **bad per round** counts tetrahedra below 0.01
  in each round, `killed` a round stopped at the time limit.
- **where** names, for a run that did not stop clean, the remaining bad elements by location and the
  worst three with their quality, centroid (µm), feature kinds and two of their groups. Every record in
  `band_edge.json` lists all of them.
- **load** is the 1-minute load average of `/proc/loadavg` before and after the run (64 cores);
  **Palace** says whether a Palace process of another agent ran at either moment.
- **pass**: `first` and `pin49` ran every row, four runs at a time, beside other agents' work and, for
  16 of the 72 rows, a 16-rank Palace run. `quiet` and `pin49-quiet` re-ran the rows that decide the
  picture one at a time, each started only when no Palace process ran (no process named `palace` or
  `palace-*`) and the 1-minute load was below 8. The summary marks with `*` the numbers that were not
  re-timed: rows whose mesh is the default's, rows far from every target (`band2`, whose first mesh on
  the banded Disk Qubit alone takes 164–178 s), and `uniform` and `wall100_band8` on the small models.

## Summary

Seconds of the whole `mesh_sgb` call, stop reason and repair rounds. A plain number is the `quiet` or
`pin49-quiet` re-time (one run at a time, no Palace process, 1-minute load 1.1–10.9); a number with `*` is
from the `first` or `pin49` pass only (up to four runs at a time, load 1.8–26.7, a 16-rank Palace run
beside 16 of the 72 rows). "= default" means the candidate asks for exactly the default's sizes on that
case, so the mesh is the same.

| case | target | `default` | `uniform` | `band2` | `band4` | `band8` | `wall100_band8` |
|---|---|---|---|---|---|---|---|
| `disk_nobands` | not slower than PR 132 (62 s) | 55.4 clean, 2 | 41.6 clean, 2 | 59.8* = default | 54.0 clean, 2 = default | 58.2* = default | 12.9 clean, 2 |
| `disk_bands` | 1 min; 3 min limit | 134.2 clean, 2 | 180 time limit | 180* time limit | **112.6 clean, 1** | 167.1 clean, 2 (mesh = default) | 156.1 stalled |
| chips without bands | not slower than PR 132 (about 1 s) | 0.5–0.6 clean, 0 | 4.3–17.8* clean | = default | 0.5–0.6 = default | 0.5–0.6 = default | 0.4–0.5* clean, 0 |
| chips with bands | 3 min (12 min for 4 mm) | 0.7–0.8 clean, 0 | 5.0–18.5* clean | 1.6–2.0* clean, 0 | 1.0–1.2 clean, 0 | 0.7–0.8 clean, 0 | 0.8–0.9* clean, 0 |
| `simplepad_A_bands` | clean at minSICN ≥ 0.01 | 3.7 stalled, 0.0031 | 6.0 clean, 0.0186 | 13.7* clean, 0.166 | 5.5 clean, 0.0235 | 2.7 clean, 0.0191 | 2.9* = `band8` |
| `simplepad_A_bands_nb` | clean at minSICN ≥ 0.01 | 4.2 stalled, 0.0043 | 26.8 clean, 0.0114 | 14.4* clean, 0.102 | 4.7 clean, 0.0388 | 2.6 clean, 0.0144 | 2.9* = `band8` |
| `disk_site_nobands` (PDK 49e539f) | not slower | 16.7 clean, 0 | 28.7* clean, 1 | 18.8* = default | 16.8 = default | 18.1* = default | 7.4 clean, 1 |
| `disk_site_bands` (PDK 49e539f) | 1 min; 3 min limit | 82.5 clean, 1 | 101.6* stalled | 180.1* time limit | **53.4 clean, 0** | 104.5 clean, 1 (mesh = default) | 180 time limit |

## What the table shows

- **3-minute limit.** `default`, `band4` and `band8` stay within it on every case at both PDK pins; the
  slowest is `band8` on `disk_bands`, 167 s. `uniform` and `band2` reach the limit on the banded Disk
  Qubit (`band2` asks for 1.37 million tetrahedra, and its first mesh alone takes 164–178 s), and
  `wall100_band8` stalls or reaches it there. The margin on the banded Disk Qubit at the PR 132 pin
  depends on the load: under load 17–19 `band8` took 177.7 s and `band4` 168.9 s.
- **1-minute target on the banded Disk Qubit.** At the PR 132 pin no candidate meets it. The fastest is
  `band4`, 112.6 s: its first mesh (573,727 tetrahedra, 53 s) has 61 bad elements, most of them
  needles with a 1 nm edge at the spikes of the TC pads (orcapen-sc-pdk#47; the worst sit near
  `(±169, ±59)` and `(±212, ±88)` µm, as in PR 132), and one more mesh (59 s) removes them. Every
  recipe needs that second mesh at this pin, and `default` a third (62, 2, 0 bad). At PDK `49e539f`,
  which rebuilt the Disk Qubit outlines (orcapen-sc-pdk#64), the spikes are gone: `band4` is clean in
  round 0 at 53.4 s (569,325 tetrahedra, minSICN 0.051), the only run under a minute. It has 7 s of
  margin when the machine is quiet and none under load: the same run took 58.9 s at load 17–19 beside a
  16-rank Palace run. `default` has one bad element in its first mesh there and needs one repair round
  (82.5 s).
- **SimplePad on Route A with bands.** `default` stalls after 5 rounds with every remaining element on
  `band_boundary` curves, in both vacua (minSICN 0.0031 and 0.0043, as in issue 84). Every band rule
  meshes it clean in round 0: `band8` with the fewest elements (49,360–51,837, minSICN 0.014–0.019),
  `band4` with about twice as many (88,628–115,346, minSICN 0.024–0.039), `band2` with 4 times as many.
  `uniform` is clean too (minSICN 0.011–0.019).
- **Models without bands.** A band rule asks for nothing on a model without band features, so `band2`,
  `band4` and `band8` give the default's mesh there: `disk_nobands` 54.0–55.4 s against 62 s in PR 132,
  the chips 0.5–0.6 s. Only `wall100_band8` changes those meshes, because it changes the wall rule.
- **The band rule's cost on Route B.** With bands, `band4` sizes the 2,503 band curves of the Disk Qubit
  (66.6 mm, against 27.2 mm of film walls) at the wall size, 1.28 µm: 43 % more tetrahedra in the first
  mesh than `default`, and on the chips 1.3–1.8 times the default's tetrahedra with a higher minSICN
  (0.085–0.137 against 0.023–0.074). On the Disk Qubit it needs one repair round fewer than `default`
  at both pins (61, 0 against 62, 2, 0 bad; 0 against 1, 0), which pays for the larger first mesh.
- **A band size that changes nothing still costs time.** `band8` asks for 2.55 µm on band curves that lie
  within 1 µm of 1.28 µm walls, so its mesh on `disk_bands` is the default's to the element, yet it took
  167.1 s against 134.2 s (and 104.5 against 82.5 s at `49e539f`). The extra time is the Gmsh Distance
  field over the 2,503 band curves, evaluated at every point. A recipe cannot change how a Distance field
  samples its curves (`scgsim.meshing.native`), only which curves it lists.
- **Coarser film walls do not work with bands.** `wall100_band8` (5.1 µm walls, 2.55 µm band curves)
  leaves slivers where a film wall meets the first band (`MA::…__sidewall` with `MS::…__top::band=0`)
  that local repair does not clear: stalled at 156 s, or the time limit at `49e539f`. Without bands it is
  the fastest (7.4–12.9 s).

Not in the table (round 0 only, under load 3–4, on `disk_bands` at `8d18025`): `band4` with walls at 50 or
100 × thickness gave 552,914 tetrahedra (4 % fewer than `band4`, the band curves set the size near the
walls) with the same 61 bad elements; band curves at 1.28 µm growing at 2 µm per µm (a separate field,
a scratch variant not kept in `band_edge.py`) gave a tetrahedron of minSICN 0.0 and 83 bad elements; on
`simplepad_A_bands`, band curves at 16 × `b` (4.49 µm) left 5 bad elements, so `band8` is the coarsest
clean band size there.

## Check of the new default {#check-of-the-new-default}

`FeatureRecipe()` at `scgsim.meshing.features/2` (`band_mesh_size_multiple=4.0`), PDK `49e539f`, one run at
a time on 2026-11-11. The first two ran with no Palace process; the other five beside another agent's two
16-rank Palace jobs (load 36–40), so their seconds are high.

| case | s | tets (round 0 → last) | min SICN | rounds | stop | load 1 min | Palace |
|---|---:|---|---:|---:|---|---|---|
| Disk Qubit with bands (`disk_site_bands`) | 54.9 | 569,325 | 0.05136 | 0 | clean | 7.3–4.6 | no |
| SimplePad on Route A with bands (`simplepad_A_bands`) | 6.3 | 115,346 | 0.02354 | 0 | clean | 12.3–11.7 | no |
| the same, notebook vacuum (`simplepad_A_bands_nb`) | 4.9 | 88,628 | 0.03882 | 0 | clean | 40.3–39.9 | yes |
| SimplePad on Route B with bands (`simplepad_B_bands`) | 9.4 | 117,168 | 0.03884 | 0 | clean | 39.5–38.9 | yes |
| the same, notebook vacuum (`simplepad_B_bands_nb`) | 31.6 | 213,040 → 213,634 | 0.0341 | 1 | clean | 38.8–37.2 | yes |
| 1 mm chip without bands (`B_1mm_nobands`) | 0.4 | 7,604 | 0.05528 | 0 | clean | 36.9–36.9 | yes |
| Disk Qubit without bands (`disk_site_nobands`) | 17.5 | 303,440 | 0.01565 | 0 | clean | 36.6–35.8 | yes |

- The banded Disk Qubit, both SimplePad Route A cases and the 1 mm chip have the meshes of the `band4`
  rows of the table to the element (569,325, 115,346, 88,628 tetrahedra) or of `features/1` without bands
  (7,604 and 303,440), so the new default asks for the sizes the `band4` candidate asked for, and nothing
  changes without bands. The banded Disk Qubit is under a minute again (54.9 s against 53.4 s quiet).
- SimplePad on Route B with bands is clean in both vacua. With the notebook vacuum it needs one repair
  round (1 bad element in its first mesh); the Palace run of issue 84 meshed that case with `features/1`
  clean in round 0 (188,254 tetrahedra, 18 s), on an earlier SGB revision and under its own load, so the
  two are not directly comparable.

## All runs

Every run of every pass, in the order of the cases above.

The tables below are the output of `band_edge.py table` over the run folders of `first`/`quiet` and
`pin49`/`pin49-quiet`/`features2`; `band_edge.json` holds the same records.

### Pass `first` (PDK 8d18025, every row, four runs at a time)

| case | candidate | sizes µm (coarse; curves) | s | tets (round 0 → last) | min SICN | bad per round | rounds | stop | where (stopped runs) | load 1 min (before–after) | Palace | pass |
|---|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| disk_nobands | default | 163.4; 1.277 ×913 | 59.4 | 301,805 → 306,258 | 0.01032 | 60, 3, 0 | 2 | clean |  | 1.8–12.1 | no | first |
| disk_nobands | uniform | 300; 5 ×1274 | 46.0 | 180,409 → 192,996 | 0.01048 | 61, 5, 0 | 2 | clean |  | 16.5–12.8 | no | first |
| disk_nobands | band2 | 163.4; 1.277 ×913 | 59.8 | 301,805 → 306,258 | 0.01032 | 60, 3, 0 | 2 | clean |  | 12.2–7.9 | no | first |
| disk_nobands | band4 | 163.4; 1.277 ×913 | 61.2 | 301,805 → 306,258 | 0.01032 | 60, 3, 0 | 2 | clean |  | 7.9–9.1 | no | first |
| disk_nobands | band8 | 163.4; 1.277 ×913 | 58.2 | 301,805 → 306,258 | 0.01032 | 60, 3, 0 | 2 | clean |  | 9.1–5.7 | no | first |
| disk_nobands | wall100_band8 | 163.4; 5.108 ×913 | 13.3 | 74,648 → 80,990 | 0.01033 | 61, 8, 0 | 2 | clean |  | 5.5–5.4 | no | first |
| disk_bands | default | 163.4; 1.277 ×913 | 138.3 | 400,325 → 415,061 | 0.01017 | 62, 2, 0 | 2 | clean |  | 1.9–12.2 | no | first |
| disk_bands | uniform | 300; 5 ×3835 | 180.0 | 244,235 → 262,732 | 0.002794 | 94, 7, 6, 11, killed | 4 | time_limit | 11 bad (7 curve, 2 point, 2 surface): 0.0028 at (232.6, 72.9, 0.0) - [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=2] · 0.0039 at (-260.5, 257.6, -3.4) band_boundary [MS::RO__bottom::band=0; MS::RO__bottom::band=1] · 0.0044 at (-402.9, 415.1, -3.7) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=1] | 11.6–5.7 | no | first |
| disk_bands | band2 | 163.4; 0.6385 ×2503, 1.277 ×913 | 180.0 | 1,365,406 | 0.001002 | 64, killed | 1 | time_limit | 64 bad (6 curve, 56 point, 2 surface): 0.001 at (-59.1, -169.1, 7.6) film_sidewall [MA::TC_S__bottom::band=0; MA::TC_S__sidewall] · 0.0012 at (59.1, 168.8, 7.5) film_sidewall [MA::TC_N__bottom::band=0; MA::TC_N__sidewall] · 0.0012 at (-168.8, 59.1, 7.5) film_sidewall [MA::TC_W__bottom::band=0; MA::TC_W__sidewall] | 5.5–16.9 | yes | first |
| disk_bands | band4 | 163.4; 1.277 ×3416 | 168.9 | 573,727 → 581,101 | 0.0101 | 61, 0 | 1 | clean |  | 16.9–19.1 | yes | first |
| disk_bands | band8 | 163.4; 1.277 ×913, 2.554 ×2503 | 177.7 | 400,325 → 415,061 | 0.01017 | 62, 2, 0 | 2 | clean |  | 19.0–18.8 | yes | first |
| disk_bands | wall100_band8 | 163.4; 2.554 ×2503, 5.108 ×913 | 180.0 | 276,500 → 302,640 | 0.006041 | 55, 2, 3, killed | 3 | time_limit | 3 bad (3 curve): 0.006 at (1.7, 369.2, 7.0) film_sidewall [MA::TC_N__bottom::band=0; MA::TC_N__sidewall] · 0.0069 at (-58.6, -344.2, 10.2) band_boundary/film_sidewall [MA::D1_BOTTOM_GROUND_PLANE_0000__sidewall; MS::D1_BOTTOM_GROUND_PLANE_0000__top::band=0] · 0.0091 at (405.7, 2.1, 10.2) band_boundary/film_sidewall [MA::TC_E__sidewall; MS::TC_E__top::band=0] | 18.8–22.1 | yes | first |
| B_1mm_nobands | default | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 2.3–2.3 | no | first |
| B_1mm_nobands | uniform | 300; 5 ×72 | 4.3 | 67,751 | 0.05418 | 0 | 0 | clean |  | 2.5–2.7 | no | first |
| B_1mm_nobands | band2 | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 2.7–2.7 | no | first |
| B_1mm_nobands | band4 | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 8.1–8.1 | no | first |
| B_1mm_nobands | band8 | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 7.7–7.7 | no | first |
| B_1mm_nobands | wall100_band8 | 126.6; 15.82 ×36 | 0.5 | 6,052 | 0.02882 | 0 | 0 | clean |  | 7.7–7.7 | no | first |
| B_1mm_bands | default | 126.6; 3.956 ×36 | 0.7 | 9,570 | 0.02674 | 0 | 0 | clean |  | 7.3–7.3 | no | first |
| B_1mm_bands | uniform | 300; 5 ×172 | 5.0 | 69,309 | 0.05406 | 0 | 0 | clean |  | 12.1–16.5 | no | first |
| B_1mm_bands | band2 | 126.6; 0.989 ×92, 3.956 ×36 | 1.6 | 22,868 | 0.08013 | 0 | 0 | clean |  | 15.5–15.5 | no | first |
| B_1mm_bands | band4 | 126.6; 1.978 ×92, 3.956 ×36 | 1.1 | 12,941 | 0.08475 | 0 | 0 | clean |  | 14.6–14.6 | no | first |
| B_1mm_bands | band8 | 126.6; 3.956 ×128 | 0.7 | 9,234 | 0.04455 | 0 | 0 | clean |  | 14.6–18.2 | no | first |
| B_1mm_bands | wall100_band8 | 126.6; 3.956 ×92, 15.82 ×36 | 0.8 | 9,485 | 0.02559 | 0 | 0 | clean |  | 18.2–18.2 | no | first |
| circB_2mm_nobands | default | 179.2; 2.8 ×36 | 0.6 | 11,287 | 0.07817 | 0 | 0 | clean |  | 17.1–17.1 | no | first |
| circB_2mm_nobands | uniform | 300; 5 ×72 | 8.4 | 120,904 | 0.05326 | 0 | 0 | clean |  | 16.1–15.1 | no | first |
| circB_2mm_nobands | band2 | 179.2; 2.8 ×36 | 0.6 | 11,287 | 0.07817 | 0 | 0 | clean |  | 14.2–14.2 | no | first |
| circB_2mm_nobands | band4 | 179.2; 2.8 ×36 | 0.5 | 11,287 | 0.07817 | 0 | 0 | clean |  | 13.5–13.5 | no | first |
| circB_2mm_nobands | band8 | 179.2; 2.8 ×36 | 0.6 | 11,287 | 0.07817 | 0 | 0 | clean |  | 12.8–12.8 | no | first |
| circB_2mm_nobands | wall100_band8 | 179.2; 11.2 ×36 | 0.5 | 9,160 | 0.03843 | 0 | 0 | clean |  | 12.8–12.8 | no | first |
| circB_2mm_bands | default | 179.2; 2.8 ×36 | 0.8 | 13,474 | 0.07369 | 0 | 0 | clean |  | 12.2–12.2 | no | first |
| circB_2mm_bands | uniform | 300; 5 ×154 | 8.9 | 122,228 | 0.03251 | 0 | 0 | clean |  | 11.6–10.7 | no | first |
| circB_2mm_bands | band2 | 179.2; 0.6999 ×74, 2.8 ×36 | 1.9 | 31,731 | 0.1649 | 0 | 0 | clean |  | 10.7–10.2 | no | first |
| circB_2mm_bands | band4 | 179.2; 1.4 ×74, 2.8 ×36 | 1.1 | 18,161 | 0.1003 | 0 | 0 | clean |  | 10.2–9.8 | no | first |
| circB_2mm_bands | band8 | 179.2; 2.8 ×110 | 0.8 | 13,375 | 0.03775 | 0 | 0 | clean |  | 9.8–9.8 | no | first |
| circB_2mm_bands | wall100_band8 | 179.2; 2.8 ×74, 11.2 ×36 | 0.9 | 13,564 | 0.0664 | 0 | 0 | clean |  | 9.4–9.4 | no | first |
| circB_4mm_nobands | default | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 9.1–9.1 | no | first |
| circB_4mm_nobands | uniform | 300; 5 ×72 | 17.8 | 229,819 | 0.04033 | 0 | 0 | clean |  | 8.7–8.1 | no | first |
| circB_4mm_nobands | band2 | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 7.9–7.9 | no | first |
| circB_4mm_nobands | band4 | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 7.8–7.8 | no | first |
| circB_4mm_nobands | band8 | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 7.8–7.8 | no | first |
| circB_4mm_nobands | wall100_band8 | 307.2; 19.2 ×36 | 0.4 | 8,328 | 0.02516 | 0 | 0 | clean |  | 7.6–7.6 | no | first |
| circB_4mm_bands | default | 307.2; 4.801 ×36 | 0.7 | 11,262 | 0.02282 | 0 | 0 | clean |  | 7.4–7.4 | no | first |
| circB_4mm_bands | uniform | 300; 5 ×154 | 18.5 | 231,018 | 0.01375 | 0 | 0 | clean |  | 7.2–6.9 | no | first |
| circB_4mm_bands | band2 | 307.2; 0.6001 ×74, 4.801 ×36 | 2.0 | 36,903 | 0.2288 | 0 | 0 | clean |  | 6.7–6.7 | no | first |
| circB_4mm_bands | band4 | 307.2; 1.2 ×74, 4.801 ×36 | 1.2 | 19,876 | 0.1373 | 0 | 0 | clean |  | 6.6–6.6 | no | first |
| circB_4mm_bands | band8 | 307.2; 2.4 ×74, 4.801 ×36 | 0.9 | 13,701 | 0.04591 | 0 | 0 | clean |  | 10.3–10.3 | no | first |
| circB_4mm_bands | wall100_band8 | 307.2; 2.4 ×74, 19.2 ×36 | 0.9 | 13,701 | 0.04591 | 0 | 0 | clean |  | 9.9–9.9 | no | first |
| simplepad_A_bands | default | 143.6; none | 3.7 | 5,942 → 14,514 | 0.00313 | 407, 10, 5, 3, 7, 4 | 5 | stalled | 4 bad (4 curve): 0.0031 at (-65.9, 123.6, -6.1) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=1] · 0.0064 at (81.5, -92.4, 10.6) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; chip/PAD::band=0] · 0.0071 at (-65.3, 100.8, -6.7) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; chip/PAD::band=0] | 5.3–5.2 | no | first |
| simplepad_A_bands | uniform | 300; 5 ×110 | 6.4 | 89,175 | 0.01855 | 0 | 0 | clean |  | 5.2–5.2 | no | first |
| simplepad_A_bands | band2 | 143.6; 0.561 ×60 | 13.7 | 235,332 | 0.1662 | 0 | 0 | clean |  | 5.1–4.9 | no | first |
| simplepad_A_bands | band4 | 143.6; 1.122 ×60 | 6.0 | 115,346 | 0.02354 | 0 | 0 | clean |  | 4.9–4.6 | no | first |
| simplepad_A_bands | band8 | 143.6; 2.244 ×60 | 3.0 | 51,837 | 0.01908 | 0 | 0 | clean |  | 4.6–4.5 | no | first |
| simplepad_A_bands | wall100_band8 | 143.6; 2.244 ×60 | 2.9 | 51,837 | 0.01908 | 0 | 0 | clean |  | 4.4–4.4 | no | first |
| simplepad_A_bands_nb | default | 163.9; none | 4.4 | 10,226 → 17,264 | 0.004289 | 380, 4, 7, 2, 2, 3 | 5 | stalled | 3 bad (3 curve): 0.0043 at (22.2, 129.1, 16.1) band_boundary [chip/GROUND::band=0; chip/GROUND::band=1] · 0.0044 at (22.1, 128.9, 16.1) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; chip/GROUND::band=0] · 0.0077 at (-119.8, 42.5, -12.5) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=1] | 4.3–4.2 | no | first |
| simplepad_A_bands_nb | uniform | 300; 5 ×106 | 28.9 | 136,394 → 137,884 | 0.01139 | 2, 1, 0 | 2 | clean |  | 4.2–4.1 | no | first |
| simplepad_A_bands_nb | band2 | 163.9; 0.6404 ×60 | 14.4 | 216,045 | 0.102 | 0 | 0 | clean |  | 4.2–4.2 | no | first |
| simplepad_A_bands_nb | band4 | 163.9; 1.281 ×60 | 5.5 | 88,628 | 0.03882 | 0 | 0 | clean |  | 4.2–4.2 | no | first |
| simplepad_A_bands_nb | band8 | 163.9; 2.561 ×60 | 4.3 | 49,360 | 0.01443 | 0 | 0 | clean |  | 14.4–14.8 | yes | first |
| simplepad_A_bands_nb | wall100_band8 | 163.9; 2.561 ×60 | 2.9 | 49,360 | 0.01443 | 0 | 0 | clean |  | 14.8–15.1 | yes | first |

### Pass `quiet` (PDK 8d18025, deciding rows, one at a time, no Palace)

| case | candidate | sizes µm (coarse; curves) | s | tets (round 0 → last) | min SICN | bad per round | rounds | stop | where (stopped runs) | load 1 min (before–after) | Palace | pass |
|---|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| disk_nobands | default | 163.4; 1.277 ×913 | 55.4 | 301,805 → 306,258 | 0.01032 | 60, 3, 0 | 2 | clean |  | 6.5–3.8 | no | quiet |
| disk_nobands | uniform | 300; 5 ×1274 | 41.6 | 180,409 → 192,996 | 0.01048 | 61, 5, 0 | 2 | clean |  | 1.9–1.5 | no | quiet |
| disk_nobands | band4 | 163.4; 1.277 ×913 | 54.0 | 301,805 → 306,258 | 0.01032 | 60, 3, 0 | 2 | clean |  | 1.9–1.8 | no | quiet |
| disk_nobands | wall100_band8 | 163.4; 5.108 ×913 | 12.9 | 74,648 → 80,990 | 0.01033 | 61, 8, 0 | 2 | clean |  | 8.2–7.3 | no | quiet |
| disk_bands | default | 163.4; 1.277 ×913 | 134.2 | 400,325 → 415,061 | 0.01017 | 62, 2, 0 | 2 | clean |  | 7.9–6.8 | no | quiet |
| disk_bands | uniform | 300; 5 ×3835 | 180.0 | 244,235 → 262,732 | 0.002794 | 94, 7, 6, 11, killed | 4 | time_limit | 11 bad (7 curve, 2 point, 2 surface): 0.0028 at (232.6, 72.9, 0.0) - [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=2] · 0.0039 at (-260.5, 257.6, -3.4) band_boundary [MS::RO__bottom::band=0; MS::RO__bottom::band=1] · 0.0044 at (-402.9, 415.1, -3.7) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=1] | 2.0–1.9 | no | quiet |
| disk_bands | band4 | 163.4; 1.277 ×3416 | 112.6 | 573,727 → 581,101 | 0.0101 | 61, 0 | 1 | clean |  | 5.8–2.6 | no | quiet |
| disk_bands | band8 | 163.4; 1.277 ×913, 2.554 ×2503 | 167.1 | 400,325 → 415,061 | 0.01017 | 62, 2, 0 | 2 | clean |  | 1.7–2.0 | no | quiet |
| disk_bands | wall100_band8 | 163.4; 2.554 ×2503, 5.108 ×913 | 156.1 | 276,500 → 303,200 | 0.006061 | 55, 2, 3, 2 | 3 | stalled | 2 bad (2 curve): 0.0061 at (189.2, -78.5, 10.4) band_boundary/film_sidewall [MA::TC_E__sidewall; MS::TC_E__top::band=0] · 0.0096 at (160.4, -162.2, 10.1) band_boundary/film_sidewall [MA::Disk__sidewall; MS::Disk__top::band=0] | 6.5–1.8 | no | quiet |
| B_1mm_nobands | default | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 1.6–1.6 | no | quiet |
| B_1mm_nobands | band4 | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 1.6–1.6 | no | quiet |
| B_1mm_nobands | band8 | 126.6; 3.956 ×36 | 0.5 | 7,604 | 0.05528 | 0 | 0 | clean |  | 1.5–1.5 | no | quiet |
| B_1mm_bands | default | 126.6; 3.956 ×36 | 0.7 | 9,570 | 0.02674 | 0 | 0 | clean |  | 3.9–3.9 | no | quiet |
| B_1mm_bands | band4 | 126.6; 1.978 ×92, 3.956 ×36 | 1.0 | 12,941 | 0.08475 | 0 | 0 | clean |  | 3.8–3.8 | no | quiet |
| B_1mm_bands | band8 | 126.6; 3.956 ×128 | 0.7 | 9,234 | 0.04455 | 0 | 0 | clean |  | 3.5–3.5 | no | quiet |
| circB_2mm_nobands | default | 179.2; 2.8 ×36 | 0.6 | 11,287 | 0.07817 | 0 | 0 | clean |  | 3.5–10.9 | no | quiet |
| circB_2mm_nobands | band4 | 179.2; 2.8 ×36 | 0.6 | 11,287 | 0.07817 | 0 | 0 | clean |  | 7.3–7.3 | no | quiet |
| circB_2mm_nobands | band8 | 179.2; 2.8 ×36 | 0.6 | 11,287 | 0.07817 | 0 | 0 | clean |  | 9.5–9.5 | no | quiet |
| circB_2mm_bands | default | 179.2; 2.8 ×36 | 0.8 | 13,474 | 0.07369 | 0 | 0 | clean |  | 6.5–6.5 | no | quiet |
| circB_2mm_bands | band4 | 179.2; 1.4 ×74, 2.8 ×36 | 1.1 | 18,161 | 0.1003 | 0 | 0 | clean |  | 5.7–5.7 | no | quiet |
| circB_2mm_bands | band8 | 179.2; 2.8 ×110 | 0.8 | 13,375 | 0.03775 | 0 | 0 | clean |  | 5.3–5.3 | no | quiet |
| circB_4mm_nobands | default | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 5.0–5.0 | no | quiet |
| circB_4mm_nobands | band4 | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 4.4–4.4 | no | quiet |
| circB_4mm_nobands | band8 | 307.2; 4.801 ×36 | 0.5 | 9,489 | 0.05717 | 0 | 0 | clean |  | 4.2–4.2 | no | quiet |
| circB_4mm_bands | default | 307.2; 4.801 ×36 | 0.7 | 11,262 | 0.02282 | 0 | 0 | clean |  | 3.9–3.9 | no | quiet |
| circB_4mm_bands | band4 | 307.2; 1.2 ×74, 4.801 ×36 | 1.2 | 19,876 | 0.1373 | 0 | 0 | clean |  | 3.7–3.5 | no | quiet |
| circB_4mm_bands | band8 | 307.2; 2.4 ×74, 4.801 ×36 | 0.8 | 13,701 | 0.04591 | 0 | 0 | clean |  | 3.3–3.3 | no | quiet |
| simplepad_A_bands | default | 143.6; none | 3.7 | 5,942 → 14,514 | 0.00313 | 407, 10, 5, 3, 7, 4 | 5 | stalled | 4 bad (4 curve): 0.0031 at (-65.9, 123.6, -6.1) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=1] · 0.0064 at (81.5, -92.4, 10.6) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; chip/PAD::band=0] · 0.0071 at (-65.3, 100.8, -6.7) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; chip/PAD::band=0] | 3.2–3.0 | no | quiet |
| simplepad_A_bands | uniform | 300; 5 ×110 | 6.0 | 89,175 | 0.01855 | 0 | 0 | clean |  | 2.8–2.8 | no | quiet |
| simplepad_A_bands | band4 | 143.6; 1.122 ×60 | 5.5 | 115,346 | 0.02354 | 0 | 0 | clean |  | 2.6–2.5 | no | quiet |
| simplepad_A_bands | band8 | 143.6; 2.244 ×60 | 2.7 | 51,837 | 0.01908 | 0 | 0 | clean |  | 2.4–2.3 | no | quiet |
| simplepad_A_bands_nb | default | 163.9; none | 4.2 | 10,226 → 17,264 | 0.004289 | 380, 4, 7, 2, 2, 3 | 5 | stalled | 3 bad (3 curve): 0.0043 at (22.2, 129.1, 16.1) band_boundary [chip/GROUND::band=0; chip/GROUND::band=1] · 0.0044 at (22.1, 128.9, 16.1) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; chip/GROUND::band=0] · 0.0077 at (-119.8, 42.5, -12.5) band_boundary [SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=0; SA::D0_SUBSTRATE\|VACUUM_REGION__interface::band=1] | 2.2–2.4 | no | quiet |
| simplepad_A_bands_nb | uniform | 300; 5 ×106 | 26.8 | 136,394 → 137,884 | 0.01139 | 2, 1, 0 | 2 | clean |  | 2.4–1.8 | no | quiet |
| simplepad_A_bands_nb | band4 | 163.9; 1.281 ×60 | 4.7 | 88,628 | 0.03882 | 0 | 0 | clean |  | 1.8–1.7 | no | quiet |
| simplepad_A_bands_nb | band8 | 163.9; 2.561 ×60 | 2.6 | 49,360 | 0.01443 | 0 | 0 | clean |  | 1.6–1.6 | no | quiet |

### Pass `pin49` (PDK 49e539f supplement, every row, beside `first`)

| case | candidate | sizes µm (coarse; curves) | s | tets (round 0 → last) | min SICN | bad per round | rounds | stop | where (stopped runs) | load 1 min (before–after) | Palace | pass |
|---|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| disk_site_nobands | default | 166.2; 1.298 ×801 | 18.3 | 303,440 | 0.01565 | 0 | 0 | clean |  | 18.9–19.2 | yes | pin49 |
| disk_site_nobands | uniform | 300; 5 ×1146 | 28.7 | 189,342 → 189,115 | 0.01159 | 2, 0 | 1 | clean |  | 19.4–19.5 | yes | pin49 |
| disk_site_nobands | band2 | 166.2; 1.298 ×801 | 18.8 | 303,440 | 0.01565 | 0 | 0 | clean |  | 19.5–24.1 | yes | pin49 |
| disk_site_nobands | band4 | 166.2; 1.298 ×801 | 17.8 | 303,440 | 0.01565 | 0 | 0 | clean |  | 23.7–22.9 | yes | pin49 |
| disk_site_nobands | band8 | 166.2; 1.298 ×801 | 18.1 | 303,440 | 0.01565 | 0 | 0 | clean |  | 22.9–26.7 | yes | pin49 |
| disk_site_nobands | wall100_band8 | 166.2; 5.194 ×801 | 7.9 | 74,998 → 75,090 | 0.01589 | 2, 0 | 1 | clean |  | 26.7–26.0 | yes | pin49 |
| disk_site_bands | default | 166.2; 1.298 ×801 | 91.0 | 402,784 → 407,093 | 0.02024 | 1, 0 | 1 | clean |  | 13.5–7.2 | no | pin49 |
| disk_site_bands | uniform | 300; 5 ×3459 | 101.6 | 249,282 → 250,097 | 0.006913 | 1, 4, 2 | 2 | stalled | 2 bad (1 curve, 1 point): 0.0069 at (-273.8, 270.4, -3.0) band_boundary/film_sidewall [MA::RO__sidewall; MS::RO__bottom::band=0] · 0.0093 at (164.8, -165.5, 9.0) band_boundary [MS::Disk__top::band=0; MS::Disk__top::band=1] | 7.0–5.7 | no | pin49 |
| disk_site_bands | band2 | 166.2; 0.6492 ×2303, 1.298 ×801 | 180.1 | 1,387,849 | 0.000242 | 9, killed | 1 | time_limit | 9 bad (9 curve): 0.00024 at (-365.7, 369.7, 0.1) band_boundary [MA::RO__top::band=0; MA::RO__top::band=1] · 0.00028 at (-365.9, 369.9, 0.1) band_boundary [MA::RO__top::band=0; MA::RO__top::band=1] · 0.0025 at (-106.0, 111.0, 0.4) film_sidewall [MA::RO__sidewall; MA::RO__top::band=0] | 5.5–16.9 | yes | pin49 |
| disk_site_bands | band4 | 166.2; 1.298 ×3104 | 58.9 | 569,325 | 0.05136 | 0 | 0 | clean |  | 17.1–18.6 | yes | pin49 |
| disk_site_bands | band8 | 166.2; 1.298 ×801, 2.597 ×2303 | 111.8 | 402,784 → 407,093 | 0.02024 | 1, 0 | 1 | clean |  | 18.6–19.1 | yes | pin49 |
| disk_site_bands | wall100_band8 | 166.2; 2.597 ×2303, 5.194 ×801 | 180.0 | 279,127 → 413,124 | 0.00229 | 10, 5, 5, 4, killed | 4 | time_limit | 4 bad (4 curve): 0.0023 at (117.9, 17.4, 10.9) band_boundary/film_sidewall [MA::Disk__sidewall; MS::Disk__top::band=0] · 0.0064 at (-41.1, -413.3, 10.4) band_boundary/film_sidewall [MA::D1_BOTTOM_GROUND_PLANE__sidewall; MS::D1_BOTTOM_GROUND_PLANE__top::band=0] · 0.0071 at (41.4, -396.0, 6.1) band_boundary/film_sidewall [MA::D1_BOTTOM_GROUND_PLANE__bottom::band=0; MA::D1_BOTTOM_GROUND_PLANE__sidewall] | 19.1–18.8 | yes | pin49 |

### Pass `pin49-quiet` (PDK 49e539f, deciding rows, one at a time, no Palace)

| case | candidate | sizes µm (coarse; curves) | s | tets (round 0 → last) | min SICN | bad per round | rounds | stop | where (stopped runs) | load 1 min (before–after) | Palace | pass |
|---|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| disk_site_nobands | default | 166.2; 1.298 ×801 | 16.7 | 303,440 | 0.01565 | 0 | 0 | clean |  | 1.4–1.7 | no | pin49-quiet |
| disk_site_nobands | band4 | 166.2; 1.298 ×801 | 16.8 | 303,440 | 0.01565 | 0 | 0 | clean |  | 1.6–1.4 | no | pin49-quiet |
| disk_site_nobands | wall100_band8 | 166.2; 5.194 ×801 | 7.4 | 74,998 → 75,090 | 0.01589 | 2, 0 | 1 | clean |  | 1.2–1.4 | no | pin49-quiet |
| disk_site_bands | default | 166.2; 1.298 ×801 | 82.5 | 402,784 → 407,093 | 0.02024 | 1, 0 | 1 | clean |  | 1.6–1.4 | no | pin49-quiet |
| disk_site_bands | band4 | 166.2; 1.298 ×3104 | 53.4 | 569,325 | 0.05136 | 0 | 0 | clean |  | 1.5–1.6 | no | pin49-quiet |
| disk_site_bands | band8 | 166.2; 1.298 ×801, 2.597 ×2303 | 104.5 | 402,784 → 407,093 | 0.02024 | 1, 0 | 1 | clean |  | 1.5–1.5 | no | pin49-quiet |
| disk_site_bands | wall100_band8 | 166.2; 2.597 ×2303, 5.194 ×801 | 180.0 | 279,127 → 413,124 | 0.00229 | 10, 5, 5, 4, killed | 4 | time_limit | 4 bad (4 curve): 0.0023 at (117.9, 17.4, 10.9) band_boundary/film_sidewall [MA::Disk__sidewall; MS::Disk__top::band=0] · 0.0064 at (-41.1, -413.3, 10.4) band_boundary/film_sidewall [MA::D1_BOTTOM_GROUND_PLANE__sidewall; MS::D1_BOTTOM_GROUND_PLANE__top::band=0] · 0.0071 at (41.4, -396.0, 6.1) band_boundary/film_sidewall [MA::D1_BOTTOM_GROUND_PLANE__bottom::band=0; MA::D1_BOTTOM_GROUND_PLANE__sidewall] | 1.4–1.1 | no | pin49-quiet |

### Pass `features2` (PDK 49e539f, the new default, one at a time)

| case | candidate | sizes µm (coarse; curves) | s | tets (round 0 → last) | min SICN | bad per round | rounds | stop | where (stopped runs) | load 1 min (before–after) | Palace | pass |
|---|---|---|---:|---|---:|---|---:|---|---|---|---|---|
| B_1mm_nobands | default | 126.6; 3.956 ×36 | 0.4 | 7,604 | 0.05528 | 0 | 0 | clean |  | 36.9–36.9 | yes | features2 |
| simplepad_A_bands | default | 143.6; 1.122 ×60 | 6.3 | 115,346 | 0.02354 | 0 | 0 | clean |  | 12.3–11.7 | no | features2 |
| simplepad_A_bands_nb | default | 163.9; 1.281 ×60 | 4.9 | 88,628 | 0.03882 | 0 | 0 | clean |  | 40.3–39.9 | yes | features2 |
| simplepad_B_bands | default | 143.6; 1.122 ×92, 2.244 ×36 | 9.4 | 117,168 | 0.03884 | 0 | 0 | clean |  | 39.5–38.9 | yes | features2 |
| simplepad_B_bands_nb | default | 163.9; 1.281 ×140 | 31.6 | 213,040 → 213,634 | 0.0341 | 1, 0 | 1 | clean |  | 38.8–37.2 | yes | features2 |
| disk_site_nobands | default | 166.2; 1.298 ×801 | 17.5 | 303,440 | 0.01565 | 0 | 0 | clean |  | 36.6–35.8 | yes | features2 |
| disk_site_bands | default | 166.2; 1.298 ×3104 | 54.9 | 569,325 | 0.05136 | 0 | 0 | clean |  | 7.3–4.6 | no | features2 |
