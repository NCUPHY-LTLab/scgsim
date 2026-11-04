# sgbproto — prototype of the redesigned SGB (phases 1, 1b, 2 and 3)

Prototype of the semantic geometry builder described in `../DESIGN_BRIEF.md`
(sections 2–12, as decided after phases 1 and 1b). Curves are straight segments or circular arcs with
exact carriers; every vertex is an integer point of the model grid (default `g = 1 nm`). It derives a complete conformal solid model from prisms
*before* any CAD kernel is involved, then emits it bottom-up through the
Gmsh/OCC API with no Boolean operation.

Everything under this folder is the only thing the prototype touches. The
numbers and files in `outputs/` are **computed** by the scripts below (so the
design pages can cite them as computed examples).

## Run

```bash
PY=.venv/bin/python  # the SCGSim development environment
cd experiments/sgb-redesign/prototype
$PY -m pytest tests -q                     # 306 tests, ~60 s (includes the direct BREP writer's)
$PY examples/run_pad_in_ground.py          # Route A and Route B -> outputs/pad_in_ground_{A,B}/
$PY examples/run_airbridge.py              # Route B, five slabs  -> outputs/airbridge_B/
$PY examples/run_inset_bands.py            # inset bands: KLayout cross-check, Palace-facing check -> outputs/inset/
$PY examples/figures.py                    # computed figures -> outputs/figures/ (HTML + SVG)
$PY examples/scaling.py                    # N = 1000, 5000, 20000 -> outputs/scaling.json (about 15 min)
```

`build(spec, tiles=T, workers=W)` is the API; the result is identical for every `T` and `W`.

Each `outputs/<case>/` holds `model.xao`, `model.sidecar.json` (planned id ↔ native tag by construction order,
primary physical groups, **unions as lists of primary group names**, embedded faces), `stage_summary.json`
(counts per stage, timings, serial/parallel split), `validation_report.json`, `identity.json`,
`arrangements.json` (every slab/level arrangement, enough to draw figures) and `sliver_report.json`.

## Modules (one per stage)

| Module | Stage | Owns |
|---|---|---|
| `arcs` | S4 | arc witness merge, arc realisation through snapped end points, per-edge deviation, arc-aware areas |
| `inset` | S3 | inset band generator: offset carriers, reflex arcs, `dist >= m` trimming, band labels |
| `klayout_check` | test | KLayout erosion oracle for the inset bands (examples/tests only) |
| `records` | S2 | `Grid`, `LineCarrier` + `Lineage`, `Footprint`, `PrismRecord`, `CurveFamilyRecord`, `Domain`, `LevelLabel` |
| `levels` | S2 | levels, slabs, active sets, sheets per level, family targets |
| `snap_rounding` | S4 | iterated snap rounding, half-open pixels, exact crossings, carrier-set preservation |
| `arrangement` | S4 | half-edge structure, cycles, faces with holes, nesting, exact representative points |
| `layers` | S4/S5 | all `S_i`, then all `A_i`, stage by stage (noding → faces → labelling → dissolution → ids) |
| `linking` | S4 | chains of level edges covering each slab edge; slab cell ↔ level cell links |
| `labels`, `dissolve` | S5 | slab labels, required edges, dissolution, refinement-only curves |
| `assembly` | S6 | vertices, edges, horizontal faces, **coalesced walls**, wall columns, volumes, shells, groups |
| `routes` | S7 | Route A / B as labelling policies; one primary group per face; PEC unions |
| `identity` | S8 | content-addressed ids (SHA-256 of canonical JSON) |
| `validate` | S9 | structural invariants (fail closed) |
| `sliver` | S9 | sliver report (shortest edges, narrowest faces, smallest angles, snap drift) |
| `emit` | S10 | bottom-up OCC emission, XAO + sidecar, mesh conformity check |
| `parallel`, `tiling` | S4–S9 | per-tile fork/join runtime with serial/parallel stage accounting; tile-parallel noding kernels |
| `pipeline`, `report`, `scenes` | — | orchestration, JSON outputs, example/random inputs |

## Phase 1b: what changed

1. **Half-open pixels and between-pixel tile cuts** are unchanged and now the official design.
2. **Wall coalescing.** A level cuts a wall only where something at that level touches it: a chain vertex of
   degree ≠ 2 in `A_i` (e.g. a port-sheet edge ending on a metal sidewall), a different chain, or a sheet face along
   the wall. Otherwise stacked walls of consecutive slabs merge into one face. Wall columns (vertical edges) are cut
   only at levels where a wall run starts/ends or a horizontal face has a vertex there, so neighbours always share
   whole edges. `pad_in_ground_B`: **115 → 68 faces** (65 faces without the port plane + the port face + 2 extra
   wall pieces where the port touches the pad wall and the cutout wall). Tests build a port that touches a sidewall
   (wall split at z = 0.1) and one that does not (no wall split) and check mesh conformity for both.
3. **Physical groups partition each dimension.** One primary group per emitted face/volume (`MA`, `MS`, `SA`,
   `EXT`, `PORT`, `VOL`; `PEC::net` is primary only for Route A sheets). Route B's "PEC of Net X" is the union
   `PEC::net = [MA::net, MS::net]`, recorded in `Topology.unions` and the sidecar, never an overlapping Gmsh group.
   Tests: no entity in two groups, every emitted entity in exactly one (also a validation check).
4. **Parallel granularity** (details and numbers below): every stage from noding to shells is a fork/join over
   per-tile item lists (`parallel.Runtime`); only volume connectivity, dissolution and the DCEL are serial.
5. **Emission profile** (below): no further obvious fix beyond explicit physical-group tags.
6. **Sliver report** per case in `sliver_report.json` (below).

## What is implemented

Everything of phase 1 (see git history of this README for the full list): records, levels/slabs, own iterated snap
rounding + DCEL, local `S_i`/`A_i`, exact labelling, dissolution, topology assembly with shells/volumes/embedded
faces, Route A/B as policies, content-addressed ids and lineage, validation, bottom-up emission with XAO + sidecar,
mesh conformity check, tile-parallel derivation, deterministic ids. Shapely is used only in tests as an oracle.

Running examples (computed; coarse meshes, conformity only):

| Case | prisms | slabs / levels | emitted volumes | faces | edges | mesh nodes | conformal |
|---|---|---|---|---|---|---|---|
| `pad_in_ground_A` (sheets) | 5 | 2 / 3 | substrate, vacuum | 15 | 44 | 655 | yes |
| `pad_in_ground_B` (0.2 µm metal, port at z = 0.1) | 5 | 4 / 5 | substrate, vacuum | **41** | 94 | 634 | yes |
| `airbridge_B` (posts + deck, five slabs) | 7 | 5 / 6 | substrate, vacuum | **37** | 76 | 559 | yes |

Volumes are exactly conserved (integer arithmetic, z scaled by the common level denominator).

## Parallel derivation (item 4)

**Runtime.** Every parallel stage is `Runtime.run(stage, fn, shared, owners)`: items (faces, edges, vertices,
volumes, wall keys) are *owned* by the tile that contains their anchor point (`T x T` cut lines between pixel
columns/rows); `fn(shared, indices)` runs once per non-empty tile; results are merged by item index. `shared` is
inherited by forked workers (no pickling of inputs); with `workers = 1` the same functions run in-process, so the
serial and parallel paths are the same code and the result cannot depend on `T` or `W`. Noding kernels use a
persistent pool and per-round hot-pixel union (the round barrier below).

| Stage | Kind | Per-tile unit |
|---|---|---|
| `noding_kernels` | parallel | crossing pixels owned by tile; hot pixels along each edge inside the tile |
| `noding_driver`, `dcel_faces`, `linking`, `dissolution` | serial | |
| `labelling` | parallel | face representative point + exact footprint tests (slab) / sheet owner (level) |
| `atomic_ids` | parallel | vertex, edge and face/cell content hashes |
| `wall_runs` | parallel | wall keys (u, v): chain tracing, coalescing runs, planarity test, wall id |
| `volume_connectivity` | serial | the global reduction (union-find over cell links) |
| `shells`, `validation_shells` | parallel | per volume: shell grouping, exact signed volume, closure/orientation/Euler |
| `assembly_*`, `*_prep`, `validation_serial` | serial | registries, loop assembly, groups, remaining checks |

**Two things the brief should say about granularity.**
(a) *Noding cannot be one independent job per tile.* Iterated rounds re-route chords whose endpoints may lie in
other tiles, so each round needs the union of hot pixels (a barrier per round, 1–3 rounds per arrangement). That is
exact and keeps the identity property, but it is the reason noding is a per-round fork/join, not one job.
(b) *Face extraction is not tiled.* Faces span tiles (the ground plane is one face with all cutouts as holes); the
DCEL (≈ 8 % of the build) and dissolution (≈ 11 %) stay serial. Everything after them that is per-entity
(ids, labels, walls, shells, validation) is per tile.

**Measured** (Route B random layouts, `pitch` 260 µm, seed 1; one subprocess per case; 64-core machine; gc paused
during the build). `ids_digest` is identical for every (T, W) of a given N and every case passed validation.
Amdahl columns: *serial stages* = wall time of serial stages; *worker compute* = summed in-tile compute of parallel
stages; *Amdahl parallel fraction* = worker compute / (serial + parallel compute); *speed-up limit* = what perfect
scaling of the parallel stages would give given the marshalling cost actually paid.

| N | T×T | workers | build s | speed-up | serial stages s | parallel stages wall s | worker compute s | Amdahl parallel fraction | speed-up limit | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1x1 | 1 | 1.4 | 1.00 | 0.5 | 0.8 | 0.8 | 0.61 | 2.54 | 324 |
| 1000 | 4x4 | 1 | 1.6 | 0.89 | 0.5 | 1.0 | 0.9 | 0.60 | 2.49 | 144 |
| 1000 | 4x4 | 8 | 1.3 | 1.08 | 0.6 | 0.6 | 1.3 | 0.66 | 2.95 | 152 |
| 1000 | 4x4 | 16 | 1.3 | 1.06 | 0.7 | 0.6 | 1.4 | 0.68 | 3.17 | 151 |
| 1000 | 4x4 | 32 | 1.4 | 1.04 | 0.6 | 0.6 | 1.5 | 0.71 | 3.42 | 152 |
| 1000 | 8x8 | 1 | 1.6 | 0.86 | 0.5 | 1.1 | 1.0 | 0.61 | 2.57 | 145 |
| 1000 | 8x8 | 8 | 1.4 | 1.01 | 0.7 | 0.6 | 1.4 | 0.67 | 3.00 | 150 |
| 1000 | 8x8 | 16 | 1.4 | 1.04 | 0.7 | 0.6 | 1.7 | 0.72 | 3.58 | 151 |
| 1000 | 8x8 | 32 | 1.6 | 0.91 | 0.6 | 0.8 | 2.0 | 0.76 | 4.14 | 150 |
| 5000 | 1x1 | 1 | 9.3 | 1.00 | 3.7 | 5.2 | 5.1 | 0.58 | 2.37 | 1358 |
| 5000 | 4x4 | 1 | 10.2 | 0.92 | 3.9 | 5.9 | 5.4 | 0.55 | 2.22 | 520 |
| 5000 | 4x4 | 8 | 7.7 | 1.21 | 4.4 | 2.8 | 6.7 | 0.61 | 2.54 | 575 |
| 5000 | 4x4 | 16 | 7.8 | 1.19 | 4.4 | 2.9 | 7.5 | 0.63 | 2.71 | 578 |
| 5000 | 4x4 | 32 | 7.7 | 1.22 | 4.3 | 2.7 | 7.5 | 0.63 | 2.72 | 575 |
| 5000 | 8x8 | 1 | 10.5 | 0.89 | 3.8 | 6.3 | 5.8 | 0.57 | 2.34 | 526 |
| 5000 | 8x8 | 8 | 7.9 | 1.18 | 4.5 | 2.8 | 7.5 | 0.63 | 2.68 | 574 |
| 5000 | 8x8 | 16 | 8.0 | 1.17 | 4.4 | 2.9 | 8.9 | 0.67 | 3.01 | 572 |
| 5000 | 8x8 | 32 | 8.2 | 1.14 | 4.4 | 3.2 | 10.0 | 0.70 | 3.29 | 578 |
| 20000 | 1x1 | 1 | 45.2 | 1.00 | 18.9 | 24.3 | 24.2 | 0.56 | 2.27 | 5272 |
| 20000 | 4x4 | 1 | 48.5 | 0.93 | 19.0 | 27.5 | 24.0 | 0.52 | 2.06 | 1929 |
| 20000 | 4x4 | 8 | 38.2 | 1.18 | 21.8 | 13.8 | 30.1 | 0.58 | 2.38 | 2128 |
| 20000 | 4x4 | 16 | 38.6 | 1.17 | 21.9 | 14.1 | 34.0 | 0.61 | 2.56 | 2143 |
| 20000 | 4x4 | 32 | 38.4 | 1.18 | 21.8 | 13.9 | 32.9 | 0.60 | 2.51 | 2138 |
| 20000 | 8x8 | 1 | 51.2 | 0.88 | 19.6 | 29.6 | 25.9 | 0.53 | 2.11 | 1919 |
| 20000 | 8x8 | 8 | 42.0 | 1.07 | 24.9 | 14.2 | 32.4 | 0.56 | 2.30 | 2134 |
| 20000 | 8x8 | 16 | 43.2 | 1.05 | 25.4 | 14.8 | 39.5 | 0.61 | 2.56 | 2136 |
| 20000 | 8x8 | 32 | 42.5 | 1.06 | 23.0 | 16.6 | 41.4 | 0.64 | 2.80 | 2129 |

Stage times of the serial case (T = 1, W = 1), seconds:

| stage | kind | N=1000 s | N=5000 s | N=20000 s |
|---|---|---|---|---|
| `noding_kernels` | parallel | 0.22 | 1.43 | 7.40 |
| `dissolution` | serial | 0.13 | 1.01 | 5.24 |
| `atomic_ids` | parallel | 0.22 | 1.17 | 4.87 |
| `shells` | parallel | 0.10 | 0.83 | 4.30 |
| `assembly_faces` | serial | 0.16 | 0.94 | 4.23 |
| `dcel_faces` | serial | 0.06 | 0.53 | 3.11 |
| `validation_shells` | parallel | 0.07 | 0.57 | 2.99 |
| `noding_driver` | serial | 0.06 | 0.48 | 2.38 |
| `labelling` | parallel | 0.10 | 0.59 | 2.38 |
| `wall_runs` | parallel | 0.11 | 0.58 | 2.34 |
| `validation_serial` | serial | 0.03 | 0.20 | 0.98 |
| `linking` | serial | 0.02 | 0.16 | 0.77 |
| `assembly_groups` | serial | 0.01 | 0.09 | 0.55 |
| `assembly_registry` | serial | 0.01 | 0.08 | 0.42 |
| `wall_prep` | serial | 0.01 | 0.07 | 0.34 |
| `shell_prep` | serial | 0.01 | 0.05 | 0.33 |
| `contact_check` | serial | 0.01 | 0.06 | 0.29 |
| `arc_realisation` | serial | 0.01 | 0.04 | 0.14 |
| `volume_connectivity` | serial | 0.00 | 0.03 | 0.10 |
| `S2_levels` | serial | 0.00 | 0.01 | 0.04 |
| **emission S10 (no mesh, no sewing)** | serial | 2.3 | 14.6 | 74.2 |

Reading the table:
* Speed-up is **1.1–1.2×** at best, at every N (it does not grow with N). Parallel-stage wall time falls 1.7–1.8×
  (22.1 s → 12.9 s at N = 20000), but the serial stages *grow* (17.6 s → 20.4 s) when workers exist, and the summed
  worker compute rises from 22 s to 27–37 s: forked CPython workers pay copy-on-write page faults on every
  refcount update, and more tiles/workers make it worse (worker compute rises with W while wall time is flat from
  W = 8). More than 8 workers buys nothing.
* The Amdahl parallel fraction of the derivation is ≈ 0.55–0.60 (so 2.2–2.5× is the ceiling with these stages
  parallel and everything else serial); `dissolution`, `assembly_faces`, `dcel_faces`, `noding_driver` and
  `validation_serial` are the serial 45 %. S10 emission is sequential and is as long as the whole derivation
  (73 s at N = 20000 versus 42 s).
* Tiles with `W = 1` cost 7–12 % extra (partitioning) — the price of the same code path.
* Peak RSS of the parent drops with tiles because per-tile results are not retained in the parent (5.1 GB → 1.8–2.0 GB
  at N = 20000), but at N = 20000 the forked workers add to the machine total.
* Pausing the cyclic garbage collector during `build` is a 30 % win on its own (11.6 s → 8.4 s at N = 5000) and
  is larger than any tile speed-up; Python's collector re-scans millions of live tuples.

## Emission profile (item 5)

cProfile of S10 (`emit`, no meshing), N = 5000 / N = 20000 (seconds, Route B random layouts):

| Call | N = 5000 | N = 20000 |
|---|---|---|
| `occ.synchronize` | 4.4 | 21.2 |
| `occ.addPlaneSurface` (63 660 / 255 048 calls) | 3.1 | 12.8 |
| `gmsh.write(model.xao)` | 2.6 | 13.5 |
| `addPhysicalGroup` (10 007 / 40 007 groups) | 0.4 | 8.0 |
| `gmsh.finalize` | 0.7 | 3.4 |
| `addLine` + `addCurveLoop` + `addPoint` + `addVolume` | ≈ 1.3 | ≈ 7.3 |
| **total** | **14.5** | **73.6** |

Before/after of what was tried:

| Change | N = 5000 | N = 20000 | Verdict |
|---|---|---|---|
| `addPhysicalGroup` without explicit tag (phase 1) | 599 s | not run | quadratic in the number of groups (Gmsh scans for a free tag) |
| explicit physical-group tags (now default) | 14.5 s | 73.6 s | kept; per-call cost still grows with the group count (43 µs → 210 µs) |
| explicit tags on every point/line/loop/surface/volume | 14.5 s | – | no change (kept: deterministic, cheaper to reason about) |
| `synchronize` once | already once | already once | – |
| `Geometry.OCCBoundsUseStl = 1` | `synchronize` 4.4 → 12.3 s | – | worse, reverted |
| `General.NumThreads = 16` | no change | – | OCC build/sync are single-threaded |
| skip the XAO write (`write_xao=False`) | −2.6 s | −13.5 s | available, off by default (the XAO is the deliverable) |

Emission time scales ≈ linearly (4× N → 5× time). Nothing in the Gmsh/OCC API removes the sequential cost: it is
`synchronize` + planar-face construction + XAO export, each proportional to the face count.

## Sliver report (item 6)

`sliver_report.json` per case (report only, geometry unchanged): `shortest_edges` (horizontal and vertical),
`narrowest_faces` (the **local feature size**: minimum distance between two non-incident boundary edges, measured on
the exact carriers with arcs analytic; `2A/P` of the chord polygon stays as the extra field `width_2a_over_p_um`),
`smallest_angles` (wedge between consecutive edges at arrangement vertices) and `snap_drift` (distance of each edge
end to its nearest source carrier; count above √2/2 g). Each entry has location (µm), plane and lineage (source
entities or carrier ids). Examples: `pad_in_ground_B` shortest edge 0.1 µm (the vertical wall segment cut by the port
plane; its lineage is `pad`/`ground`/`p1`), narrowest face 0.1 µm (that wall), smallest angle 90°, drift 0;
`pad_in_ground_A` shortest edge 7.07 µm (the rounded-corner chord), narrowest face 7.07 µm, then the 10 µm port.

## Phase 2: arcs, inset bands, arc walls, band meshing, figures

**1. Arc carriers.** `ArcCarrier` (exact centre and radius, orientation, start/end points on the circle, lineage, root
carrier) and `circle_arcs` (a full circle is four quarter arcs, so no arc edge is a loop or a parallel edge). The
running example's pad now has a **true circle** hole (r = 8 µm) and a **quarter-circle rounded corner** (r = 5 µm);
`hole_sides=32, rounded_corner=False` reproduces the phase-1b polygon pad.

**2. D3 polyline witness.** An arc is noded as chords of sagitta ≤ g/4 (101 chords per 8 µm quarter) carrying
`<arc id>#<k>` carrier ids. After snap rounding, dissolution removes degree-2 vertices whose two edges carry pieces of
the same arc, so each run of chords with no foreign hot pixel becomes **one arc edge**; the edge is realised as the
circle arc of the carrier's radius through its two snapped end points with the centre on the carrier's side.
`pad_in_ground_A` with bands: the z = 0 level arrangement drops from 2 932 noded edges to 160 edges (47 arc edges over all layers).
Per-edge deviation from the exact circle is in `sliver_report.json` (`arc_edges`): **0 for source arcs** (end points
are exact grid points) and **mean 0.057, max 0.632 grid units** for derived (offset) arcs with snapped end points,
inside the √2/2 = 0.707 bound. The slab arcs re-enter the level arrangement as their *realised* circle (derived
carrier), so a wall's bottom and top arcs are the same arc.
Validation adds `arc_edges_do_not_cross` (sampled realised arcs vs all edges, 8-unit steps) and a deviation check; the
tests repeat it with 1-unit sampling (0 crossings on A, B, with and without bands). Areas: a circle face equals πr²
to 1e-9 (exact end points); with a bar crossing it at snapped points the area matches the shapely union within the snap
bound (< 1e-3 relative). Determinism and tiling identity hold (same ids for T ∈ {1,2,3,4}, workers ∈ {1,2}, and two
`PYTHONHASHSEED`s). Volumes are exact for lines and exact-plus-analytic-segments for arcs (pad volume = analytic
5421.573 µm² × 0.2 µm to 1e-9).

**3. Inset generator** (`inset.py`; brief section 6). `R_m = R ⊖ D(m)` from *offset carriers*: lines offset, convex arcs
r → r−m (vanish for m ≥ ρ), concave arcs/holes r → r+m, **reflex corners → arcs of radius m centred at the vertex**,
convex corners stay sharp. The candidates are noded by snap rounding and an edge is kept iff its midpoint is ≥ m from
the support boundary (the `dist ≥ m` rule; dangling stubs removed), which also produces topology changes with no special
code. The result is a `required` cut family (`CurveFamilyRecord` with `arcs` and a `support` region); the band index
joins the level label and the group name: `SHEET::pad::band=2` (the pad sheet in Route A), `MA::pad::band=2` /
`MS::pad::band=2` (Route B top / bottom face) and `SA::substrate|vacuum::band=2` (the gap ring). The union
`PEC::pad` lists the pad's band groups. Support
regions: (a) pad top surface (outline minus hole), (b) the gap ring between pad and ground cutout. The generator takes
only the region's boundary: **adding the port changes no band boundary** (test: identical family records, band areas
equal to 6e-8 grid²). A port cut by bands stays one group `PORT::p1` (7 faces).

**KLayout cross-check** (band areas, margins 0.1, 0.5, 2.0 µm; `R_m` = union of bands ≥ k; KLayout: region at DBU 1 nm,
complement dilated by a 512-gon, subtracted; "exact" = chord polygon + analytic circular segments of the realised arcs):

| region | margin (µm) | exact area (µm²) | KLayout area (µm²) | difference (µm²) | relative | oracle rounding bound (µm²) | components |
|---|---|---|---|---|---|---|---|
| band_pad_top | 0 | 5421.5730 | 5421.5732 | -0.00017 | -3.1e-08 | 0.196 | 1 |
| band_pad_top | 0.1 | 5382.4661 | 5382.4575 | +0.00867 | +1.6e-06 | 0.196 | 1 |
| band_pad_top | 0.5 | 5226.2429 | 5226.2351 | +0.00779 | +1.5e-06 | 0.195 | 1 |
| band_pad_top | 2 | 4643.0033 | 4643.0019 | +0.00142 | +3.1e-07 | 0.194 | 1 |
| band_gap | 0 | 13577.3650 | 13577.3652 | -0.00014 | -9.9e-09 | 0.450 | 1 |
| band_gap | 0.1 | 13487.2885 | 13487.2992 | -0.01075 | -8.0e-07 | 0.450 | 1 |
| band_gap | 0.5 | 13127.2637 | 13127.2746 | -0.01094 | -8.3e-07 | 0.450 | 1 |
| band_gap | 2 | 11780.8720 | 11780.8722 | -0.00016 | -1.3e-08 | 0.448 | 1 |

Every difference is far below the oracle's own rounding noise (±0.5 nm on each boundary point, last column) and
≤ 1.6e-6 relative; the margin-0 rows (no erosion) isolate the polygonised-arc chord error (≈ 1e-4 µm²). Route B
(pad top face) gives identical numbers (`klayout_comparison_route_B.json`).

**Topology change** (`outputs/inset/topology_change.json`; margins 4, 9, 30 µm on the pad top; nothing fixed): B_3 (>= 30 um) is empty: R_30 vanished because the pad top has no point 30 um from its boundary (R_23 still exists, R_30 does not). R_9 splits into 3 components (KLayout) and B_2 into the 3 matching faces.
Faces per band (µm²): {"0": [1298.6015, 251.3274], "1": [1873.4919], "2": [250.5969, 36.7232, 1710.8321]}. KLayout components of `R_m` for m = 0, 4, 9, 21, 23 µm:
{'0.0': 1, '4.0': 1, '9.0': 3, '21.0': 1, '23.0': 1}. Validation passes.

**4. Walls of arc edges.** A curved wall (Route B hole circle: 4 walls; rounded corner: 1) has four shared edges (two
arcs, two verticals) and is emitted with `occ.addBSplineFilling` on the wire: **the surface is the exact cylinder**
(max radial deviation from the exact cylinder 1.2e-11 grid units, sampled on an 11×11 grid of the surface), the mesh
stays conformal. `addSurfaceFilling` with default options on the same wire is *not* exact (278 grid units = 0.28 µm
off the cylinder for r = 8 µm; 26–57 grid units with degree-3/50-point/constraint-point options), so it is only the
fallback for walls with more than four boundary edges. Arc centre points are construction aids (created, used by
`addCircleArc`, removed; otherwise they mesh as stray nodes) and are listed in the sidecar.

**5. Palace-facing check** (`pad_in_ground_A` with bands, 40 µm mesh with a size field of 1 µm near band curves):
13 222 nodes, 70 986 tets, conformal, 0 duplicate nodes. Every band is its own physical surface group; each of its
triangles is a face of exactly one tet of the substrate and one of the vacuum volume, and every triangle's centroid lies
inside its band (no element crosses a band boundary):

| band group | faces | triangles | tets on each side | all triangles shared by tets | centroids inside band |
|---|---|---|---|---|---|
| `SA::substrate|vacuum::band=0` | 2 | 1753 | 1753 + 1753 | True | True |
| `SA::substrate|vacuum::band=1` | 2 | 1804 | 1804 + 1804 | True | True |
| `SA::substrate|vacuum::band=2` | 2 | 3522 | 3522 + 3522 | True | True |
| `SA::substrate|vacuum::band=3` | 1 | 3574 | 3574 + 3574 | True | True |
| `SHEET::pad::band=0` | 2 | 790 | 790 + 790 | True | True |
| `SHEET::pad::band=1` | 2 | 807 | 807 + 807 | True | True |
| `SHEET::pad::band=2` | 2 | 1591 | 1591 + 1591 | True | True |
| `SHEET::pad::band=3` | 1 | 1607 | 1607 + 1607 | True | True |

**6. Figures** (`outputs/figures/`; self-contained Plotly HTML via `include_plotlyjs=True` plus matplotlib SVG, since
kaleido is not installed): `fig_level_arrangement_bands` (a; view buttons for chip/cutout/pad/notch/corner/hole),
`fig_exact_arcs_vs_512gon` (b; offset arcs vs the KLayout 512-gon boundary and their radial deviation in nm: exact ≈ 0,
512-gon ±1.4 nm from integer rounding), `fig_dissolution_before_after` (c; 2 932 → 160 edges), `fig_exploded_volumes_B`
(d; the 0.2 µm metal layer is drawn 200× thicker so its walls are visible).

## Phase 3: fixes from the documentation review

1. **Different-Net contact is an error in every route** (`contacts.py`, `ContactError`): after the layers are built,
   a vertex (or a face stacked below/above) of any slab or level arrangement whose incident faces carry conductors
   of more than one Net fails, naming both Nets, the entities of each that reach the point, the location (µm), the
   plane and the grid. This covers two Route-A sheets sharing an edge or a vertex, finite metal sharing an edge or
   vertex, a post of one Net standing on metal of another, and a contact that exists only on the grid (a 4 nm gap
   between Nets passes at g = 1 nm and fails at g = 10 nm). Same-Net contact (a shared edge, or sheets meeting at a
   corner) passes.
2. **Refinement pieces never vanish silently.** A `refinement_only` curve in a cell with no face, or in a face of no
   emitted volume, raises `SgbError`. The (edge, host face) pairs are in `refinement_edges` and the (face, host volume)
   pairs in `embedded_faces` of **both** sidecars (gmsh-API oracle and direct writer; a test compares them).
3. **The `dissolve` edge policy is removed**: tile seams are not a family. `CurveFamilyRecord` rejects any policy but
   `required` / `refinement_only`.
4. **Area-weighted membership** (`TopoFace.membership`, `identity.json` faces, direct-writer sidecar
   `lineage.membership`): per face, for every contributing source entity (prisms of the slabs below/above and the
   sheets on the level), the fraction of the face area covered by its footprint, computed exactly from the raw cells
   (integer chord areas). Shares are non-exclusive and may sum to more than 1 (every cell is also covered by the
   background substrate/vacuum prisms; a conductor wins the label but the dielectric footprint still covers it). Walls
   have no footprint: the share is the fraction of the bottom boundary length carried by edges of that entity.
5. **Straight inset members keep `<support>/ring<r>/edge<k>`** as lineage (`Lineage.member`) and in their carrier id
   (`band_pad_top/ring0/edge8@m100.0`: primitive, margin, piece); reflex arcs are `.../ring<r>/corner<k>@m<margin>`.
6. **Validation: `band_area_monotone`.** The bands of every inset family partition its support region (areas
   from the realised arcs, region area analytic) and `area(R_m)` (union of bands >= k) never grows with `m`; a test
   breaks the partition and checks it fails closed.
7. **Sliver width = local feature size** (`feature_size.py`): minimum distance between non-incident boundary edges on
   the exact carriers (segment/arc distances are analytic: end points plus the stationary points on the line of
   centres or centre-to-segment foot); walls use `min(height, distance between the vertical edges)`. Large faces use a
   spatial hash (exact below the cell size, flagged `bounded` above).
8. **Naming.** Route A band groups are `SHEET::<net>::band=k`; `surface_epr_rows` ({row -> group}) in every sidecar and
   `Topology.surface_epr_rows` map the `MA`/`MS` rows of a sheet to that group (and every other banded group to
   itself). The Route B band cases now band the pad's top face (`MA::pad::band=k`) **and** bottom face
   (`MS::pad::band=k`, new family `band_pad_bottom` at z = 0).
9. `stage_summary.json` / `ModelSpec.name` carry the `_bands` suffix for band cases.
10. All outputs regenerated (examples, `outputs/inset`, `outputs/figures`, `outputs/e3_brep`, `outputs/scaling.json`).

**Output changes** (faces / edges / groups; vertices in brackets):

| case | before | after |
|---|---|---|
| `pad_in_ground_A` | 15 / 44 / 8 (39) | 15 / 44 / 8 (39), unchanged |
| `pad_in_ground_B` | 41 / 94 / 10 (70) | 41 / 94 / 10 (70), unchanged |
| `airbridge_B` | 37 / 76 / 7 (44) | 37 / 76 / 7 (44), unchanged |
| `pad_in_ground_A_bands` | 33 / 176 / 15 (201), groups `MA::pad::band=k` | 33 / 176 / 15 (201), groups `SHEET::pad::band=k` |
| `pad_in_ground_B_bands` | 53 / 202 / 18 (220) | **59 / 250 / 21 (289)** (adds the 4 `MS::pad::band=k` bottom faces) |

Other changed values: every `narrowest_face_um` in the sliver reports (feature size instead of 2A/P: A 5.66 -> 7.07,
B 0.0998 -> 0.1, bands 0.0707 -> 0.0767), band carrier ids (`band_pad_top/m100/L3` -> `band_pad_top/ring0/edge8@m100.0`,
so edge ids of band edges), `identity.json` faces gain `membership`, the gmsh-API sidecar gains `surface_epr_rows`,
`refinement_edges`, `embedded_faces`, `inside_out_solids`.

**Known limitation found on the way: the gmsh-API oracle cannot guarantee the sense of a solid.** `addSurfaceLoop`
(sewing) sets the solid's sense in a way that does not follow the face order, the first face or the tag order, so some
solids come out inside out (negative `getMass`) while meshing still works. `emit` tries up to 12 deterministic face
orders per solid, records the rest in the sidecar (`inside_out_solids`) and the six test cases never need the fallback;
random layouts with thousands of faces do (N = 1000: the vacuum). The direct BREP writer sets every orientation
explicitly, which is another argument for D8. `Same-Net` corner-only contact of finite metal is a non-manifold pinch of
the conductor volume (a vertical edge used four times) and fails validation: sheets are fine, finite metal needs an
edge, not a corner.

## Findings from phase 2 (for `DESIGN_BRIEF.md`)

1. **Offset carriers are exact only up to the grid.** The offset of an oblique edge (the 45° chamfer) is a line whose
   position involves √2, so it is not representable; the generator snaps the curve end points like every vertex. "Exact"
   in section 6 should read "exact carriers, snapped end points": the measured effect is ≤ 0.63 grid units on arcs and
   ≤ 1.6e-6 relative on band areas against KLayout.
2. **Circles must be split.** A full circle has one vertex, which is a loop edge; a half circle shares both end points
   with its chord (parallel edges). The DCEL (and the planned XAO topology) needs every arc edge to join two distinct
   vertices and no two edges to join the same pair, so circles are four quarter arcs and reflex arcs are split at 90°.
3. **Merging a witness chain needs a clean vertex.** Chords merge only where no foreign carrier makes a hot pixel
   between them; a crossing splits the arc, and each piece is re-realised through its own snapped end points
   (deviation ≤ 0.63 here). Dissolved foreign curves leave no trace (the vertex merges back).
4. **The witness is expensive before it merges**: 8 µm quarter arcs are 100 chords each, so the pad's hot pixels grow
   from 69 (polygon version: 32-gon) to 516, and the banded level arrangement has 2 932 noded edges that dissolve to
   160. Witness size grows like √(r/g); fine for prototype figures, a cost to budget for arc-heavy layouts.
5. **Volumes are exact only for lines.** With arcs the chord polyhedron is still exact integer arithmetic, but each
   cylinder wall adds a circular-segment term (float): the pad volume matches the analytic value to 1e-9 relative, not
   exactly. "Exactly conserved" in the validation section holds for polygonal models only.
6. **Floating-point enters band classification.** Which band a cell belongs to is decided by the distance of its
   representative point to the support boundary (float); a cell thinner than the snap bound could be classified
   differently from the exact carrier. Topology (which edges exist) stays integer.
7. **Route A names.** A sheet face is the metal-air *and* metal-substrate surface at once: the banded pad sheet is the
   group `SHEET::pad::band=k` (brief section 14) and both Surface-EPR rows reference it (`surface_epr_rows`).
8. **Wall surface construction.** `addBSplineFilling` is exact for the four-edge wall of an arc; `addSurfaceFilling` is
   not (0.03–0.28 µm off at r = 8 µm), so walls with more than four boundary edges (a hot pixel on the arc) still
   need the direct BREP writer (D8) or a split wall.
9. **Arc centres need construction points in Gmsh** (`addCircleArc`); left in the model they mesh as stray nodes and
   break the one-planned-id-per-native-entity rule, so they are created and removed around the arc.
10. **Port in bands.** Band curves cut the port rectangle into 7 faces; they stay one group, `PORT::p1`, because a port
    is not part of the Surface-EPR surface.

## Findings from phase 1b (for `DESIGN_BRIEF.md`)

1. **Wall face ids cannot include the vertical edges**, because the vertical column segments depend on the global
   set of cut levels (known only after all walls exist). A wall's id is therefore
   `H(domains, u, v, z-range, bottom and top chain edge ids)`; the brief's "face = H(label, boundary edge ids)"
   should say vertical edges are derived data of the walls.
2. **Coalescing needs two more rules than the brief states**: (i) a *sheet face along a wall chain* cuts the wall
   even if no vertex lies on it; (ii) a wall column is split at levels where a *neighbouring wall run* ends or a
   *horizontal face has a vertex on that column* (a sheet touching only a wall corner), otherwise the faces would
   not share whole edges. Both are implemented and tested by conformity.
3. **Noding rounds are a barrier** (see Parallel (a)); the brief's "per level × tile job" holds for everything after
   noding, not for the rounds themselves.
4. **Python parallelism limits**: forked CPython workers inflate compute (COW refcount faults) and the serial
   fraction ≈ 0.45 caps the derivation at ≈ 2.3×. Realistic larger gains need either coarser jobs (per layer or
   per net group), a non-CPython kernel for noding/DCEL, or fewer Python objects (flat arrays). This is a limit of
   the prototype language, not of the algorithm.
5. **GC pause** is part of the performance model (30 %); a production implementation should not allocate per-edge
   Python objects.
6. **`addPhysicalGroup` cost grows with the number of groups** even with explicit tags; a design with one group per
   Net (2N + O(1) groups) pays for it. Per-Net surface groups could be written as one group per *kind*
   (`MA`, `MS`) plus a per-entity attribute table in the sidecar if N reaches 10⁵.
7. **Route B port**: embedded face, XAO cannot carry it (now stated in the brief); the sidecar `embedded` map and the
   `embedded_note` carry the `(face, host volume)` pairs.
8. Earlier phase-1 findings that the brief has adopted (pixel convention, drift, tile seams, level labels via links,
   overlapping groups) are no longer listed here.

## Known limitations

* Lines and circular arcs only (no B-splines, no ellipses); arcs are < 360° (a circle is four quarter arcs). Route C not implemented.
* Footprint tests and inset band classification use the polyline witness / float distances at representative points,
  never near a boundary; only sliver cells smaller than the snap bound can be classified differently from the exact carrier.
* The DCEL, dissolution and linking are serial; faces that span tiles are not stitched per tile (they are not
  needed to be, see Parallel (b)).
* The non-planar wall path (`addSurfaceFilling`) is untested (never triggered).
* Mesh quality is not evaluated; the mesh is only used to prove conformity.
* The sliver report measures; the Human-gated thresholds (D2) are not applied anywhere.
