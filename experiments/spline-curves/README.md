# Spline curves: feasibility study (issue 109)

These scripts answer whether a smooth boundary curve (an Euler bend, a B-spline, an interpolating spline)
helps SGB mesh, and whether converting it **before SGB** into a chain of tangent circular arcs works with
SGB as it is. Nothing here is production code; no SGB module is changed.

## Where the results are

The findings, the measured tables and the recommendation are in the
[feasibility study comment on issue 109](https://github.com/NCUPHY-LTLab/scgsim/issues/109#issuecomment-6762386553).
They were measured on develop `1bc64ca`; the cell 21 curve laws are the PDK's at its revision `c75a1d4`,
coded in `arcchain.py`, so the scripts need no PDK. On a later develop several of those numbers differ;
[What differs on develop now](#what-differs-on-develop-now) says which, and why.

## What the scripts do

| file | what it does |
|---|---|
| `arcchain.py` | The exact curves (the gdsfactory Euler bend of cells 14 and 15 and its CPW edge offsets; the PDK's Hermite and B-spline strips of cell 21), their biarc fit within a bound, and two ways to put the arcs on the 1 nm lattice: `naive` (round every joint) and `tangent` (every joint exactly tangent but the one before a prescribed end tangent). |
| `scenes.py` | The polygon scenes of `tests/sgb_supported_pdk_cells.py` unchanged, and their arc-chain versions (`<cell>_arcs`, `_arcs_naive`, `_arcs_5nm`, `_arcs_50nm`). The arcs enter SGB through `arc_through_points`, the path authored `circular_arc` curves take today. |
| `measure.py` | Builds a scene (Route B, MA/MS/SA bands at 0.5 and 1.0 um by type), counts boundary pieces, emitted entities and band edges under 25 grid steps (an arc edge measured along its arc), checks the band areas against the exact erosion, and meshes it with `mesh_sgb` defaults at the provisional minSICN threshold 0.01. `--no-bands` is the control without bands; `--no-mesh` stops after the build. |
| `fit_table.py` | Arc counts, the largest deviation from the exact curve, the smallest radius and the shortest chain piece (measured along its arc), per fitting bound, against the polygons the scenes draw. |
| `summarize.py` | One table row per run from the JSON files `measure.py` wrote; a run without bands or without a mesh says so after the scene name. |

## Run

From the repository root, in the locked environment (`uv sync --locked --group dev`):

```bash
uv run --locked --no-sync python experiments/spline-curves/measure.py OUT cell14_euler_bend cell14_arcs
uv run --locked --no-sync python experiments/spline-curves/summarize.py OUT
uv run --locked --no-sync python experiments/spline-curves/fit_table.py
```

`OUT` receives, for each run, `<scene>__<mode>.json` and a folder of the same name with the model files and
meshes; keep it outside the repository. The mode is `bands` or `nobands`, with `__nomesh` after it for
`--no-mesh`, so the control without bands sits beside the run with bands. A run whose files already exist
is refused before anything runs; choose another `OUT` to measure it again.
A scene takes a few seconds to half a minute; the arc conversion itself is unoptimised Python
(0.2 to 8 s per scene). The cheapest check that the polygon and the arc path both build and mesh is
cell 13, about 10 s:

```bash
uv run --locked --no-sync python experiments/spline-curves/measure.py OUT cell13_circular_bend cell13_arcs
```

## What differs on develop now

The scripts were last run on develop `464ba5b`: `measure.py` on cell 13 with meshing, on the scenes of
cells 14, 15 and 21 without meshing, and on cell 14 without bands; `fit_table.py` in full. The comment on
issue 109 stays as measured. A rerun on this develop gives these differences:

| What changed on develop | Effect on a rerun | Measured on `464ba5b` |
|---|---|---|
| The default `FeatureRecipe` is `scgsim.meshing.features/2`: every Surface-EPR band boundary curve is sized at 4 × the narrowest band width (PR 158). A model without bands keeps the features/1 sizes. | Every mesh with bands changes. The control without bands does not. | Cell 13 polygon: 29,087 tetrahedra, minSICN 0.0176, 0 rounds, 2.4 s (issue 109: 18,608, 0.0136, 0 rounds, 1.8 s). Cell 13 exact arcs: 29,199, 0.0188, 0 rounds, 2.6 s (issue 109: 18,724, 0.0220, 0 rounds, 1.5 s). Cell 14 without bands: 12,304, 0.0788, as on issue 109. |
| Two straight band offsets that meet at a convex junction end at their crossing (PR 148). | The polygon scenes lose most of their short band edges; the arc chains keep theirs, because the rule covers line–line junctions only. | Short band edges, issue 109 → now. Polygons: cell 14 36 → 0, cell 15 84 → 0, cell 21 interpolation resampled 120 → 6, cell 21 B-spline resampled 45 → 9. Arc chains: unchanged at 0, 9, 5 and 0. |
| The rule of at most 128 points per full circle is gone (PR 148). | The strips as the PDK draws them (258 points) are no longer refused. | Both drawn strips build with the bands of their exact erosion: 12 short band edges (interpolation) and 15 (B-spline). Not meshed. |
| `arc_through_points` takes the grid as a keyword, to give a refusal's location in micrometres (PR 145). | `scenes.py` passes it; nothing measured changes. | |

The with-bands mesh table of issue 109 compares polygons and arc chains, and both sides change on this
develop; its polygon side changes the most. That comparison was not rerun with meshing here. The fit
table is unchanged.

## Correction: kinks of the naive chains

Before this correction, `naive_chain` counted only the kinks between two of its arcs. Rounding the joints
also turns the first arc's start tangent and the last arc's end tangent, so a CPW curve kinks where it
meets its straight lead-in and lead-out too. It now takes the lead directions and counts those two joints
as well. A curve that ends on the die edge (cell 21) meets an end cap at a corner, not a lead, so it has
no such joint. The tangent chains leave and arrive along the leads exactly, so their counts stand.

The naive scenes on develop `464ba5b`, before → after the correction (`kinks` in `summarize.py`):

| scene | kinks over 10⁻⁹ rad | the 8 lead kinks | largest kink | build |
|---|---|---|---|---|
| `cell13_arcs_naive` | 4 → 12 | 3.3–8.0 × 10⁻⁶ rad | 4.6 × 10⁻⁵ rad, unchanged | builds |
| `cell14_arcs_naive` | 100 → 108 | 1.4–7.2 × 10⁻⁵ rad | 8.4 × 10⁻⁴ rad, unchanged | refused, unchanged |
| `cell15_arcs_naive` | 172 → 180 | 4.1–7.1 × 10⁻⁵ rad | 6.0 × 10⁻⁴ rad, unchanged | refused, unchanged |
| `cell21_*_arcs_naive` | unchanged (310 and 138) | none | unchanged | refused, unchanged |

The comment on issue 109 gives the naive kinks as 2 to 16 × 10⁻⁴ rad and does not mention the lead
joints. The four naive scenes it reports are still refused, and the largest kink of each is unchanged.

## Correction: lengths along the arc

Two counts used an arc edge's chord where its length along the arc was meant. `measure.py` now measures a
curved band edge along its arc (`2 r asin(chord / 2r)`, as `tests/sgb_supported_kit.py` does) before it
compares it with 25 grid steps, and `fit_table.py` reports the shortest chain piece as radius × sweep.

No short-band-edge count moves. On develop `464ba5b`, all 27 scenes of `scenes.py` give the same counts
before and after (`measure.py --no-mesh`). The band arcs are at least 0.5 um in radius, so an arc edge under
25 grid steps is at most 6 × 10⁻⁴ grid steps longer than its chord.

The shortest chain piece in `fit_table.py`, in um, chord → along the arc, at the bounds 0.71, 5, 50 and
200 nm:

| edge | 0.71 nm | 5 nm | 50 nm | 200 nm |
|---|---|---|---|---|
| cell 14 inner edge | 4.409 → 4.410 | 6.626 → 6.628 | 13.317 → 13.335 | 26.185 → 26.354 |
| cell 15 inner edge | 6.757 → 6.759 | 11.282 → 11.290 | 22.618 → 22.676 | 33.959 → 34.145 |
| interpolation strip edge | 2.038 → 2.039 | 3.971 → 3.978 | 7.630 → 7.663 | 12.275 → 12.383 |
| B-spline strip edge | 3.444 → 3.445 | 6.379 → 6.384 | 13.351 → 13.386 | 16.904 → 16.969 |

The comment on issue 109 quotes one of these, cell 14's shortest arc at the snap bound, as 4.4 um; it
still rounds to 4.4 um.
