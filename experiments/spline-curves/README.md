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
| `measure.py` | Builds a scene (Route B, MA/MS/SA bands at 0.5 and 1.0 um by type), counts boundary pieces, emitted entities and band edges under 25 grid steps, checks the band areas against the exact erosion, and meshes it with `mesh_sgb` defaults at the provisional minSICN threshold 0.01. `--no-bands` is the control without bands; `--no-mesh` stops after the build. |
| `fit_table.py` | Arc counts and the largest deviation from the exact curve, per fitting bound, against the polygons the scenes draw. |
| `summarize.py` | One table row per scene from the JSON files `measure.py` wrote. |

## Run

From the repository root, in the locked environment (`uv sync --locked --group dev`):

```bash
uv run --locked --no-sync python experiments/spline-curves/measure.py OUT cell14_euler_bend cell14_arcs
uv run --locked --no-sync python experiments/spline-curves/summarize.py OUT
uv run --locked --no-sync python experiments/spline-curves/fit_table.py
```

`OUT` receives one JSON file per scene plus the model files and meshes; keep it outside the repository.
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
