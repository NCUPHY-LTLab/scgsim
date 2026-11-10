# Spline curves: feasibility study (issue 109)

These scripts answer whether a smooth boundary curve (an Euler bend, a B-spline, an interpolating spline)
helps SGB mesh, and whether converting it **before SGB** into a chain of tangent circular arcs works with
SGB as it is. Nothing here is production code; no SGB module is changed. The findings are on issue 109.

## What the scripts do

| file | what it does |
|---|---|
| `arcchain.py` | The exact curves (the gdsfactory Euler bend of cells 14 and 15 and its CPW edge offsets; the PDK's Hermite and B-spline strips of cell 21), their biarc fit within a bound, and two ways to put the arcs on the 1 nm lattice: `naive` (round every joint) and `tangent` (every joint exactly tangent but the one before a prescribed end tangent). |
| `scenes.py` | The polygon scenes of `tests/sgb_supported_pdk_cells.py` unchanged, and their arc-chain versions (`<cell>_arcs`, `_arcs_naive`, `_arcs_5nm`, `_arcs_50nm`). The arcs enter SGB through `arc_through_points`, the path authored `circular_arc` curves take today. |
| `measure.py` | Builds a scene (Route B, MA/MS/SA bands at 0.5 and 1.0 um by type), counts boundary pieces, emitted entities and band edges under 25 grid steps, checks the band areas against the exact erosion, and meshes it with `mesh_sgb` defaults at the provisional minSICN threshold 0.01. `--no-bands` is the control without bands. |
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
(0.2 to 8 s per scene).
