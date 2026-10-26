# Analyze a sealed AEDT saved-field solution

`scgsim.aedt.analyze_epr()` re-evaluates requested EPR quantities from a completed HFSS Eigenmode solution's retained fields. It opens AEDT in a separate workcopy and performs native field evaluation; it does not prepare a model or solve again. Use it when the saved-field cohort, its prepared geometry and its original EPR request are all available.

## Required inputs

| Argument | Object or value | Purpose |
|---|---|---|
| `saved_solution` | `SavedSolution` from `resolve_saved_solution(...)` | Identifies and verifies the complete returned saved-field cohort. |
| `geometry` | `PreparedPlanarGeometry` retained in the parsed AEDT spec | Provides the prepared physical geometry and contribution bindings used by the run. |
| `project_name`, `design_name` | Strings from the parsed spec | Identify the AEDT project and design. |
| `run_control` | Existing `EigenmodeRunControl` from the spec | Supplies the recorded setup and mode context used to bind the requested fields. |
| `epr_request` | Existing `EprAnalysisRequest` from the spec | Selects mode and requested surface, bulk and junction contribution IDs. |
| `aedt_version` | Explicitly selected `2023.2` or `2024.2` | Selects the AEDT release that will open the workcopy. |
| `output_dir` | New, unused path | Holds the separate analysis workcopy and returned analysis artifacts. |
| `surface_field_selectors` | Optional mapping from selected `contribution_id` to `"surface"` or `"adjacent"` | Chooses the AEDT calculator command for that contribution. |

Use the original objects parsed from the returned `aedt_spec.json`; do not rebuild geometry or infer IDs from labels. Every selector key must name a surface contribution selected by `epr_request`. `"surface"` requests `EnterSurface`; `"adjacent"` requests `EnterAdjacentSurface`. The selector records the calculator operation, not the material side of the sampled field. Keep it distinct from `SurfaceEprSpec.field_side`.

## Minimal call

This example uses a completed returned run and the exact objects parsed from its portable spec. Select a new analysis directory for each call.

```python
import json
from pathlib import Path

from scgsim.aedt import (
    analyze_epr,
    parse_aedt_spec,
    resolve_saved_solution,
)

returned_run_dir = Path(".artifacts/returned-aedt-epr")
spec = parse_aedt_spec(json.loads(
    (returned_run_dir / "aedt_spec.json").read_text(encoding="utf-8")
))
saved = resolve_saved_solution(
    returned_run_dir / "saved-solution-manifest.json"
)

analysis = analyze_epr(
    saved_solution=saved,
    geometry=spec.geometry,
    project_name=spec.project_name,
    design_name=spec.design_name,
    run_control=spec.run_control,
    epr_request=spec.epr_request,
    aedt_version="2024.2",
    output_dir=Path(".artifacts/aedt-saved-field-analysis"),
)
```

This minimal call omits `surface_field_selectors`, so it retains the request's existing selector behavior. If the physical study requires an explicit calculator choice, pass a mapping keyed only by selected surface contribution IDs. Use only choices justified by the study; SCGSim does not infer a correction from the selected command.

## What the result means

Before evaluation, the reader verifies the sealed project and complete result-directory inventory. AEDT then evaluates the requested fields in a native workcopy and returns an EPR result with its analysis record. If a requested field or expression is missing, the operation fails rather than deriving a value from adaptive history. This operation requires an available AEDT installation and license; it is not offline analysis.

`reanalyze_epr()` has a different role: it recomputes film loss estimates from already recorded energy numerators and new assumptions without opening AEDT. Use the [saved-field course](../course/aedt-saved-fields.qmd) for the full retrieval workflow and the [AEDT runtime contract](../specs/aedt-runtime.qmd#explicit-native-field-selectors-for-saved-analysis-converging) for binding and failure rules.
