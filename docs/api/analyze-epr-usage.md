# `analyze_epr`: saved-field EPR usage

`analyze_epr` evaluates retained fields through native AEDT and requires an explicitly selected release and license; `reanalyze_epr` updates assumptions on a resolved result offline. Follow the complete [saved-fields lesson](../course/aedt-saved-fields.qmd) and use the [knowledge guide](../knowledge-guide.qmd) for related references.

## Minimal call

This block resolves the retained saved-field solution and parses its matching EPR specification before calling `analyze_epr`. Replace the example directory with the returned run directory for your study.

```python
import json
from pathlib import Path

from scgsim.aedt import analyze_epr, parse_aedt_spec, resolve_saved_solution

returned_run_dir = Path(".artifacts/returned-aedt-epr")
spec = parse_aedt_spec(json.loads(
    (returned_run_dir / "aedt_spec.json").read_text(encoding="utf-8")
))
saved = resolve_saved_solution(returned_run_dir / "saved-solution-manifest.json")

result = analyze_epr(
    saved_solution=saved,
    aedt_version="2024.2",
    geometry=spec.geometry,
    project_name=spec.project_name,
    design_name=spec.design_name,
    run_control=spec.run_control,
    epr_request=spec.epr_request,
    output_dir=Path(".artifacts/saved-field-epr"),
)
```

## Parameters

| Parameter | Meaning |
|---|---|
| `saved_solution` | Verified saved-field cohort resolved as `SavedSolution`. |
| `aedt_version` | Explicit AEDT release: `"2023.2"` or `"2024.2"`. |
| `geometry` | Prepared planar geometry used by the original request. |
| `project_name`, `design_name` | Native project and design identities for the saved model. |
| `run_control` | Eigenmode setup and pass controls associated with the saved solution. |
| `epr_request` | Surface, bulk and junction quantities to evaluate. |
| `output_dir` | New destination for the analysis handoff and returned result. |
| `geometry_workers` | Optional worker count for pure geometry preparation; omit to use the implementation default. |
| `surface_field_selectors` | Optional mapping from selected contribution IDs to `"surface"` or `"adjacent"` AEDT calculator operations. This selection is separate from `SurfaceEprSpec.field_side`, which identifies the physical contribution side. |
