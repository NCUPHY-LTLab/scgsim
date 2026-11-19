# `analyze_epr`: saved-field EPR usage

`analyze_epr` evaluates retained fields through native AEDT and requires an explicitly selected release and license; `reanalyze_epr` updates assumptions on a resolved result offline. Follow the complete [saved-fields lesson](../../guides/saved-fields.qmd) and use the [Reference overview](../index.qmd) for related references.

## Minimal call

This block resolves the retained saved-field solution and parses its matching EPR specification, whose SGB model is bound to the run folder's `model/` files, before calling `analyze_epr`. Replace the example directory with the returned run folder for your study.

```python
import json
from pathlib import Path

from scgsim.aedt import analyze_epr, parse_aedt_spec, resolve_saved_solution

returned_run_dir = Path(".artifacts/returned-aedt-epr")
spec = parse_aedt_spec(
    json.loads((returned_run_dir / "aedt_spec.json").read_text(encoding="utf-8")),
    base_dir=returned_run_dir,
)
saved = resolve_saved_solution(returned_run_dir / "saved-solution-manifest.json")

result = analyze_epr(
    saved_solution=saved,
    aedt_version="2024.2",
    model=spec.model,
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
| `model` | The `PreparedSgbModel` the saved solution was solved from: `spec.model` of the returned run, or `EigenmodeSim.prepared_model`. A solution of another model, or of the native builder, is refused. |
| `project_name`, `design_name` | Native project and design identities for the saved model. |
| `run_control` | Eigenmode setup and pass controls associated with the saved solution. |
| `epr_request` | Surface, bulk and junction quantities to evaluate. |
| `output_dir` | New destination for the analysis handoff and returned result. |
| `surface_field_selectors` | Optional mapping from sidecar Surface-EPR groups (those `epr_request.surface_epr_groups` selects) to `"surface"` (the field on the row's faces) or `"adjacent"` (the field just off them). A group without a selector reads `"surface"`. |
