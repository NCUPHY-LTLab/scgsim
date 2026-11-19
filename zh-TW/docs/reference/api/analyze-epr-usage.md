# `analyze_epr`：儲存場 EPR 用法

`analyze_epr` 會透過 AEDT 原生功能評估保留場資料，必須明確選擇 AEDT 版本並提供授權；`reanalyze_epr` 則離線更新已解析結果的假設。請依[儲存場完整教學](../course/aedt-saved-fields.qmd)操作，相關參考資料見[參考指南](../knowledge-guide.qmd)。

## 最小呼叫

此範例會先解析保留的儲存場解，再讀取相符的 EPR 規格（其 SGB 模型綁定到執行資料夾的 `model/` 檔案），最後呼叫 `analyze_epr`。請將範例資料夾換成研究實際回傳的執行資料夾。

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

## 參數

| 參數 | 說明 |
|---|---|
| `saved_solution` | 已驗證並解析為 `SavedSolution` 的儲存場資料。 |
| `aedt_version` | 明確選定的 AEDT 版本：`"2023.2"` 或 `"2024.2"`。 |
| `model` | 儲存解所求解的 `PreparedSgbModel`：回傳執行的 `spec.model`，或 `EigenmodeSim.prepared_model`。其他模型或原生建模器的解會被拒絕。 |
| `project_name`、`design_name` | 儲存模型的原生專案與設計識別名稱。 |
| `run_control` | 與儲存解相符的本徵模設定與求解通過控制。 |
| `epr_request` | 要評估的表面、體積與接面物理量。 |
| `output_dir` | 新分析交接資料與回傳結果的目的地。 |
| `surface_field_selectors` | 可選對照表，將 sidecar 的 Surface-EPR 群組（`epr_request.surface_epr_groups` 所選者）對應至 `"surface"`（讀取該列面上的場）或 `"adjacent"`（讀取緊鄰面外的場）。未指定的群組讀取 `"surface"`。 |
