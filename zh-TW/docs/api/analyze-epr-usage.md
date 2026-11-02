# `analyze_epr`：儲存場 EPR 用法

`analyze_epr` 會透過 AEDT 原生功能評估保留場資料，必須明確選擇 AEDT 版本並提供授權；`reanalyze_epr` 則離線更新已解析結果的假設。請依[儲存場完整教學](../course/aedt-saved-fields.qmd)操作，相關參考資料見[參考指南](../knowledge-guide.qmd)。

## 最小呼叫

此範例會先解析保留的儲存場解，再讀取相符的 EPR 規格，最後呼叫 `analyze_epr`。請將範例資料夾換成研究實際回傳的執行資料夾。

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

## 參數

| 參數 | 說明 |
|---|---|
| `saved_solution` | 已驗證並解析為 `SavedSolution` 的儲存場資料。 |
| `aedt_version` | 明確選定的 AEDT 版本：`"2023.2"` 或 `"2024.2"`。 |
| `geometry` | 原始請求使用的平面幾何。 |
| `project_name`、`design_name` | 儲存模型的原生專案與設計識別名稱。 |
| `run_control` | 與儲存解相符的本徵模設定與求解通過控制。 |
| `epr_request` | 要評估的表面、體積與接面物理量。 |
| `output_dir` | 新分析交接資料與回傳結果的目的地。 |
| `geometry_workers` | 幾何準備階段可選的工作程序數；省略時使用實作預設值。 |
| `surface_field_selectors` | 可選對照表，將已選貢獻識別碼對應至 `"surface"` 或 `"adjacent"` AEDT 計算器操作。此選擇與描述物理貢獻側的 `SurfaceEprSpec.field_side` 分開。 |
