# 分析已封存的 AEDT 已儲存場解

`scgsim.aedt.analyze_epr()` 會從已完成的 HFSS 本徵模解所保留的場，重新評估指定的 EPR 數量。它會在獨立工作複本中開啟 AEDT 並執行原生場計算；不會準備模型，也不會再次求解。請在已取得完整已儲存場結果、其準備完成的幾何，以及原始 EPR 請求時使用。

## 必要輸入

| 參數 | 物件或值 | 用途 |
|---|---|---|
| `saved_solution` | `resolve_saved_solution(...)` 回傳的 `SavedSolution` | 識別並驗證完整回傳的已儲存場結果。 |
| `geometry` | 從 AEDT 規格解析出的 `PreparedPlanarGeometry` | 提供該次執行使用的已準備物理幾何與貢獻綁定。 |
| `project_name`、`design_name` | 從規格解析出的字串 | 指定 AEDT 專案與設計。 |
| `run_control` | 規格中的既有 `EigenmodeRunControl` | 提供綁定所請求場量時使用的 setup 與模態脈絡。 |
| `epr_request` | 規格中的既有 `EprAnalysisRequest` | 選擇模態，以及指定的表面、體積與接面貢獻 ID。 |
| `aedt_version` | 明確選取 `2023.2` 或 `2024.2` | 指定要用來開啟工作複本的 AEDT 版本。 |
| `output_dir` | 新的未使用路徑 | 存放獨立分析工作複本及回傳的分析產物。 |
| `surface_field_selectors` | 可選映射：已選取的 `contribution_id` 對應 `"surface"` 或 `"adjacent"` | 指定該貢獻使用的 AEDT 場計算器命令。 |

請使用從回傳的 `aedt_spec.json` 解析出的原始物件；不要重新建立幾何，也不要從標籤推測 ID。每個選擇器鍵都必須是 `epr_request` 已選取的表面貢獻。`"surface"` 會要求 `EnterSurface`；`"adjacent"` 會要求 `EnterAdjacentSurface`。選擇器記錄的是計算器操作，不是取樣場所屬的材料側。請將它與 `SurfaceEprSpec.field_side` 分開處理。

## 最小呼叫

此範例使用已完成並回傳的執行結果，以及從可攜式規格解析出的原始物件。每次呼叫都請指定新的分析資料夾。

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

此最小呼叫省略 `surface_field_selectors`，因此沿用請求既有的選擇器行為。若物理研究需要明確指定計算器操作，請傳入映射，且鍵只能使用已選取的表面貢獻 ID。選擇須有研究依據；SCGSim 不會根據所選命令推導修正。

## 結果代表什麼

開始評估前，SCGSim 會驗證已封存專案及完整結果資料夾清單。接著 AEDT 在原生工作複本中評估所請求的場量，並回傳含分析記錄的 EPR 結果。如果缺少所請求的場量或運算式，操作會明確失敗，不會改用自適應歷史資料推算。此操作需要可用的 AEDT 安裝與授權；它不是離線分析。

`reanalyze_epr()` 的用途不同：它只根據已記錄的能量分子與新的損耗假設重算薄膜損耗估計，不會開啟 AEDT。完整的讀回流程請參閱[已儲存場教學](../course/aedt-saved-fields.qmd)；來源綁定與失敗規則請參閱[AEDT 執行契約](../specs/aedt-runtime.qmd#explicit-native-field-selectors-for-saved-analysis-converging)。
