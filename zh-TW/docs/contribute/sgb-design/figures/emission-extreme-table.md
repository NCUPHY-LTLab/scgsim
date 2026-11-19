| 結果 | 組合數（共 63 個，23 個 fixture） | 內容 |
|:---|---:|:---|
| 通過全部檢查 | 54 | build 與結構檢查、寫出 BREP 與 XAO、Gmsh 載入與 `verify_loaded`、conformal mesh、`BRepCheck_Analyzer`、`BOPAlgo_ArgumentAnalyzer`（全部模式，含 self-interference）、`ShapeAnalysis`、與 plan 的比對 |
| 預期的拒絕：build 階段 | 9 | `ContactError` 3 個、`NonManifoldError` 6 個；在 build 開始後數毫秒內丟出 |
