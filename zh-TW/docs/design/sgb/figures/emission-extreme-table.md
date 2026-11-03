| 結果 | 組合數（共 39 個，22 個 fixture） | 內容 |
|:---|---:|:---|
| 通過全部檢查 | 32 | build 與結構檢查、寫出 BREP 與 XAO、Gmsh 載入與檢查、conformal mesh、`BRepCheck_Analyzer`、`BOPAlgo_ArgumentAnalyzer`（全部模式）、`ShapeAnalysis`、與 plan 的比對 |
| 預期的拒絕：build 階段 | 5 | `NonManifoldError` 3 個、`ContactError` 2 個；在 build 開始後數毫秒內丟出 |
| 預期的拒絕：direct writer | 2 | 5-edge cylinder wall；Gmsh-API oracle 可以寫出並得到 conformal 的 mesh |
