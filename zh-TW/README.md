# SCGSim

[![License: Apache-2.0](docs/assets/badges/license.svg)](https://github.com/NCUPHY-LTLab/scgsim/blob/develop/LICENSE)
[![Python: 3.12–3.13](docs/assets/badges/python.svg)](docs/course/installation.qmd)
[![Tutorial: Layout EM](docs/assets/badges/tutorial.svg)](docs/examples.qmd)
[![AEDT: HFSS / Q3D / Q2D](docs/assets/badges/aedt.svg)](docs/backend-support.qmd)
[![Palace: Electrostatic / Eigenmode](docs/assets/badges/palace.svg)](docs/course/palace-eigenmode.qmd)
[![Gmsh: Meshing](docs/assets/badges/gmsh.svg)](docs/concepts/geometry.qmd)
[![gsim: Unofficial Downstream](docs/assets/badges/gsim.svg)](docs/goals-and-upstream.qmd)

![SCGSim 虎鯨與企鵝，展示晶片版圖、有限元素網格及電磁場示意。](docs/assets/scgsim-overview-orca-penguin.png)

SCGSim 是用於超導電路電磁模擬的 Python 工具。你以命名導體與材料描述版圖；SCGSim 會準備 Palace 或 Ansys Electronics Desktop（AEDT）的輸入，再將求解器輸出載入報告，並附上單位、導體名稱與模態編號。

SCGSim 源自 [gsim](https://github.com/gdsfactory/gsim)，目前已是具有自有幾何建構器與求解器 API 的獨立專案，並非 gsim 的官方發行版本。

## 可計算的量

<div class="table-responsive">

| 問題 | 工作流程 | 結果 |
|---|---|---|
| 導體之間的電性耦合有多強？ | Palace Electrostatic 或 AEDT Q3D | 電容矩陣（F） |
| 共振在哪裡？能量分布如何？ | Palace Eigenmode 或 HFSS Eigenmode | 模態頻率（Hz）、品質因數、能量參與率 |
| 電路在端口之間如何響應？ | HFSS Driven Modal 或 Driven Terminal | 隨頻率變化的 S 參數 |
| 有限結構的導體矩陣為何？ | AEDT Q3D | C/G 與 R/L 矩陣 |
| 傳輸線的單位長度特性為何？ | AEDT Q2D | 單位長度 C/G/L/R |

</div>

SCGSim 不支援 Palace Driven 或 Palace Magnetostatic 模擬。完整清單請見[支援的計算](docs/backend-support.qmd)。

## 工作方式

![版圖與材料定義物理問題。SCGSim 準備模擬資料夾；你在本機或其他主機執行求解器，再由 SCGSim 讀取回傳結果。](docs/assets/scgsim-workflow.svg)

1. **描述模型。** 從 PDK 載入元件，並將形狀指派給電氣導體（Net），例如 `pad` 和 `ground`。
2. **選擇計算。** 設定 Electrostatic、Eigenmode、Driven、Q3D 或 Q2D 問題，以及網格和求解器參數。
3. **準備模擬資料夾。** SCGSim 寫入網格、求解器設定與啟動指令檔，不會自行啟動求解器。
4. **執行求解器。** 可在同一台電腦、安裝 AEDT 的工作站或 HPC 叢集執行。
5. **讀取結果。** 載入回傳資料夾，在報告中取得電容、頻率、參與率或 S 參數。

其他功能：

- **曲線幾何。** Palace 模型保留來源元件中繪製的圓弧與樣條曲線，不會將它們轉成多邊形。
- **能量參與與損耗。** 計算表面、體積與接面參與率，也能在不重新求解的情況下變更損耗假設。
- **網格控制。** 設定 Palace 網格器、執行緒數與幾何階數，並在執行前檢視網格摘要。

## 開始使用

先[安裝 SCGSim](docs/course/installation.qmd)，再完成第一個教學[焊墊電容](docs/tutorials/capacitance/prepare.qmd)。此範例使用單一矽晶粒上的 200 µm 見方焊墊，周圍是接地平面。每個教學分為三步：準備模擬資料夾、執行求解器、分析回傳結果。

![SimplePad：接地區域內的方形焊墊；東側間隙有一小片定位片，供共振教學連接接面。](docs/assets/simple-pad-layout.png)

[教學總覽](docs/examples.qmd)列出所有教學與操作指南。

### 系統需求

- Python 3.12 或 3.13。
- Palace 計算：執行求解器的電腦須有 Palace 執行檔。
- AEDT 計算：AEDT 2024.2、有效授權及 PyAEDT 1.3.0。

安裝 SCGSim 不會一併安裝 Palace 或 AEDT。你可以在筆記型電腦準備模擬，再到其他主機執行求解器。

## 文件

- [教學與操作指南](docs/examples.qmd)：逐步完成計算。
- 概念：[選擇物理問題](docs/concepts/problems.qmd)、[描述幾何](docs/concepts/geometry.qmd)、[能量參與與損耗](docs/eigenmode-epr.qmd)、[邊界與數值方法](docs/concepts/model-and-numerics.qmd)。
- [參考資料](docs/backend-support.qmd)：精確 API 與輸入規則。
- [維護者指南](docs/ownership.qmd)：提供 SCGSim 貢獻者使用。

## Python API

```python
from scgsim.geometry import GeometryPlan, VacuumRegionSpec
from scgsim.palace import ElectrostaticSim, EigenmodeSim, resolve_palace_result
```

AEDT 工作流程位於 `scgsim.aedt`。若要升級既有專案，請參閱 [API 遷移指南](docs/api-migration.qmd)。
