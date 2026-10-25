# SCGSim

**使用 Python 模擬超導電路版圖。**

SCGSim 會準備 Palace 與 Ansys AEDT 模擬，並回傳具單位的電容、共振模態、S 參數及單位長度線路結果。

## 選擇模擬目標

| 目標 | 從這裡開始 |
|---|---|
| 電容 | [Palace Electrostatic 或 AEDT Q3D](docs/tutorials/capacitance.qmd) |
| 共振與 EPR | [Palace 或 HFSS 模態](docs/tutorials/resonance.qmd) · [能量參與率](docs/tutorials/resonance-epr.qmd) |
| Driven S 參數 | [HFSS Modal 或 Terminal](docs/tutorials/scattering.qmd) |
| 傳輸線參數 | [AEDT Q2D](docs/tutorials/transmission-line.qmd) |

## 準備、設定、執行與讀取

定義幾何並設定模型，在指定求解主機上執行，再讀取附有單位的回傳量。

![SimplePad 版圖：方形焊盤、周圍接地與已繪製定位片。](docs/assets/simple-pad-layout.png)

**[安裝 SCGSim](docs/course/installation.qmd)** · **[選擇模擬流程](docs/examples.qmd)**
