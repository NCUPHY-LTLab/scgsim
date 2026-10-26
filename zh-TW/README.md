# SCGSim

**使用 Python 模擬超導電路版圖。**

[![License: Apache-2.0](docs/assets/badges/license.svg)](https://github.com/NCUPHY-LTLab/scgsim/blob/develop/LICENSE)
[![Python: 3.12–3.13](docs/assets/badges/python.svg)](docs/course/installation.qmd)
[![Tutorial: Layout EM](docs/assets/badges/tutorial.svg)](docs/examples.qmd)
[![AEDT: HFSS / Q3D / Q2D](docs/assets/badges/aedt.svg)](docs/backend-support.qmd)
[![Palace: Electrostatic / Eigenmode](docs/assets/badges/palace.svg)](docs/course/palace-eigenmode.qmd)
[![Gmsh: Meshing](docs/assets/badges/gmsh.svg)](docs/concepts/geometry.qmd)
[![gsim: Unofficial Downstream](docs/assets/badges/gsim.svg)](docs/goals-and-upstream.qmd)

![SCGSim 虎鯨與企鵝，介紹晶片版圖、有限元素網格與電磁場示意。](docs/assets/scgsim-overview-orca-penguin.png)

SCGSim 會準備 Palace 與 Ansys AEDT 模擬，並回傳具單位的電容、共振模態、S 參數及單位長度線路結果。

## 安裝

先安裝 [uv](https://docs.astral.sh/uv/getting-started/installation/) 與 Git，再建立研究專案。以下指令會安裝本頁範例使用的 SCGSim `2.0.0rc15` 與 OrPen SC PDK 元件庫；兩者都固定到明確的來源版本。

```bash
uv init --python 3.12 scgsim-study
cd scgsim-study
uv add 'scgsim[aedt] @ git+https://github.com/NCUPHY-LTLab/scgsim.git@e018c587d5fab1ed1fa6e32f2b00cc31663b941c' \
  'orpen-sc-pdk @ git+https://github.com/NCUPHY-LTLab/orcapen-sc-pdk.git@51985d9507025e8241205712e866c0b804de29b9'
```

`uv` 會建立 `.venv`，並將解析後的版本寫入 `uv.lock`。把 `pyproject.toml` 和 `uv.lock` 留在研究專案中，其他電腦即可用 `uv sync` 重建環境。

> **執行條件：** 下列範例需要本機已安裝 AEDT Q3D 2024.2 並具有可用授權。Python 套件不包含原生求解器；其他後端的設定見[安裝指南](docs/course/installation.qmd)。

## 使用範例：用一支腳本完成 SimplePad 電容抽取

本例使用 200 µm 方形金屬焊盤、20 µm 間隙及周圍接地，放在 1000 µm 方形晶片上。元件庫提供版圖、材料與層堆疊；腳本明確宣告 `pad` 與 `ground` 的電氣歸屬，再交給 Q3D 抽取電容與電導矩陣。

![SimplePad 版圖：方形焊盤、周圍接地與已繪製定位片。](docs/assets/simple-pad-layout.png)

將以下完整程式存成 `simplepad_q3d.py`。四個區塊依序完成「建立來源 → 準備求解器輸入 → 執行 → 讀取」，不需要先執行其他頁面的程式。

```python
from pathlib import Path
from uuid import uuid4
import argparse
import subprocess
import sys

import gdsfactory as gf
import orpen_sc_pdk
from orpen_sc_pdk import get_material_records
from orpen_sc_pdk.tech import get_single_die_layer_stack
from scgsim.geometry import GeometryPlan
from scgsim.aedt import (
    AedtResources,
    MatrixRunControl,
    PdkMaterial,
    prepare_q3d_from_geometry,
    resolve_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-only", action="store_true")
    args = parser.parse_args()

    # 1. Build the layout and declare electrical ownership.
    orpen_sc_pdk.activate()
    component = gf.get_component(
        "square_pad_capacitor",
        pad_side_um=200.0,
        gap_um=20.0,
        die_side_um=1000.0,
        junction_width_um=2.0,
    )
    records = get_material_records()
    plan = GeometryPlan(
        component=component,
        root_id="chip",
        layer_stack=get_single_die_layer_stack(),
        material_records=records,
        coupon_padding_um=0.0,
    )
    plan.set_nets({"pad": ("chip/PAD",), "ground": ("chip/GROUND",)})
    plan.set_vacuum_region(padding=(250.0, 250.0, 1000.0))
    snapshot = plan.prepare()

    # 2. Prepare a new, portable Q3D input directory.
    materials = {
        name: PdkMaterial(
            name, records[name]["material_kind"],
            records[name]["is_superconducting"],
            records[name]["aedt_library_name"],
        )
        for name in ("Al", "Si", "vacuum")
    }
    handoff = prepare_q3d_from_geometry(
        snapshot,
        aedt_version="2024.2",
        modeling="solid",
        output_dir=Path(".artifacts/simplepad-q3d") / uuid4().hex[:12],
        project_name="SimplePad",
        design_name="Capacitance",
        materials=materials,
        net_types={"pad": "Signal", "ground": "Signal"},
        physical_ground_nets=("ground",),
        run_control=MatrixRunControl(
            setup_name="Setup1", frequency_ghz=5.0,
            maximum_passes=10, convergence_percent=1.0,
        ),
        region_padding_um=(250.0, 250.0, 250.0, 250.0, 1000.0, 1000.0),
        resources=AedtResources(cores=4, ram_limit_percent=80),
    )
    print("Inputs:", handoff.run_dir)
    if args.handoff_only:
        return

    # 3. Run AEDT on this licensed host using the current Python environment.
    subprocess.run(
        [sys.executable, "-m", "scgsim.aedt.run",
         "--handoff", str(handoff.metadata_path), "--execute"],
        cwd=handoff.run_dir,
        check=True,
    )

    # 4. Read the returned matrix entries with their names and exported units.
    result = resolve_results(handoff.run_dir)
    for row in result.physics_results():
        print(row)


if __name__ == "__main__":
    main()
```

在上述研究專案資料夾執行：

```bash
uv run simplepad_q3d.py
```

每次執行都會在 `.artifacts/simplepad-q3d/` 建立新的結果資料夾。終端機列出的矩陣項目包含物理量、列名、欄名、數值與原生匯出單位；使用結果時，請一起保留這些欄位。

若要先檢視輸入，執行 `uv run simplepad_q3d.py --handoff-only`。此選項只準備輸入與可攜套件，不會開啟 AEDT；完整模式則在同一主機執行並讀回結果。

本例把兩個 Net 都設為 Q3D 的 `Signal`，讓兩者保留在矩陣中；`physical_ground_nets` 另外記錄接地的物理角色。模型尺寸、邊界選擇與結果解讀詳見[電容教學](docs/tutorials/capacitance.qmd)；各完整教學保留其頁面列出的環境與來源版本。

## 選擇下一個模擬目標

| 目標 | 從這裡開始 |
|---|---|
| 電容 | [Palace Electrostatic 或 AEDT Q3D](docs/tutorials/capacitance.qmd) |
| 共振與 EPR | [Palace 或 HFSS 模態](docs/tutorials/resonance.qmd) · [能量參與率](docs/tutorials/resonance-epr.qmd) |
| Driven S 參數 | [HFSS Modal 或 Terminal](docs/tutorials/scattering.qmd) |
| 傳輸線參數 | [AEDT Q2D](docs/tutorials/transmission-line.qmd) |
