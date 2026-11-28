| 檢查 | 種類 | 檢查什麼 | N = 1,000（s） | N = 5,000（s） | N = 20,000（s） | 斜率 |
|:---|:---|:---|---:|---:|---:|---:|
| `planarity` | 結構 | 各 arrangement 的 edge，用 noding kernel | 0.145 | 0.978 | 4.542 | 1.15 |
| `face_areas` | 載入後 | 每個 face 一次 Gmsh mass 呼叫 | 0.192 | 0.998 | 3.993 | 1.01 |
| `shells` | 結構 | 每個 shell 的 face | 0.095 | 0.770 | 3.925 | 1.24 |
| `counts_and_vertex_placement` | 載入後 | 每個 vertex 一次 Gmsh 呼叫 | 0.059 | 0.938 | 3.408 | 1.35 |
| `boundaries` | 載入後 | 每個 edge、face、solid 一次 getBoundary | 0.152 | 0.775 | 3.078 | 1.00 |
| `volumes` | 載入後 | 每個 solid 一次 mass 積分 | 0.115 | 0.634 | 2.501 | 1.03 |
| `physical_groups` | 載入後 | 每個 entity 查詢一次 group | 0.061 | 0.349 | 1.440 | 1.06 |
| `groups` | 結構 | 每個 face 與 volume 的 group 歸屬 | 0.016 | 0.128 | 0.662 | 1.25 |
| `edges_in_faces` | 結構 | 每個 face 的 edge 與 vertical edge | 0.022 | 0.132 | 0.638 | 1.13 |
| `contact_check` | 結構 | 不同 Net 的接觸 | 0.011 | 0.072 | 0.350 | 1.14 |
| `manifold_check` | 結構 | non-manifold pinch | 0.010 | 0.065 | 0.283 | 1.12 |
| `atomic_ids` | 結構 | vertex、edge、face id 的唯一性 | 0.008 | 0.043 | 0.271 | 1.17 |
| `grid_and_lengths` | 結構 | edge 長度與 vertex 的 grid 位置 | 0.012 | 0.063 | 0.263 | 1.04 |
| `face_sides` | 結構 | 每個 face 的兩側 | 0.003 | 0.021 | 0.095 | 1.11 |
| `areas` | 結構 | 各 arrangement 的面積守恆 | 0.003 | 0.014 | 0.059 | 1.03 |
| `prism_areas` | 結構 | prism 面積在 cell 上守恆 | 0.003 | 0.013 | 0.059 | 1.00 |
| `cylinders` | 結構 | 每面 curved wall 是單一 cylinder | 0.001 | 0.006 | 0.031 | 1.21 |
| `bands` | 結構 | band 面積單調 | 0.000 | 0.001 | 0.004 | 0.99 |
| `arcs` | 結構 | arc edge 互不相交且在容許偏差內 | 0.000 | 0.001 | 0.002 | 0.89 |
