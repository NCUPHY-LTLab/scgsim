| 檢查 | 做什麼 | N = 1,000 (F = 14,215) | N = 5,000 (F = 71,969) | N = 20,000 (F = 285,871) | 每個 face 的 µs（首項到末項） | 對 F 的 slope |
|:---|:---|---:|---:|---:|---:|---:|
| `planarity` | S9：各 arrangement 的 edge，用 noding kernel（逐 tile 平行） | 0.145 s | 0.978 s | 4.542 s | 10.17 -> 15.89 | 1.15 |
| `face_areas` | 載入側：每個 face 一次 gmsh mass 呼叫，O(F) | 0.192 s | 0.998 s | 3.993 s | 13.49 -> 13.97 | 1.01 |
| `shells` | S9：每個 shell 的 face（逐 shell 平行） | 0.095 s | 0.770 s | 3.925 s | 6.70 -> 13.73 | 1.24 |
| `counts_and_vertex_placement` | 載入側：每個 vertex 一次 gmsh 呼叫，O(V) | 0.059 s | 0.938 s | 3.408 s | 4.15 -> 11.92 | 1.35 |
| `boundaries` | 載入側：每個 edge、face、solid 一次 getBoundary，O(E + F) | 0.152 s | 0.775 s | 3.078 s | 10.67 -> 10.77 | 1.00 |
| `volumes` | 載入側：每個 solid 一次 mass 積分，O(solid) | 0.115 s | 0.634 s | 2.501 s | 8.12 -> 8.75 | 1.03 |
| `physical_groups` | 載入側：每個 entity 問一次 group，O(entity) | 0.061 s | 0.349 s | 1.440 s | 4.26 -> 5.04 | 1.06 |
| `groups` | S9：每個 face 與 volume 的 group 歸屬，O(F) | 0.016 s | 0.128 s | 0.662 s | 1.10 -> 2.32 | 1.25 |
| `edges_in_faces` | S9：每個 face 的 edge 與 vertical edge，O(E + F) | 0.022 s | 0.132 s | 0.638 s | 1.53 -> 2.23 | 1.13 |
| `contact_check` | S5：different-Net 接觸，每個 vertex 與 face，O(V + F) | 0.011 s | 0.072 s | 0.350 s | 0.80 -> 1.23 | 1.14 |
| `manifold_check` | S5：non-manifold pinch，O(E) | 0.010 s | 0.065 s | 0.283 s | 0.70 -> 0.99 | 1.12 |
| `atomic_ids` | S9：各 arrangement 的 vertex、edge、face id 的唯一性，O(V + E + F) | 0.008 s | 0.043 s | 0.271 s | 0.57 -> 0.95 | 1.17 |
| `grid_and_lengths` | S9：每條 edge 的長度與 vertex 的 grid 位置，O(E) | 0.012 s | 0.063 s | 0.263 s | 0.82 -> 0.92 | 1.04 |
| `face_sides` | S9：每個 face 的兩側，O(F) | 0.003 s | 0.021 s | 0.095 s | 0.24 -> 0.33 | 1.11 |
| `areas` | S9：各 arrangement 的面積守恆，O(E) | 0.003 s | 0.014 s | 0.059 s | 0.19 -> 0.21 | 1.03 |
| `prism_areas` | S9：prism 面積在 cell 上守恆，O(E) | 0.003 s | 0.013 s | 0.059 s | 0.20 -> 0.20 | 1.00 |
| `cylinders` | S9：curved wall 是單一 cylinder，O(wall) | 0.001 s | 0.006 s | 0.031 s | 0.06 -> 0.11 | 1.21 |
| `bands` | S9：band 面積單調，O(含 band 的 level 的 face) | 0.000 s | 0.001 s | 0.004 s | 0.02 -> 0.01 | 0.99 |
| `arcs` | S9：arc edge 互不相交與偏差，O(arc edge) | 0.000 s | 0.001 s | 0.002 s | 0.01 -> 0.01 | 0.89 |
