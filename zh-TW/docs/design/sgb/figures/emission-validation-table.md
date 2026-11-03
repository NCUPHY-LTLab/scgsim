| 載入後的檢查 | pad_in_ground_A | pad_in_ground_B | airbridge_B |
|:---|:---|:---|:---|
| Gmsh 中的元素數（vertex / edge / face / solid），與規劃相符 | 34 / 44 / 15 / 2 | 60 / 90 / 37 / 2 | 44 / 72 / 31 / 2 |
| 以 `getBoundary` 檢查的邊界不一致數（edge / face / solid） | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| vertex 與規劃位置的最大偏差 (µm) | 0 | 0 | 0 |
| 從 XAO 載入的 physical group 數（BREP 不含 group） | 8 of 8 | 10 of 10 | 7 of 7 |
| volume 對規劃的最大相對誤差（XAO） | 0 | 0 | 2.1e-16 |
| 載入後 XAO 的粗略 3-D mesh | 651 nodes，2597 tets，0 個重複 node，conformal：是 | 643 nodes，2136 tets，0 個重複 node，conformal：是 | 561 nodes，1736 tets，0 個重複 node，conformal：是 |
| face 面積對規劃的最大相對誤差 | 4.8e-15 | 1.3e-12 | 2.5e-16 |
| sidecar 與 gmsh-API oracle 的 sidecar 相同（groups、unions、embeds、各維度 ids） | 是 | 是 | 是 |
| 解析 volume（封閉形式） | 未計算 | substrate 9000000，解析值 9000000 µm³；vacuum 8984755.69，解析值 8984755.69 µm³ | 未計算 |
