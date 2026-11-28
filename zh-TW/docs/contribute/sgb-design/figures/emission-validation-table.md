| 載入後的檢查 | pad_in_ground_A | pad_in_ground_B | airbridge_B |
|:---|:---|:---|:---|
| Gmsh 中的元素數（vertex / edge / face / solid），與規劃相符 | 34 / 44 / 15 / 2 | 56 / 84 / 35 / 2 | 44 / 74 / 33 / 2 |
| 以 `getBoundary` 檢查的邊界不一致數（edge / face / solid） | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| vertex 與規劃位置的最大偏差 (µm) | 0 | 0 | 0 |
| 從 XAO 載入的 physical group 數（BREP 不含 group） | 10 of 10 | 14 of 14 | 18 of 18 |
| volume 對規劃的最大相對誤差（XAO） | 0 | 5.4e-13 | 2.1e-16 |
| 載入後 XAO 的粗略 3-D mesh | 655 nodes，2626 tets，0 個重複 node，conformal：是 | 642 nodes，2105 tets，0 個重複 node，conformal：是 | 558 nodes，1709 tets，0 個重複 node，conformal：是 |
| face 面積對規劃的最大相對誤差 | 1.3e-12 | 3.1e-12 | 2.5e-16 |
| 與 Gmsh-API oracle 的 entity 數與 group 成員相同 | 是 | 是 | 是 |
| 與 oracle 的 face 面積最大相對差（平面 face；curved wall） | 1.3e-12；無 curved wall | 1.3e-12；1.8e-08（5 面） | 6.8e-16；無 curved wall |
| 解析 volume（封閉形式） | 未計算 | substrate 9000000，解析值 9000000 µm³；vacuum 8984755.69，解析值 8984755.69 µm³ | 未計算 |
| `verify_loaded` 是否通過 | 是 | 是 | 是 |
