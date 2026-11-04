| 項目 | Route B | Route C |
|:---|:---|:---|
| `verify_loaded` 是否通過 | True | True |
| 載入後與規劃的實體數（vertex／edge／face／solid） | 60 / 86 / 36 / 2 | 60 / 86 / 36 / 4 |
| physical group（規劃／載入） | 16 / 16 | 18 / 18 |
| vertex 最大偏移 (µm) | 0 | 0 |
| 邊界不符的數量（edge、face、solid） | 0, 0, 0 | 0, 0, 0 |
| face 面積最大相對誤差 | 1.46e-15 | 1.46e-15 |
| solid 體積最大相對誤差 | 1.32e-16 | 1.78e-16 |
| `apply_sidecar_embeds` 嵌入的 face／edge | 0 / 1 | 0 / 1 |
| 粗 mesh（`Mesh.MeshSizeMax` = 60 µm）的節點數 | 420 | 427 |
| `export_aedt_step` 的 object 名稱 | `VACUUM_REGION`、`substrate` | `VACUUM_REGION`、`ground_0000`、`pad`、`substrate` |
| 各 solid 的 guard face 數 | VACUUM_REGION 34、substrate 9 | VACUUM_REGION 34、ground_0000 13、pad 10、substrate 9 |
