| 項目 | Route B | Route C |
|:---|:---|:---|
| `verify_loaded` 是否通過 | True | True |
| 載入後與規劃的實體數（vertex／edge／face／solid） | 64 / 92 / 38 / 2 | 64 / 92 / 38 / 4 |
| physical group（規劃／載入） | 16 / 16 | 18 / 18 |
| vertex 最大偏移 (µm) | 0 | 0 |
| 邊界不符的數量（edge、face、solid） | 0, 0, 0 | 0, 0, 0 |
| face 面積最大相對誤差 | 1.46e-15 | 1.46e-15 |
| solid 體積最大相對誤差 | 1.32e-16 | 3.55e-16 |
| `apply_sidecar_embeds` 嵌入的 face／edge | 1 / 1 | 1 / 1 |
| 粗 mesh（`Mesh.MeshSizeMax` = 60 µm）的節點數 | 429 | 432 |
| `export_aedt_step` 的 object 名稱 | `VOL__VACUUM_REGION`、`VOL__substrate` | `VOL__VACUUM_REGION`、`VOL__ground_0000`、`VOL__pad`、`VOL__substrate` |
| 各 solid 的 guard face 數 | VACUUM_REGION 35、substrate 8 | VACUUM_REGION 35、ground_0000 14、pad 11、substrate 8 |
