| 步驟（N = 20,000，一個 process） | 秒 |
|:---|---:|
| derivation（`build_model_from_catalogue`） | 61.6 |
| 寫出 BREP 與 XAO（`plan`、`write_brep`、`write_xao`，1 worker） | 13.8 |
| sliver report 與 validation report | 49.7 |
| sidecar | 19.6 |
| Gmsh 載入 XAO | 49.7 |
| `verify_loaded` | 15.3 |
| **合計** | **209.8** |
