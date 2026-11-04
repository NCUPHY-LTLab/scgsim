| 群組（`faces[*].group`） | 面數 | `interface_type` | `entities` | `face_kind` | `z_um` 範圍 | `adjacent_domains` |
|:---|---:|:---|:---|:---|:---|:---|
| <code>MA::ground_0000::sidewall</code> | 11 | `MA` | `ground_0000` | `sidewall` | 0 到 0.2 | `VACUUM_REGION`（vacuum） |
| <code>MA::ground_0000::top</code> | 1 | `MA` | `ground_0000` | `top` | 0.2 | `VACUUM_REGION`（vacuum） |
| <code>MA::pad::sidewall</code> | 6 | `MA` | `pad` | `sidewall` | 0 到 0.2 | `VACUUM_REGION`（vacuum） |
| <code>MA::pad::top::band=0</code> | 1 | `MA` | `pad` | `top` | 0.2 | `VACUUM_REGION`（vacuum） |
| <code>MA::pad::top::band=1</code> | 1 | `MA` | `pad` | `top` | 0.2 | `VACUUM_REGION`（vacuum） |
| <code>MA::pad::top::band=2</code> | 1 | `MA` | `pad` | `top` | 0.2 | `VACUUM_REGION`（vacuum） |
| <code>MS::ground_0000::bottom</code> | 1 | `MS` | `ground_0000` | `bottom` | 0 | `substrate`（dielectric） |
| <code>MS::pad::bottom</code> | 1 | `MS` | `pad` | `bottom` | 0 | `substrate`（dielectric） |
| <code>SA::substrate&#124;VACUUM_REGION::interface</code> | 2 | `SA` | `VACUUM_REGION`、`substrate` | `interface` | -10 到 0 | `VACUUM_REGION`（vacuum）、`substrate`（dielectric） |
| <code>SA::substrate&#124;VACUUM_REGION::sidewall</code> | 4 | `SA` | `VACUUM_REGION`、`substrate` | `sidewall` | -10 到 0 | `VACUUM_REGION`（vacuum）、`substrate`（dielectric） |
