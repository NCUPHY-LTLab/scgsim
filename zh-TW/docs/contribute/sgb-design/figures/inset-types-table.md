| 輸入 | 請求 | 展開成的 face（family id） | 略過的 face |
|---|---|---|---|
| pad-in-ground，Route B | <code>InterfaceSurfaces("MA")</code> | <code>ma::ground_0000::top</code><br><code>ma::pad::top</code> | 無 |
|  | <code>InterfaceSurfaces("MS")</code> | <code>ms::ground_0000::bottom</code><br><code>ms::pad::bottom</code> | 無 |
|  | <code>InterfaceSurfaces("SA")</code> | <code>sa::substrate&#124;VACUUM_REGION::z=0.0</code> | <code>substrate</code>–<code>VACUUM_REGION</code>（z = -10 µm） |
| thin-film pair，Route A | <code>InterfaceSurfaces("MA")</code> | <code>ma::chip/left::sheet</code><br><code>ma::chip/right::sheet</code> | 無 |
|  | <code>InterfaceSurfaces("MS")</code> | <code>ma::chip/left::sheet</code><br><code>ma::chip/right::sheet</code> | 無 |
|  | <code>InterfaceSurfaces("SA")</code> | <code>sa::substrate&#124;upper_air::z=0.0</code> | 無 |
| 整個 die 被 ground 蓋住，Route B | <code>InterfaceSurfaces("MA")</code> | 無 | <code>ground</code> 的 <code>top</code> |
