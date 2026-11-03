| 注入的缺陷 | BRepCheck_Analyzer | BOPAlgo_ArgumentAnalyzer | ShapeAnalysis_Shell | plan 面積／volume 比對 | 全部檢查耗時 (ms) |
|:---|:---|:---|:---|:---|---:|
| `reversed_inner_wire` | 抓到（face:BadOrientationOfSubshape，shell:BadOrientationOfSubshape） | 抓到（SelfIntersect） | 抓到 | 抓到 | 29.3 |
| `flipped_shared_face_in_one_shell` | 抓到 | 未抓到 | 抓到 | 抓到 | 21.7 |
| `vertex_moved_1e-4_um_off_its_curve` | 抓到 | 抓到（TooSmallEdge） | 未抓到 | 未抓到 | 68.7 |
| `missing_face_open_shell` | 抓到 | 未抓到 | 抓到 | 抓到 | 37.8 |
| `wire_with_a_gap` | 抓到（face:UnorientableShape，shell:NotClosed） | 抓到（SelfIntersect） | 抓到 | 未抓到 | 26.9 |
