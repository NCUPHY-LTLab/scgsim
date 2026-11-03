| 檢查 | 複雜度 | N = 1,000 (F = 12,761) | N = 5,000 (F = 63,664) | N = 20,000 (F = 255,052) | 每個 face 的 µs（首項到末項） | 對 F 的 slope |
|:---|:---|---:|---:|---:|---:|---:|
| `face_areas` | O(F) gmsh mass calls | 0.342 s | 1.852 s | 8.379 s | 26.81 -> 32.85 | 1.07 |
| `planarity` | O(E) with a hot-pixel grid | 0.217 s | 1.272 s | 5.486 s | 16.99 -> 21.51 | 1.08 |
| `edge_boundaries` | O(E) gmsh calls | 0.166 s | 0.893 s | 4.079 s | 13.0 -> 15.99 | 1.07 |
| `volumes` | O(solids) mass integrals | 0.158 s | 0.858 s | 4.049 s | 12.39 -> 15.87 | 1.08 |
| `shells` | O(F) per shell (parallel per tile) | 0.135 s | 1.097 s | 3.592 s | 10.54 -> 14.08 | 1.10 |
| `face_boundaries` | O(F) gmsh calls | 0.275 s | 0.719 s | 2.976 s | 21.53 -> 11.67 | 0.79 |
| `counts_and_vertex_placement` | O(V) gmsh calls | 0.113 s | 0.732 s | 2.822 s | 8.86 -> 11.07 | 1.08 |
| `physical_groups` | O(entities) (was O(groups x entities): quadratic, fixed) | 0.080 s | 0.564 s | 2.113 s | 6.29 -> 8.29 | 1.09 |
| `writer_plan_checks` | O(E + pcurves) | 0.061 s | 0.418 s | 1.408 s | 4.78 -> 5.52 | 1.05 |
| `contact_check` | O(V + F) | 0.015 s | 0.120 s | 0.482 s | 1.2 -> 1.89 | 1.16 |
| `manifold_check` | O(E) | 0.017 s | 0.124 s | 0.409 s | 1.29 -> 1.6 | 1.08 |
| `edges_in_faces` | O(E + F) | 0.028 s | 0.122 s | 0.348 s | 2.18 -> 1.37 | 0.85 |
| `grid_and_lengths` | O(E) | 0.010 s | 0.066 s | 0.275 s | 0.81 -> 1.08 | 1.10 |
| `groups` | O(F) | 0.023 s | 0.097 s | 0.252 s | 1.78 -> 0.99 | 0.81 |
| `solid_boundaries` | O(solids x faces) | 0.009 s | 0.072 s | 0.232 s | 0.68 -> 0.91 | 1.10 |
| `areas` | O(E) | 0.003 s | 0.022 s | 0.090 s | 0.26 -> 0.35 | 1.11 |
| `face_sides` | O(F) | 0.005 s | 0.040 s | 0.076 s | 0.42 -> 0.3 | 0.90 |
| `prism_areas` | O(E) | 0.003 s | 0.016 s | 0.066 s | 0.22 -> 0.26 | 1.05 |
| `arcs` | O(arc edges) with sampled crossings | 0.000 s | 0.002 s | 0.006 s | 0.03 -> 0.03 | 1.00 |
| `bands` | O(F of the banded levels) | 0.000 s | 0.000 s | 0.000 s | 0.0 -> 0.0 | n/a |
