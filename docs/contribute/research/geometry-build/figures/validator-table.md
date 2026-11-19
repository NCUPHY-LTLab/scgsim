| Check | Kind | What it checks | N = 1,000 (s) | N = 5,000 (s) | N = 20,000 (s) | Slope |
|:---|:---|:---|---:|---:|---:|---:|
| `planarity` | structural | each arrangement edge, with the noding kernel | 0.145 | 0.978 | 4.542 | 1.15 |
| `face_areas` | loaded model | one Gmsh mass call per face | 0.192 | 0.998 | 3.993 | 1.01 |
| `shells` | structural | the faces of each shell | 0.095 | 0.770 | 3.925 | 1.24 |
| `counts_and_vertex_placement` | loaded model | one Gmsh call per vertex | 0.059 | 0.938 | 3.408 | 1.35 |
| `boundaries` | loaded model | one getBoundary call per edge, face and solid | 0.152 | 0.775 | 3.078 | 1.00 |
| `volumes` | loaded model | one mass integral per solid | 0.115 | 0.634 | 2.501 | 1.03 |
| `physical_groups` | loaded model | one group query per entity | 0.061 | 0.349 | 1.440 | 1.06 |
| `groups` | structural | group membership of every face and volume | 0.016 | 0.128 | 0.662 | 1.25 |
| `edges_in_faces` | structural | the edges and vertical edges of every face | 0.022 | 0.132 | 0.638 | 1.13 |
| `contact_check` | structural | contact between different Nets | 0.011 | 0.072 | 0.350 | 1.14 |
| `manifold_check` | structural | non-manifold pinches | 0.010 | 0.065 | 0.283 | 1.12 |
| `atomic_ids` | structural | uniqueness of vertex, edge and face ids | 0.008 | 0.043 | 0.271 | 1.17 |
| `grid_and_lengths` | structural | edge lengths and vertex grid positions | 0.012 | 0.063 | 0.263 | 1.04 |
| `face_sides` | structural | the two sides of every face | 0.003 | 0.021 | 0.095 | 1.11 |
| `areas` | structural | area conservation per arrangement | 0.003 | 0.014 | 0.059 | 1.03 |
| `prism_areas` | structural | prism areas conserved on the cells | 0.003 | 0.013 | 0.059 | 1.00 |
| `cylinders` | structural | each curved wall is one cylinder | 0.001 | 0.006 | 0.031 | 1.21 |
| `bands` | structural | band areas are monotone | 0.000 | 0.001 | 0.004 | 0.99 |
| `arcs` | structural | arc edges do not intersect and stay within the deviation | 0.000 | 0.001 | 0.002 | 0.89 |
