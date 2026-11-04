## Band areas (computed by inset.py from the production Route A build, margins 0.1, 0.5, 2 um; areas of the faces of each band group)

| support surface | band | group | faces | area (um^2) | R_m = bands >= k (um^2) |
|---|---|---|---|---|---|
| pad top | B0 | `pad::band=0` | 2 | 39.1069 | 5421.5730 |
| pad top | B1 | `pad::band=1` | 2 | 156.2232 | 5382.4661 |
| pad top | B2 | `pad::band=2` | 2 | 583.2396 | 5226.2429 |
| pad top | B3 | `pad::band=3` | 1 | 4643.0033 | 4643.0033 |
| gap | B0 | `SA::substrate|vacuum::interface::band=0` | 3 | 93.0717 | 13478.4270 |
| gap | B1 | `SA::substrate|vacuum::interface::band=1` | 3 | 371.3770 | 13385.3553 |
| gap | B2 | `SA::substrate|vacuum::interface::band=2` | 3 | 1380.0089 | 13013.9783 |
| gap | B3 | `SA::substrate|vacuum::interface::band=3` | 2 | 11633.9694 | 11633.9694 |

PORT::p1: 7 faces, 300.0000 um^2 (the port footprint takes this area out of the gap bands)

## Band areas by family (cells carrying the band label, port cells included)

| family | band | area (um^2) |
|---|---|---|
| band_pad_top | B0 | 39.1069 |
| band_pad_top | B1 | 156.2232 |
| band_pad_top | B2 | 583.2396 |
| band_pad_top | B3 | 4643.0033 |
| band_gap | B0 | 95.0717 |
| band_gap | B1 | 379.3770 |
| band_gap | B2 | 1410.0089 |
| band_gap | B3 | 11893.9694 |

## KLayout cross-check (tests/sgb_klayout.py::comparison_rows on the Route A build)

| region | margin (um) | components (KLayout) | exact area (um^2) | KLayout area (um^2) | difference (um^2) | relative | oracle rounding bound (um^2) |
|---|---|---|---|---|---|---|---|
| pad top | 0 | 1 | 5421.5730 | 5421.5732 | -0.00017 | -3.1e-08 | 0.196 |
| pad top | 0.1 | 1 | 5382.4661 | 5382.4575 | +0.00867 | +1.6e-06 | 0.196 |
| pad top | 0.5 | 1 | 5226.2429 | 5226.2351 | +0.00779 | +1.5e-06 | 0.195 |
| pad top | 2 | 1 | 4643.0033 | 4643.0019 | +0.00142 | +3.1e-07 | 0.194 |
| gap | 0 | 2 | 13778.4270 | 13778.4268 | +0.00017 | +1.2e-08 | 0.476 |
| gap | 0.1 | 2 | 13683.3553 | 13683.3599 | -0.00466 | -3.4e-07 | 0.475 |
| gap | 0.5 | 2 | 13303.9783 | 13303.9840 | -0.00572 | -4.3e-07 | 0.474 |
| gap | 2 | 2 | 11893.9694 | 11893.9680 | +0.00136 | +1.1e-07 | 0.467 |

## Mesh check of the Route A band groups (production writer, Gmsh; no Palace is run)

mesh size 40 um, 1 um on the band curves; nodes: 13455, tetrahedra: 72356, duplicate nodes: 0, conformal: True

| band group | faces | triangles | tets on each side | all triangles shared by tets | centroids inside band |
|---|---|---|---|---|---|
| `SA::substrate|vacuum::interface::band=0` | 3 | 1857 | 1857 + 1857 | True | True |
| `SA::substrate|vacuum::interface::band=1` | 3 | 1904 | 1904 + 1904 | True | True |
| `SA::substrate|vacuum::interface::band=2` | 3 | 3692 | 3692 + 3692 | True | True |
| `SA::substrate|vacuum::interface::band=3` | 2 | 3684 | 3684 + 3684 | True | True |
| `pad::band=0` | 2 | 790 | 790 + 790 | True | True |
| `pad::band=1` | 2 | 807 | 807 + 807 | True | True |
| `pad::band=2` | 2 | 1595 | 1595 + 1595 | True | True |
| `pad::band=3` | 1 | 1659 | 1659 + 1659 | True | True |

## The same mesh check under every route

| route | mesh size (um) | band groups | nodes | tetrahedra | duplicate nodes | conformal | every band group passes |
|---|---|---|---|---|---|---|---|
| A | 40 | 8 | 13455 | 72356 | 0 | True | True |
| B | 60 | 12 | 13276 | 51151 | 0 | True | True |
| C | 60 | 12 | 13367 | 68945 | 0 | True | True |

## Mask against imprint (inset-mask-vs-imprint.json)

| h | triangles | straddling the contour | error, 1-point mask (median / max) | error, 3-point mask (median / max) | triangles (aligned) | straddling (aligned) | error (aligned) |
|---|---|---|---|---|---|---|---|
| 0.4 | 66 | 19 | 7.7e-03 / 1.9e-02 | 6.2e-02 / 6.8e-02 | 74 | 0 | 2.44e-02 |
| 0.2 | 246 | 32 | 2.9e-03 / 1.5e-02 | 6.9e-03 / 1.1e-02 | 274 | 0 | 6.23e-03 |
| 0.1 | 948 | 71 | 2.0e-03 / 5.5e-03 | 1.2e-03 / 1.8e-03 | 958 | 0 | 1.57e-03 |
| 0.05 | 3702 | 138 | 5.0e-04 / 1.2e-03 | 4.0e-04 / 7.4e-04 | 3808 | 0 | 3.92e-04 |
| 0.025 | 14792 | 276 | 2.5e-04 / 4.1e-04 | 1.2e-04 / 4.6e-04 | 15026 | 0 | 9.81e-05 |
| 0.0125 | 59322 | 552 | 4.9e-05 / 9.9e-05 | 6.4e-05 / 2.3e-04 | 59600 | 0 | 2.53e-05 |

## Route B (top and bottom faces banded)

groups: MA::pad::top::band=0, MA::pad::top::band=1, MA::pad::top::band=2, MA::pad::top::band=3, MS::pad::bottom::band=0, MS::pad::bottom::band=1, MS::pad::bottom::band=2, MS::pad::bottom::band=3
validation passed: True (31 checks)
counts: 66 faces, 280 edges, 321 vertices, 23 groups
unions NET::pad: ["MA::pad::sidewall", "MA::pad::top::band=0", "MA::pad::top::band=1", "MA::pad::top::band=2", "MA::pad::top::band=3", "MS::pad::bottom::band=0", "MS::pad::bottom::band=1", "MS::pad::bottom::band=2", "MS::pad::bottom::band=3"]
surface_epr_rows (Route A): {"MA::pad::sheet::band=0": "pad::band=0", "MA::pad::sheet::band=1": "pad::band=1", "MA::pad::sheet::band=2": "pad::band=2", "MA::pad::sheet::band=3": "pad::band=3", "MS::pad::sheet::band=0": "pad::band=0", "MS::pad::sheet::band=1": "pad::band=1", "MS::pad::sheet::band=2": "pad::band=2", "MS::pad::sheet::band=3": "pad::band=3", "SA::substrate|vacuum::interface::band=0": "SA::substrate|vacuum::interface::band=0", "SA::substrate|vacuum::interface::band=1": "SA::substrate|vacuum::interface::band=1", "SA::substrate|vacuum::interface::band=2": "SA::substrate|vacuum::interface::band=2", "SA::substrate|vacuum::interface::band=3": "SA::substrate|vacuum::interface::band=3"}

## Port invariance (Route A, with and without the lumped port)

family records equal: True
largest band-area difference: 5.96e-08 grid unit^2

## Topology change (margins 4, 9, 30 um on the pad top)

{"margins_um": [4.0, 9.0, 30.0], "band_faces": {"0": 2, "1": 1, "2": 3}, "band_face_areas_um2": {"0": [251.32741228718345, 1298.601465150257], "1": [1873.4914035101017], "2": [36.723199528808955, 250.59743126898456, 1710.8321125098535]}, "bands_present": [0, 1, 2], "klayout_components_of_R_m": {"0": 1, "4": 1, "9": 3, "21": 1, "23": 1, "30": 0}}

## Sagitta of a 128-gon disc, m (1 - cos(pi / 128)), in nm

| margin (um) | sagitta (nm) |
|---|---|
| 0.1 | 0.030 |
| 0.5 | 0.151 |
| 2.0 | 0.602 |
