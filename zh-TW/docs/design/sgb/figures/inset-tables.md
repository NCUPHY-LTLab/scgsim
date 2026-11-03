## Band areas (computed by inset.py from the Route A build, margins 0.1, 0.5, 2 um)

| support region | band | faces | area (um^2) | R_m = bands >= k (um^2) |
|---|---|---|---|---|
| pad top | B0 | 2 | 39.1069 | 5421.5730 |
| pad top | B1 | 2 | 156.2232 | 5382.4661 |
| pad top | B2 | 2 | 583.2396 | 5226.2429 |
| pad top | B3 | 1 | 4643.0033 | 4643.0033 |
| gap ring | B0 | 4 | 90.0766 | 13577.3650 |
| gap ring | B1 | 4 | 360.0248 | 13487.2885 |
| gap ring | B2 | 4 | 1346.3917 | 13127.2637 |
| gap ring | B3 | 2 | 11780.8720 | 11780.8720 |

## KLayout cross-check (examples/run_inset_bands.py, outputs/inset/klayout_comparison.json)

| region | margin (um) | exact area (um^2) | KLayout area (um^2) | difference (um^2) | relative | oracle rounding bound (um^2) |
|---|---|---|---|---|---|---|
| band_pad_top | 0 | 5421.5730 | 5421.5732 | -0.00017 | -3.1e-08 | 0.196 |
| band_pad_top | 0.1 | 5382.4661 | 5382.4575 | +0.00867 | +1.6e-06 | 0.196 |
| band_pad_top | 0.5 | 5226.2429 | 5226.2351 | +0.00779 | +1.5e-06 | 0.195 |
| band_pad_top | 2 | 4643.0033 | 4643.0019 | +0.00142 | +3.1e-07 | 0.194 |
| band_gap | 0 | 13577.3650 | 13577.3652 | -0.00014 | -9.9e-09 | 0.450 |
| band_gap | 0.1 | 13487.2885 | 13487.2992 | -0.01075 | -8.0e-07 | 0.450 |
| band_gap | 0.5 | 13127.2637 | 13127.2746 | -0.01094 | -8.3e-07 | 0.450 |
| band_gap | 2 | 11780.8720 | 11780.8722 | -0.00016 | -1.3e-08 | 0.448 |

## Palace-facing check (outputs/inset/palace_check.json)

| band group | faces | triangles | tets on each side | all triangles shared by tets | centroids inside band |
|---|---|---|---|---|---|
| `SA::substrate|vacuum::band=0` | 2 | 1753 | 1753 + 1753 | True | True |
| `SA::substrate|vacuum::band=1` | 2 | 1804 | 1804 + 1804 | True | True |
| `SA::substrate|vacuum::band=2` | 2 | 3522 | 3522 + 3522 | True | True |
| `SA::substrate|vacuum::band=3` | 1 | 3574 | 3574 + 3574 | True | True |
| `SHEET::pad::band=0` | 2 | 790 | 790 + 790 | True | True |
| `SHEET::pad::band=1` | 2 | 807 | 807 + 807 | True | True |
| `SHEET::pad::band=2` | 2 | 1591 | 1591 + 1591 | True | True |
| `SHEET::pad::band=3` | 1 | 1607 | 1607 + 1607 | True | True |

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

groups: MA::pad::band=0, MA::pad::band=1, MA::pad::band=2, MA::pad::band=3, MS::pad::band=0, MS::pad::band=1, MS::pad::band=2, MS::pad::band=3
validation passed: True
counts: 59 faces, 250 edges, 289 vertices, 21 groups
unions: ["MA::pad", "MA::pad::band=0", "MA::pad::band=1", "MA::pad::band=2", "MA::pad::band=3", "MS::pad::band=0", "MS::pad::band=1", "MS::pad::band=2", "MS::pad::band=3"]
surface_epr_rows (Route A): {"SA::substrate|vacuum::band=0": "SA::substrate|vacuum::band=0", "SA::substrate|vacuum::band=1": "SA::substrate|vacuum::band=1", "SA::substrate|vacuum::band=2": "SA::substrate|vacuum::band=2", "SA::substrate|vacuum::band=3": "SA::substrate|vacuum::band=3", "MA::pad::band=0": "SHEET::pad::band=0", "MS::pad::band=0": "SHEET::pad::band=0", "MA::pad::band=1": "SHEET::pad::band=1", "MS::pad::band=1": "SHEET::pad::band=1", "MA::pad::band=2": "SHEET::pad::band=2", "MS::pad::band=2": "SHEET::pad::band=2", "MA::pad::band=3": "SHEET::pad::band=3", "MS::pad::band=3": "SHEET::pad::band=3"}

## Port invariance (outputs/inset/port_invariance.json)

{"families_identical": true, "max_band_area_difference_grid2": 5.960464477539063e-08}

## Topology change (outputs/inset/topology_change.json)

{"margins_um": [4, 9, 30], "faces_per_band_um2": {"0": [1298.6015, 251.3274], "1": [1873.4919], "2": [250.5969, 36.7232, 1710.8321]}, "note": "B_3 (>= 30 um) is empty: R_30 vanished because the pad top has no point 30 um from its boundary (R_23 still exists, R_30 does not). R_9 splits into 3 components (KLayout) and B_2 into the 3 matching faces.", "validation_passed": true, "components_of_R_m_klayout": {"0.0": 1, "4.0": 1, "9.0": 3, "21.0": 1, "23.0": 1}, "wide_margins_um": [4, 9, 21, 23]}

## Sagitta of a 128-gon disc, m (1 - cos(pi / 128)), in nm

| margin (um) | sagitta (nm) |
|---|---|
| 0.1 | 0.030 |
| 0.5 | 0.151 |
| 2.0 | 0.602 |
