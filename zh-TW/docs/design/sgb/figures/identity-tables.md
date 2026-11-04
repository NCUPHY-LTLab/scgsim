## Determinism (identity-determinism.json)

| tiles | workers | PYTHONHASHSEED | pad_in_ground B with bands | random layout (60 features, seed 5, Route B) |
|---|---|---|---|---|
| 1x1 | 1 | 0 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 1x1 | 1 | 1 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 1x1 | 1 | 12345 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 2x2 | 1 | 0 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 2x2 | 1 | 1 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 2x2 | 1 | 12345 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 3x3 | 2 | 0 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 3x3 | 2 | 1 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 3x3 | 2 | 12345 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 4x4 | 4 | 0 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 4x4 | 4 | 1 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |
| 4x4 | 4 | 12345 | `model:29ac282c3d8eb0bd` | `model:b00423575e10e9af` |

distinct digests per model: 1 and 1

## Stability under an unrelated edit (identity-stability.json)

| kind | with port | without port | shared | only with port | only without port |
|---|---|---|---|---|---|
| vertices | 321 | 305 | 305 | 16 | 0 |
| edges | 274 | 250 | 242 | 32 | 8 |
| faces | 60 | 59 | 48 | 12 | 11 |
| volumes | 4 | 4 | 4 | 0 | 0 |

groups in only one of the two builds: PORT::p1

| group | members with port | members without port | identical members |
|---|---|---|---|
| `EXT::substrate::bottom` | 1 | 1 | True |
| `EXT::substrate::sidewall` | 4 | 4 | True |
| `EXT::vacuum::sidewall` | 4 | 4 | True |
| `EXT::vacuum::top` | 1 | 1 | True |
| `MA::ground::sidewall` | 4 | 4 | False |
| `MA::ground::top` | 1 | 1 | True |
| `MA::pad::sidewall` | 14 | 14 | False |
| `MA::pad::top::band=0` | 2 | 2 | True |
| `MA::pad::top::band=1` | 2 | 2 | True |
| `MA::pad::top::band=2` | 2 | 2 | True |
| `MA::pad::top::band=3` | 1 | 1 | True |
| `MS::ground::bottom` | 1 | 1 | False |
| `MS::pad::bottom::band=0` | 2 | 2 | False |
| `MS::pad::bottom::band=1` | 2 | 2 | True |
| `MS::pad::bottom::band=2` | 2 | 2 | True |
| `MS::pad::bottom::band=3` | 1 | 1 | True |
| `PORT::p1` | 1 | None | False |
| `SA::substrate|vacuum::interface::band=0` | 3 | 3 | False |
| `SA::substrate|vacuum::interface::band=1` | 3 | 3 | False |
| `SA::substrate|vacuum::interface::band=2` | 3 | 3 | False |
| `SA::substrate|vacuum::interface::band=3` | 2 | 2 | False |
| `substrate` | 1 | 1 | True |
| `vacuum` | 1 | 1 | True |
