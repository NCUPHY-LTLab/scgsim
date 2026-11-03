## Determinism (identity-determinism.json)

| tiles | workers | PYTHONHASHSEED | pad_in_ground B with bands | random layout (60 features, seed 5, Route B) |
|---|---|---|---|---|
| 1x1 | 1 | 0 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 1x1 | 1 | 1 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 1x1 | 1 | 12345 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 2x2 | 1 | 0 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 2x2 | 1 | 1 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 2x2 | 1 | 12345 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 3x3 | 2 | 0 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 3x3 | 2 | 1 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 3x3 | 2 | 12345 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 4x4 | 4 | 0 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 4x4 | 4 | 1 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |
| 4x4 | 4 | 12345 | `build:22a139898b1f72d2` | `build:4208c3b044b51b94` |

distinct digests per model: 1 and 1

## Stability under an unrelated edit (identity-stability.json)

| kind | with port | without port | shared | only with port | only without port |
|---|---|---|---|---|---|
| vertices | 289 | 281 | 281 | 8 | 0 |
| edges | 250 | 238 | 234 | 16 | 4 |
| faces | 59 | 56 | 54 | 5 | 2 |
| volumes | 4 | 4 | 1 | 3 | 3 |

| group | members with port | members without port | identical members |
|---|---|---|---|
| `EXT::substrate` | 5 | 5 | True |
| `EXT::vacuum` | 5 | 5 | True |
| `MA::Ground` | 6 | 5 | False |
| `MA::pad` | 15 | 14 | False |
| `MA::pad::band=0` | 2 | 2 | True |
| `MA::pad::band=1` | 2 | 2 | True |
| `MA::pad::band=2` | 2 | 2 | True |
| `MA::pad::band=3` | 1 | 1 | True |
| `MS::Ground` | 1 | 1 | True |
| `MS::pad::band=0` | 2 | 2 | True |
| `MS::pad::band=1` | 2 | 2 | True |
| `MS::pad::band=2` | 2 | 2 | True |
| `MS::pad::band=3` | 1 | 1 | True |
| `PORT::p1` | 1 | None | False |
| `SA::substrate|vacuum` | 1 | 1 | True |
| `SA::substrate|vacuum::band=0` | 2 | 2 | True |
| `SA::substrate|vacuum::band=1` | 2 | 2 | True |
| `SA::substrate|vacuum::band=2` | 2 | 2 | True |
| `SA::substrate|vacuum::band=3` | 1 | 1 | True |
| `VOL::substrate` | 1 | 1 | True |
| `VOL::vacuum` | 1 | 1 | False |
