## Determinism (identity-determinism.json)

| PYTHONHASHSEED | pad_in_ground B with bands | random layout (60 features, seed 5, Route B) |
|---|---|---|
| 0 | `model:ea7f893fddc6733c` | `model:c10a91b4661ff323` |
| 1 | `model:ea7f893fddc6733c` | `model:c10a91b4661ff323` |
| 12345 | `model:ea7f893fddc6733c` | `model:c10a91b4661ff323` |

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
| `EXT::substrate__bottom` | 1 | 1 | True |
| `EXT::substrate__sidewall` | 4 | 4 | True |
| `EXT::vacuum__sidewall` | 4 | 4 | True |
| `EXT::vacuum__top` | 1 | 1 | True |
| `MA::ground__sidewall` | 4 | 4 | False |
| `MA::ground__top` | 1 | 1 | True |
| `MA::pad__sidewall` | 14 | 14 | False |
| `MA::pad__top::band=0` | 2 | 2 | True |
| `MA::pad__top::band=1` | 2 | 2 | True |
| `MA::pad__top::band=2` | 2 | 2 | True |
| `MA::pad__top::band=3` | 1 | 1 | True |
| `MS::ground__bottom` | 1 | 1 | False |
| `MS::pad__bottom::band=0` | 2 | 2 | False |
| `MS::pad__bottom::band=1` | 2 | 2 | True |
| `MS::pad__bottom::band=2` | 2 | 2 | True |
| `MS::pad__bottom::band=3` | 1 | 1 | True |
| `PORT::p1` | 1 | None | False |
| `SA::substrate|vacuum__interface::band=0` | 3 | 3 | False |
| `SA::substrate|vacuum__interface::band=1` | 3 | 3 | False |
| `SA::substrate|vacuum__interface::band=2` | 3 | 3 | False |
| `SA::substrate|vacuum__interface::band=3` | 2 | 2 | False |
| `substrate` | 1 | 1 | True |
| `vacuum` | 1 | 1 | True |
