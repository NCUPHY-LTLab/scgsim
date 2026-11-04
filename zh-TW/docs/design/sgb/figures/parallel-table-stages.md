| Stage | 種類 | 1×1 tile、1 worker (s) | 4×4 tile、16 worker (s) | 倍數 |
|:---|:---|---:|---:|---:|
| `noding_kernels` | 平行 | 9.78 | 6.09 | 1.61x |
| `assembly_faces` | 序列 | 8.66 | 9.06 | 0.96x |
| `dissolution` | 序列 | 6.97 | 7.83 | 0.89x |
| `shells` | 平行 | 6.02 | 5.16 | 1.17x |
| `atomic_ids` | 平行 | 5.60 | 1.69 | 3.31x |
| `validation_shells` | 平行 | 3.90 | 2.81 | 1.39x |
| `dcel_faces` | 序列 | 3.85 | 4.33 | 0.89x |
| `labelling` | 平行 | 3.13 | 0.79 | 3.95x |
| `noding_driver` | 序列 | 3.04 | 3.58 | 0.85x |
| `validation_serial` | 序列 | 1.93 | 2.13 | 0.90x |
| `routes` | 序列 | 1.44 | 1.49 | 0.97x |
| `linking` | 序列 | 0.97 | 1.19 | 0.81x |
| `wall_runs` | 平行 | 0.75 | 0.64 | 1.18x |
| `assembly_registry` | 序列 | 0.73 | 0.71 | 1.02x |
| `shell_prep` | 序列 | 0.49 | 0.49 | 1.00x |
| `wall_prep` | 序列 | 0.44 | 0.45 | 0.98x |
| `contact_check` | 序列 | 0.36 | 0.39 | 0.92x |
| `arc_realisation` | 序列 | 0.30 | 0.33 | 0.91x |
| `manifold_check` | 序列 | 0.29 | 0.33 | 0.86x |
| `volume_connectivity` | 序列 | 0.15 | 0.16 | 0.93x |
| `levels` | 序列 | 0.04 | 0.05 | 0.92x |
| **序列 stage 合計** |  | **29.7** | **32.5** | **0.91x** |
| **平行 stage 合計** |  | **29.2** | **17.2** | **1.70x** |
| **build 的 wall time** |  | **61.6** | **53.2** | **1.16x** |
