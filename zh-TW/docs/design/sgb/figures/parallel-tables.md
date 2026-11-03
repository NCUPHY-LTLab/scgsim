## Measured speed-up (outputs/scaling.json; speed-up = build time of the 1x1, 1-worker run / build time)

| N | T x T | workers | build (s) | speed-up | serial stages (s) | parallel stages, wall (s) | worker compute (s) | Amdahl parallel fraction | speed-up limit | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1x1 | 1 | 1.4 | 1.00 | 0.5 | 0.8 | 0.8 | 0.61 | 2.54 | 324 |
| 1000 | 4x4 | 1 | 1.6 | 0.89 | 0.5 | 1.0 | 0.9 | 0.60 | 2.49 | 144 |
| 1000 | 4x4 | 8 | 1.3 | 1.08 | 0.6 | 0.6 | 1.3 | 0.66 | 2.95 | 152 |
| 1000 | 4x4 | 16 | 1.3 | 1.06 | 0.7 | 0.6 | 1.4 | 0.68 | 3.17 | 151 |
| 1000 | 4x4 | 32 | 1.4 | 1.04 | 0.6 | 0.6 | 1.5 | 0.71 | 3.42 | 152 |
| 1000 | 8x8 | 1 | 1.6 | 0.86 | 0.5 | 1.1 | 1.0 | 0.61 | 2.57 | 145 |
| 1000 | 8x8 | 8 | 1.4 | 1.01 | 0.7 | 0.6 | 1.4 | 0.67 | 3.00 | 150 |
| 1000 | 8x8 | 16 | 1.4 | 1.04 | 0.7 | 0.6 | 1.7 | 0.72 | 3.58 | 151 |
| 1000 | 8x8 | 32 | 1.6 | 0.91 | 0.6 | 0.8 | 2.0 | 0.76 | 4.14 | 150 |
| 5000 | 1x1 | 1 | 9.3 | 1.00 | 3.7 | 5.2 | 5.1 | 0.58 | 2.37 | 1358 |
| 5000 | 4x4 | 1 | 10.2 | 0.92 | 3.9 | 5.9 | 5.4 | 0.55 | 2.22 | 520 |
| 5000 | 4x4 | 8 | 7.7 | 1.21 | 4.4 | 2.8 | 6.7 | 0.61 | 2.54 | 575 |
| 5000 | 4x4 | 16 | 7.8 | 1.19 | 4.4 | 2.9 | 7.5 | 0.63 | 2.71 | 578 |
| 5000 | 4x4 | 32 | 7.7 | 1.22 | 4.3 | 2.7 | 7.5 | 0.63 | 2.72 | 575 |
| 5000 | 8x8 | 1 | 10.5 | 0.89 | 3.8 | 6.3 | 5.8 | 0.57 | 2.34 | 526 |
| 5000 | 8x8 | 8 | 7.9 | 1.18 | 4.5 | 2.8 | 7.5 | 0.63 | 2.68 | 574 |
| 5000 | 8x8 | 16 | 8.0 | 1.17 | 4.4 | 2.9 | 8.9 | 0.67 | 3.01 | 572 |
| 5000 | 8x8 | 32 | 8.2 | 1.14 | 4.4 | 3.2 | 10.0 | 0.70 | 3.29 | 578 |
| 20000 | 1x1 | 1 | 45.2 | 1.00 | 18.9 | 24.3 | 24.2 | 0.56 | 2.27 | 5272 |
| 20000 | 4x4 | 1 | 48.5 | 0.93 | 19.0 | 27.5 | 24.0 | 0.52 | 2.06 | 1929 |
| 20000 | 4x4 | 8 | 38.2 | 1.18 | 21.8 | 13.8 | 30.1 | 0.58 | 2.38 | 2128 |
| 20000 | 4x4 | 16 | 38.6 | 1.17 | 21.9 | 14.1 | 34.0 | 0.61 | 2.56 | 2143 |
| 20000 | 4x4 | 32 | 38.4 | 1.18 | 21.8 | 13.9 | 32.9 | 0.60 | 2.51 | 2138 |
| 20000 | 8x8 | 1 | 51.2 | 0.88 | 19.6 | 29.6 | 25.9 | 0.53 | 2.11 | 1919 |
| 20000 | 8x8 | 8 | 42.0 | 1.07 | 24.9 | 14.2 | 32.4 | 0.56 | 2.30 | 2134 |
| 20000 | 8x8 | 16 | 43.2 | 1.05 | 25.4 | 14.8 | 39.5 | 0.61 | 2.56 | 2136 |
| 20000 | 8x8 | 32 | 42.5 | 1.06 | 23.0 | 16.6 | 41.4 | 0.64 | 2.80 | 2129 |

## Ranges quoted in the text

- largest speed-up of any run: 1.22x
- speed-up of runs with workers > 1: 0.91x to 1.22x
- Amdahl parallel fraction, one process (1x1, 1 worker): 0.56 to 0.61
- Amdahl parallel fraction, N >= 5000 with workers: 0.56 to 0.70
- speed-up limit, N >= 5000 with workers: 2.30x to 3.29x
- speed-up limit, one process: 2.27x to 2.54x
- same identity digest for every (T, workers) of one N: True
- every case passed validation: True
- Amdahl curve with serial share from N = 20000, one process: fraction 0.560, S(8) = 1.96x, limit 2.27x

## Stage times of the serial run (N = 20000, 1x1 tile, 1 worker)

| stage | kind | seconds |
|---|---|---|
| `noding_kernels` | parallel | 7.40 |
| `dissolution` | serial | 5.24 |
| `atomic_ids` | parallel | 4.87 |
| `shells` | parallel | 4.30 |
| `assembly_faces` | serial | 4.23 |
| `dcel_faces` | serial | 3.11 |
| `validation_shells` | parallel | 2.99 |
| `noding_driver` | serial | 2.38 |
| `labelling` | parallel | 2.38 |
| `wall_runs` | parallel | 2.34 |
| `validation_serial` | serial | 0.98 |
| `linking` | serial | 0.77 |
| `assembly_groups` | serial | 0.55 |
| `assembly_registry` | serial | 0.42 |
| `wall_prep` | serial | 0.34 |
| `shell_prep` | serial | 0.33 |
| `contact_check` | serial | 0.29 |
| `arc_realisation` | serial | 0.14 |
| `volume_connectivity` | serial | 0.10 |
| `S2_levels` | serial | 0.04 |
| S10 emission (no meshing) | sequential | 74.22 |

serial stages total 18.9 s, parallel stages total 24.3 s, build wall 45.2 s

## End to end at N = 20000 (derivation + emission, one process)

- derivation 45.2 s + emission 74.2 s = 119.4 s; emission is 62% of it
- derivation at its speed-up limit (2.27x, 19.9 s): total 94.1 s, 1.27x end to end
- derivation at the best measured speed-up (1.18x): total 112.4 s, 1.06x end to end

## Worker compute against workers (N = 20000, 4x4 tiles)

| workers | worker compute (s) | serial stages (s) | parallel stages wall (s) |
|---|---|---|---|
| 1 | 24.0 | 19.0 | 27.5 |
| 8 | 30.1 | 21.8 | 13.8 |
| 16 | 34.0 | 21.9 | 14.1 |
| 32 | 32.9 | 21.8 | 13.9 |

## Tile statistics of the running example (z = 0 level, Route A with bands; computed here from the prototype runtime)

| tiles | hot pixels | pixels per tile | noded edges | edges whose ends lie in different tiles | x cuts (um) | y cuts (um) |
|---|---|---|---|---|---|---|
| 2x2 | 2918 | [1251, 1257, 295, 115] | 2932 | 56 | -0.0005 | -0.0005 |
| 3x3 | 2918 | [5, 167, 5, 0, 2576, 0, 5, 155, 5] | 2932 | 38 | -50.0005, 49.9995 | -50.0005, 49.9995 |
| 4x4 | 2918 | [1, 4, 4, 1, 0, 1246, 1252, 0, 0, 286, 106, 0, 1, 8, 8, 1] | 2932 | 58 | -75.0005, -0.0005, 74.9995 | -75.0005, -0.0005, 74.9995 |
