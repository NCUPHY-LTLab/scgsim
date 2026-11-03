## Measured speed-up (outputs/scaling.json; speed-up = build time of the 1x1, 1-worker run / build time)

| N | T x T | workers | build (s) | speed-up | serial stages (s) | parallel stages, wall (s) | worker compute (s) | Amdahl parallel fraction | speed-up limit | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1x1 | 1 | 2.3 | 1.00 | 0.9 | 1.4 | 1.3 | 0.60 | 2.49 | 331 |
| 1000 | 4x4 | 1 | 2.6 | 0.90 | 0.9 | 1.6 | 1.5 | 0.59 | 2.42 | 147 |
| 1000 | 4x4 | 8 | 1.5 | 1.51 | 0.8 | 0.6 | 1.8 | 0.69 | 3.27 | 158 |
| 1000 | 4x4 | 16 | 1.9 | 1.23 | 1.0 | 0.8 | 2.0 | 0.68 | 3.12 | 158 |
| 1000 | 4x4 | 32 | 1.8 | 1.31 | 0.9 | 0.8 | 2.1 | 0.70 | 3.38 | 159 |
| 1000 | 8x8 | 1 | 2.5 | 0.95 | 0.8 | 1.5 | 1.4 | 0.60 | 2.51 | 148 |
| 1000 | 8x8 | 8 | 2.0 | 1.19 | 1.0 | 0.8 | 2.0 | 0.67 | 2.99 | 158 |
| 1000 | 8x8 | 16 | 1.8 | 1.32 | 0.8 | 0.8 | 2.3 | 0.74 | 3.85 | 158 |
| 1000 | 8x8 | 32 | 2.1 | 1.10 | 0.9 | 1.1 | 2.6 | 0.73 | 3.73 | 157 |
| 5000 | 1x1 | 1 | 10.2 | 1.00 | 4.1 | 5.6 | 5.6 | 0.58 | 2.36 | 1387 |
| 5000 | 4x4 | 1 | 11.0 | 0.92 | 4.1 | 6.5 | 6.0 | 0.56 | 2.30 | 540 |
| 5000 | 4x4 | 8 | 9.3 | 1.09 | 5.2 | 3.4 | 9.7 | 0.65 | 2.87 | 602 |
| 5000 | 4x4 | 16 | 10.8 | 0.94 | 6.2 | 3.8 | 11.0 | 0.64 | 2.79 | 596 |
| 5000 | 4x4 | 32 | 10.8 | 0.94 | 6.2 | 3.8 | 11.1 | 0.64 | 2.79 | 596 |
| 5000 | 8x8 | 1 | 11.7 | 0.87 | 4.2 | 7.1 | 6.5 | 0.58 | 2.39 | 544 |
| 5000 | 8x8 | 8 | 9.2 | 1.10 | 5.1 | 3.5 | 10.8 | 0.68 | 3.14 | 600 |
| 5000 | 8x8 | 16 | 10.9 | 0.93 | 6.2 | 3.9 | 13.4 | 0.68 | 3.16 | 595 |
| 5000 | 8x8 | 32 | 10.8 | 0.94 | 5.8 | 4.2 | 13.7 | 0.70 | 3.38 | 592 |
| 20000 | 1x1 | 1 | 52.1 | 1.00 | 21.9 | 28.0 | 27.9 | 0.56 | 2.27 | 5386 |
| 20000 | 4x4 | 1 | 56.6 | 0.92 | 22.3 | 32.2 | 27.6 | 0.51 | 2.03 | 2002 |
| 20000 | 4x4 | 8 | 53.9 | 0.97 | 32.0 | 18.6 | 44.7 | 0.58 | 2.40 | 2238 |
| 20000 | 4x4 | 16 | 43.3 | 1.20 | 24.8 | 15.7 | 35.1 | 0.58 | 2.41 | 2228 |
| 20000 | 4x4 | 32 | 44.6 | 1.17 | 25.2 | 16.6 | 36.7 | 0.59 | 2.45 | 2216 |
| 20000 | 8x8 | 1 | 56.6 | 0.92 | 22.0 | 32.6 | 27.8 | 0.51 | 2.04 | 2010 |
| 20000 | 8x8 | 8 | 54.2 | 0.96 | 30.5 | 20.3 | 48.2 | 0.61 | 2.58 | 2223 |
| 20000 | 8x8 | 16 | 59.8 | 0.87 | 34.9 | 21.4 | 61.7 | 0.64 | 2.77 | 2215 |
| 20000 | 8x8 | 32 | 66.4 | 0.78 | 35.4 | 27.2 | 81.4 | 0.70 | 3.30 | 2224 |

## Ranges quoted in the text

- largest speed-up of any run: 1.51x
- speed-up of runs with workers > 1: 0.78x to 1.51x
- Amdahl parallel fraction, one process (1x1, 1 worker): 0.56 to 0.60
- Amdahl parallel fraction, N >= 5000 with workers: 0.58 to 0.70
- speed-up limit, N >= 5000 with workers: 2.40x to 3.38x
- speed-up limit, one process: 2.27x to 2.49x
- same identity digest for every (T, workers) of one N: True
- every case passed validation: True
- Amdahl curve with serial share from N = 20000, one process: fraction 0.559, S(8) = 1.96x, limit 2.27x

## Stage times of the serial run (N = 20000, 1x1 tile, 1 worker)

| stage | kind | seconds |
|---|---|---|
| `noding_kernels` | parallel | 9.41 |
| `dissolution` | serial | 5.83 |
| `atomic_ids` | parallel | 5.50 |
| `assembly_faces` | serial | 4.50 |
| `shells` | parallel | 4.26 |
| `dcel_faces` | serial | 3.53 |
| `noding_driver` | serial | 3.47 |
| `labelling` | parallel | 3.12 |
| `validation_shells` | parallel | 3.00 |
| `wall_runs` | parallel | 2.69 |
| `validation_serial` | serial | 1.02 |
| `linking` | serial | 0.93 |
| `assembly_groups` | serial | 0.54 |
| `assembly_registry` | serial | 0.45 |
| `wall_prep` | serial | 0.37 |
| `contact_check` | serial | 0.34 |
| `shell_prep` | serial | 0.31 |
| `manifold_check` | serial | 0.30 |
| `arc_realisation` | serial | 0.17 |
| `volume_connectivity` | serial | 0.11 |
| `S2_levels` | serial | 0.04 |
| S10 emission (no meshing) | sequential | 80.23 |

serial stages total 21.9 s, parallel stages total 28.0 s, build wall 52.1 s

## End to end at N = 20000 (derivation + emission, one process)

- derivation 52.1 s + emission 80.2 s = 132.3 s; emission is 61% of it
- derivation at its speed-up limit (2.27x, 23.0 s): total 103.2 s, 1.28x end to end
- derivation at the best measured speed-up (1.20x): total 123.5 s, 1.07x end to end

## Worker compute against workers (N = 20000, 4x4 tiles)

| workers | worker compute (s) | serial stages (s) | parallel stages wall (s) |
|---|---|---|---|
| 1 | 27.6 | 22.3 | 32.2 |
| 8 | 44.7 | 32.0 | 18.6 |
| 16 | 35.1 | 24.8 | 15.7 |
| 32 | 36.7 | 25.2 | 16.6 |

## Tile statistics of the running example (z = 0 level, Route A with bands; computed here from the prototype runtime)

| tiles | hot pixels | pixels per tile | noded edges | edges whose ends lie in different tiles | x cuts (um) | y cuts (um) |
|---|---|---|---|---|---|---|
| 2x2 | 2918 | [1251, 1257, 295, 115] | 2932 | 56 | -0.0005 | -0.0005 |
| 3x3 | 2918 | [5, 167, 5, 0, 2576, 0, 5, 155, 5] | 2932 | 38 | -50.0005, 49.9995 | -50.0005, 49.9995 |
| 4x4 | 2918 | [1, 4, 4, 1, 0, 1246, 1252, 0, 0, 286, 106, 0, 1, 8, 8, 1] | 2932 | 58 | -75.0005, -0.0005, 74.9995 | -75.0005, -0.0005, 74.9995 |
