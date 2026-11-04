## Measured speed-up (figures/why-scaling-measured.json; speed-up = build time of the 1x1, 1-worker run / build time)

| N | T x T | workers | build (s) | speed-up | serial stages (s) | parallel stages, wall (s) | worker compute (s) | Amdahl parallel fraction | speed-up limit | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1x1 | 1 | 2.3 | 1.00 | 1.1 | 1.2 | 1.2 | 0.52 | 2.10 | 135 |
| 1000 | 4x4 | 1 | 2.3 | 1.02 | 1.0 | 1.2 | 1.1 | 0.52 | 2.07 | 143 |
| 1000 | 4x4 | 8 | 2.0 | 1.15 | 1.2 | 0.7 | 1.4 | 0.55 | 2.23 | 148 |
| 1000 | 4x4 | 16 | 2.0 | 1.19 | 1.2 | 0.7 | 1.6 | 0.59 | 2.42 | 148 |
| 1000 | 4x4 | 32 | 1.9 | 1.22 | 1.1 | 0.7 | 1.7 | 0.60 | 2.52 | 150 |
| 1000 | 8x8 | 1 | 2.3 | 1.03 | 0.9 | 1.3 | 1.2 | 0.53 | 2.14 | 136 |
| 1000 | 8x8 | 8 | 1.9 | 1.21 | 1.1 | 0.6 | 1.6 | 0.58 | 2.41 | 144 |
| 1000 | 8x8 | 16 | 2.0 | 1.15 | 1.2 | 0.7 | 1.9 | 0.63 | 2.68 | 145 |
| 1000 | 8x8 | 32 | 2.2 | 1.06 | 1.2 | 0.8 | 2.1 | 0.63 | 2.72 | 146 |
| 5000 | 1x1 | 1 | 14.5 | 1.00 | 6.9 | 6.9 | 6.8 | 0.50 | 1.99 | 534 |
| 5000 | 4x4 | 1 | 15.0 | 0.96 | 6.6 | 7.7 | 6.8 | 0.48 | 1.91 | 536 |
| 5000 | 4x4 | 8 | 11.9 | 1.22 | 7.5 | 3.6 | 7.8 | 0.51 | 2.05 | 590 |
| 5000 | 4x4 | 16 | 11.7 | 1.24 | 7.2 | 3.6 | 8.6 | 0.54 | 2.19 | 592 |
| 5000 | 4x4 | 32 | 11.5 | 1.25 | 7.2 | 3.5 | 8.6 | 0.54 | 2.19 | 597 |
| 5000 | 8x8 | 1 | 14.9 | 0.97 | 6.4 | 7.8 | 7.0 | 0.49 | 1.97 | 533 |
| 5000 | 8x8 | 8 | 11.5 | 1.26 | 7.1 | 3.5 | 8.5 | 0.55 | 2.20 | 590 |
| 5000 | 8x8 | 16 | 11.5 | 1.26 | 7.2 | 3.5 | 10.0 | 0.58 | 2.40 | 595 |
| 5000 | 8x8 | 32 | 11.9 | 1.22 | 7.2 | 3.8 | 11.4 | 0.61 | 2.59 | 594 |
| 20000 | 1x1 | 1 | 61.6 | 1.00 | 29.7 | 29.2 | 29.1 | 0.49 | 1.98 | 2000 |
| 20000 | 4x4 | 1 | 66.8 | 0.92 | 30.4 | 33.5 | 29.3 | 0.46 | 1.85 | 1979 |
| 20000 | 4x4 | 8 | 53.6 | 1.15 | 32.7 | 17.3 | 34.4 | 0.51 | 2.05 | 2220 |
| 20000 | 4x4 | 16 | 53.2 | 1.16 | 32.5 | 17.2 | 36.6 | 0.53 | 2.13 | 2219 |
| 20000 | 4x4 | 32 | 53.1 | 1.16 | 32.4 | 17.1 | 37.0 | 0.53 | 2.14 | 2219 |
| 20000 | 8x8 | 1 | 69.1 | 0.89 | 30.9 | 35.2 | 30.9 | 0.47 | 1.88 | 1975 |
| 20000 | 8x8 | 8 | 52.7 | 1.17 | 32.3 | 16.7 | 36.2 | 0.53 | 2.12 | 2207 |
| 20000 | 8x8 | 16 | 53.0 | 1.16 | 32.4 | 17.0 | 41.1 | 0.56 | 2.27 | 2220 |
| 20000 | 8x8 | 32 | 54.2 | 1.14 | 32.0 | 18.5 | 43.7 | 0.58 | 2.36 | 2233 |

## Ranges quoted in the text

- largest speed-up of any run: 1.26x
- smallest speed-up of any run: 0.89x
- speed-up of runs with workers > 1: 1.06x to 1.26x
- speed-up of runs with workers > 1 at N = 20000: 1.14x to 1.17x
- speed-up of tiled runs with one worker: 0.89x to 1.03x
- Amdahl parallel fraction, one process (1x1, 1 worker): 0.49 to 0.52
- Amdahl parallel fraction, N >= 5000 with workers: 0.51 to 0.61
- speed-up limit, N >= 5000 with workers: 2.05x to 2.59x
- speed-up limit, one process: 1.98x to 2.10x
- same identity digest for every (T, workers) of one N: True
- every case passed validation: True
- Amdahl curve with serial share from N = 20000, one process: fraction 0.494, S(8) = 1.76x, limit 1.98x

## Stage times of the serial run (N = 20000, 1x1 tile, 1 worker) and the steps after it

| stage | kind | seconds |
|---|---|---|
| `noding_kernels` | parallel | 9.78 |
| `assembly_faces` | serial | 8.66 |
| `dissolution` | serial | 6.97 |
| `shells` | parallel | 6.02 |
| `atomic_ids` | parallel | 5.60 |
| `validation_shells` | parallel | 3.90 |
| `dcel_faces` | serial | 3.85 |
| `labelling` | parallel | 3.13 |
| `noding_driver` | serial | 3.04 |
| `validation_serial` | serial | 1.93 |
| `routes` | serial | 1.44 |
| `linking` | serial | 0.97 |
| `wall_runs` | parallel | 0.75 |
| `assembly_registry` | serial | 0.73 |
| `shell_prep` | serial | 0.49 |
| `wall_prep` | serial | 0.44 |
| `contact_check` | serial | 0.36 |
| `arc_realisation` | serial | 0.30 |
| `manifold_check` | serial | 0.29 |
| `volume_connectivity` | serial | 0.15 |
| `levels` | serial | 0.04 |
| plan + BREP + XAO files | after the derivation | 13.80 |
| sliver and validation reports | after the derivation | 49.74 |
| sidecar | after the derivation | 19.64 |
| Gmsh load of the XAO | after the derivation | 49.67 |
| verify_loaded | after the derivation | 15.27 |

serial stages total 29.7 s, parallel stages total 29.2 s, sum of stages 58.8 s, build wall 61.6 s

## End to end at N = 20000 (one process: derivation, the files, the reports, the sidecar, Gmsh load, verify_loaded)

- derivation 61.6 s, steps after it 148.1 s, total 209.8 s; the derivation is 29% of it
- derivation at its Amdahl limit (1.98x, 31.2 s): total 179.3 s, 1.17x end to end
- derivation at the best measured speed-up (1.17x, 8x8 tiles, 8 workers): total 200.8 s, 1.04x end to end
- only the derivation and the CAD files: 75.4 s; with the best speed-up 66.5 s, 1.13x

## Stage by stage at N = 20000: one process against 4x4 tiles with 16 workers

| stage | kind | 1x1, 1 worker (s) | 4x4, 16 workers (s) | ratio |
|---|---|---|---|---|
| `noding_kernels` | parallel | 9.78 | 6.09 | 1.61x |
| `assembly_faces` | serial | 8.66 | 9.06 | 0.96x |
| `dissolution` | serial | 6.97 | 7.83 | 0.89x |
| `shells` | parallel | 6.02 | 5.16 | 1.17x |
| `atomic_ids` | parallel | 5.60 | 1.69 | 3.31x |
| `validation_shells` | parallel | 3.90 | 2.81 | 1.39x |
| `dcel_faces` | serial | 3.85 | 4.33 | 0.89x |
| `labelling` | parallel | 3.13 | 0.79 | 3.95x |
| `noding_driver` | serial | 3.04 | 3.58 | 0.85x |
| `validation_serial` | serial | 1.93 | 2.13 | 0.90x |
| `routes` | serial | 1.44 | 1.49 | 0.97x |
| `linking` | serial | 0.97 | 1.19 | 0.81x |
| `wall_runs` | parallel | 0.75 | 0.64 | 1.18x |
| `assembly_registry` | serial | 0.73 | 0.71 | 1.02x |
| `shell_prep` | serial | 0.49 | 0.49 | 1.00x |
| `wall_prep` | serial | 0.44 | 0.45 | 0.98x |
| `contact_check` | serial | 0.36 | 0.39 | 0.92x |
| `arc_realisation` | serial | 0.30 | 0.33 | 0.91x |
| `manifold_check` | serial | 0.29 | 0.33 | 0.86x |
| `volume_connectivity` | serial | 0.15 | 0.16 | 0.93x |
| `levels` | serial | 0.04 | 0.05 | 0.92x |

## Worker compute against workers (N = 20000, 4x4 tiles)

| workers | worker compute (s) | serial stages (s) | parallel stages wall (s) | build (s) |
|---|---|---|---|---|
| 1 | 29.3 | 30.4 | 33.5 | 66.8 |
| 8 | 34.4 | 32.7 | 17.3 | 53.6 |
| 16 | 36.6 | 32.5 | 17.2 | 53.2 |
| 32 | 37.0 | 32.4 | 17.1 | 53.1 |

## Peak memory of the parent process (MB) by tiling and workers

| N | 1x1, 1 worker | 4x4, 1 worker | 4x4, 8 | 4x4, 16 | 4x4, 32 | 8x8, 1 worker | 8x8, 8 | 8x8, 16 | 8x8, 32 |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 135 | 143 | 148 | 148 | 150 | 136 | 144 | 145 | 146 |
| 5000 | 534 | 536 | 590 | 592 | 597 | 533 | 590 | 595 | 594 |
| 20000 | 2000 | 1979 | 2220 | 2219 | 2219 | 1975 | 2207 | 2220 | 2233 |

## Tile statistics of the running example (z = 0 level, Route A with bands; computed here with the production runtime)

| tiles | hot pixels | pixels per tile | noded edges | edges whose ends lie in different tiles | of them arc witness chords | median length (um) | x cuts (um) | y cuts (um) |
|---|---|---|---|---|---|---|---|---|
| 2x2 | 4050 | [1530, 2110, 295, 115] | 4064 | 62 | 14 | 47.9 | -0.0005 | -0.0005 |
| 3x3 | 4050 | [5, 167, 5, 0, 3708, 0, 5, 155, 5] | 4064 | 38 | 6 | 54.9 | -50.0005, 49.9995 | -50.0005, 49.9995 |
| 4x4 | 4050 | [1, 4, 4, 1, 0, 1525, 2105, 0, 0, 286, 106, 0, 1, 8, 8, 1] | 4064 | 64 | 14 | 47.3 | -75.0005, -0.0005, 74.9995 | -75.0005, -0.0005, 74.9995 |
