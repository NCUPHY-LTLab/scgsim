| stage | Kind | One process (s) | 4 × 4 tiles, 16 workers (s) | Speed-up |
|:---|:---|---:|---:|---:|
| `assembly_faces` | serial | 8.7 | 9.1 | 0.96× |
| `dissolution` | serial | 7.0 | 7.8 | 0.89× |
| `dcel_faces` | serial | 3.9 | 4.3 | 0.89× |
| `noding_driver` | serial | 3.0 | 3.6 | 0.85× |
| `noding_kernels` | parallel | 9.8 | 6.1 | 1.61× |
| `shells` | parallel | 6.0 | 5.2 | 1.17× |
| `atomic_ids` | parallel | 5.6 | 1.7 | 3.31× |
| `validation_shells` | parallel | 3.9 | 2.8 | 1.39× |
