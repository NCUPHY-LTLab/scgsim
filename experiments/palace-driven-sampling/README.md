# Palace Driven sampling, hanger resonator

`run_study.py` runs the sweep-strategy study of [issue 307](https://github.com/NCUPHY-LTLab/scgsim/issues/307)
on its second geometry, a quarter-wave hanger resonator beside a feedline, through the public `EigenmodeSim` and
`DrivenSim` APIs and stock Palace 0.18.1. An Eigenmode run gives the resonator's f0 and Q; Driven runs that
differ in one setting each give the S21 notch, which `fit` fits with the hanger model of Probst et al. (2015).
It records numbers and judges none: every agreement tolerance is the Human's.

## The case

`hanger_layout.py` builds the layout from public OrcaPen SC PDK cells; the Human accepted it on the issue.

| Part | Value |
|---|---|
| Feedline | `cpw_6_10_6` (signal 10 µm, gaps 6 µm), 800 µm between the launcher necks, along y = 0 |
| Launchers | PDK `launcher`: pad 150 × 150 µm, taper 150 µm, 85 µm of gap behind and beside the pad |
| Resonator | PDK `resonator()` at its defaults: 4000 µm long, `cpw_6_10_6`, a 200 µm coupling straight at y = 27 µm (a 5 µm ground strip between the two gaps), shorted at the coupling straight's start, open at the meander's end |
| Nets | `signal`: the feedline and launchers; `ground`: the ground plane with the resonator |
| Port sheets | 85 × 150 µm behind each pad, landing 10 nm into the pad and the ground; 50 Ω `LumpedPort`s to `ground`, left first (port 1) |
| Die, stack | 1870 × 2058 µm, 150 µm of ground beyond every ground-mask edge; PDK single-die stack (Si εr 11.9, tan δ 2.6e-7, 500 µm; Al 0.1 µm) |
| Vacuum | padding (0, 0, 1000) µm |
| Surface EPR | the PDK's `LiteratureCentral` MA, MS and SA presets, no bands: 7 `Dielectric` rows |

Every run is on Route B at FEM order 2 with no mesh refinement, 8 MPI processes, on the mesh `mesh()` writes.

## Run it

Use an environment with the OrcaPen SC PDK revision CI's `pdk-tests` job pins and the SCGSim wheel of this
checkout, installed as that job installs them. Then, for each run, in the order of the table below:

```bash
export PALACE_EXECUTABLE=/path/to/palace        # the stock Palace 0.18.1 wrapper script
export PALACE_MPI_BIN=/path/to/mpi/bin          # the folder holding the mpirun Palace was built with
python run_study.py prepare  STUDY eigenmode
python run_study.py execute  STUDY eigenmode
python run_study.py readback STUDY eigenmode
# ... the same three stages for eigenmode-linear-1e-9, eigenmode-linear-1e-9-eigen-1e-9, adaptive, add-to-prom, tolerance-1e-5, tolerance-1e-5-max-40,
# no-surface-epr, full-solves, full-solves-linear-1e-9 and uniform-2500 (uniform-25000 is defined too)
python run_study.py fit STUDY
```

`prepare` runs `simulation_drc`, `mesh()`, `write_config()` and `prepare_handoff(profile="direct-local")` into
`STUDY/RUN/run` and writes `STUDY/RUN/prepare.json`. The Driven windows are placed from the Eigenmode run's
`report.json`, so `eigenmode` is read back first. `execute` runs the handoff's script; `readback` resolves the run
with `resolve_palace_result` and writes `STUDY/RUN/report.json` (and, for a Driven run, `hanger.s2p` and
`s21.json`); `fit` writes `STUDY/study.json` and prints the two tables. `STUDY` holds the meshes and Palace
output, several GB; keep it out of the repository. Run one Palace job at a time: Open MPI binds each job's ranks
from the first core, so two jobs started side by side share the same cores.

## The runs

Each Driven run varies one setting of `adaptive`. f0 and Q are the first Eigenmode run's resonator mode,
f0 = 7.356647804 GHz and Q = 13,947.6, so f0/Q = 527.4 kHz (the Eigenmode runs with a tighter linear solve give
Q = 11,141.7, so the window spans ±16 of their linewidths): the dense window runs from 7.346098839 to
7.367196768 GHz, 10.55 kHz apart, and the full solves from 7.35506546 to 7.35823015 GHz, about 395 kHz apart.

| Run | What it solves |
|---|---|
| `eigenmode` | `EigenmodeSim`, 4 modes from 6.5 GHz, the two ports as 50 Ω loads; every mode saved as a ParaView field |
| `eigenmode-linear-1e-9` | `eigenmode` with `set_numerical(tolerance=1e-9)`: the linear solves inside the eigensolver to 1e-9 |
| `eigenmode-linear-1e-9-eigen-1e-9` | `eigenmode-linear-1e-9` with `set_eigenmode(..., tolerance=1e-9)` as well |
| `adaptive` | `[Linear(4e9, 9e9, points=501), Linear(f0 − 20·f0/Q, f0 + 20·f0/Q, points=2001)]`, `adaptive_tolerance=1e-3`: 2,502 frequencies |
| `add-to-prom` | `adaptive`, with `{"Type": "Point", "Freq": [f0], "AddToPROM": true}` added to the written `config.json`'s `Solver.Driven.Samples` before the handoff: a full solve at f0 that SCGSim does not expose |
| `tolerance-1e-5` | `adaptive` with `adaptive_tolerance=1e-5` |
| `tolerance-1e-5-max-40` | `tolerance-1e-5` with `adaptive_max_samples=40`: `tolerance-1e-5` reached Palace's default cap of 20 |
| `no-surface-epr` | `adaptive` on a plan without `set_surface_epr` |
| `full-solves` | `[Points([...])]`, 9 frequencies of the window's own grid 0.75·f0/Q apart (±3 f0/Q), solved in full |
| `full-solves-linear-1e-9` | `full-solves` with `set_numerical(tolerance=1e-9)`: the linear solver to 1e-9 instead of SCGSim's 1e-6 |
| `uniform-2500` | `[Linear(4e9, 9e9, points=2500)]`, `adaptive_tolerance=1e-3`, 2.0 MHz apart |

The issue's uniform 25,000-point sweep was not run. `adaptive`'s measured cost predicts 1,190 s to build the
model plus 25,000 × (0.548 + 0.555) s online, 28,765 s (8.0 h), past the study's limit of about 7 h; so
`uniform-2500` was run instead; its own cost scales to 28,690 s (below).

The resonator mode is the Eigenmode mode with the largest share of its electric energy above y = 200 µm, where
only the meander is: the energy density `U_e` of each mode's ParaView field, integrated cell by cell.

## Results

All runs are on stock Palace 0.18.1 (git tag `0dc74cd`) with 8 MPI processes, one job at a time, on one mesh:
`mesh()` wrote the same `palace.msh`, byte for byte, in every run folder (521,337 tetrahedra; 3,677,860
unknowns at order 2). Every run's `Solver.Linear` is GMRES to a relative tolerance of 1e-6 with at most 400
iterations, SCGSim's default, unless the run says otherwise. Times and memory are Palace's own: the offline time
and each excitation's elapsed times from its log (printed to 3 significant digits), the total and the peak memory
per node (`PeakNodeMemoryMegabytes`) from `palace.json`.

### Runs

| Run | Settings | Tetrahedra | Unknowns | Samples per port | Offline (s) | Online per frequency per port (s) | Total (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|---|
| eigenmode | Eigenmode, 4 modes from 6.5 GHz | 521,337 | 3,677,860 | - | - | - | 1431.3 | 29,475 |
| eigenmode-linear-1e-9 | Eigenmode, 4 modes from 6.5 GHz, linear tol 1e-09 | 521,337 | 3,677,860 | - | - | - | 2074.0 | 29,459 |
| eigenmode-linear-1e-9-eigen-1e-9 | Eigenmode, 4 modes from 6.5 GHz, linear tol 1e-09, eigen tol 1e-09 | 521,337 | 3,677,860 | - | - | - | 2649.0 | 29,463 |
| adaptive | 501 band + 2001 window, adaptive 0.001 | 521,337 | 3,677,860 | 9, 4 | 1190.0 | 0.548, 0.555 | 3947.6 | 35,738 |
| add-to-prom | 501 band + 2001 window, adaptive 0.001, AddToPROM at f0 | 521,337 | 3,677,860 | 5, 5 | 921.0 | 0.527, 0.539 | 3587.8 | 34,996 |
| tolerance-1e-5 | 501 band + 2001 window, adaptive 1e-05 | 521,337 | 3,677,860 | 20 (cap; error 0.000198), 20 (cap; error 0.000371) | 3710.0 | 0.739, 0.747 | 7428.4 | 40,512 |
| tolerance-1e-5-max-40 | 501 band + 2001 window, adaptive 1e-05, max samples 40 | 521,337 | 3,677,860 | 32, 15 | 4370.0 | 0.791, 0.8 | 8351.7 | 41,625 |
| no-surface-epr | 501 band + 2001 window, adaptive 0.001, no Surface EPR | 521,337 | 3,677,860 | 9, 4 | 1190.0 | 0.528, 0.532 | 3842.0 | 35,739 |
| full-solves | 9 full solves | 521,337 | 3,677,860 | - | - | 86.7, 88 | 1586.0 | 33,325 |
| full-solves-linear-1e-9 | 9 full solves, linear tol 1e-09 | 521,337 | 3,677,860 | - | - | 119, 117 | 2136.7 | 34,233 |
| uniform-2500 | 2500 uniform, adaptive 0.001 | 521,337 | 3,677,860 | 9, 4 | 1190.0 | 0.544, 0.556 | 3940.4 | 35,796 |

Where the adaptive sweeps placed their full solves, in GHz as Palace's log prints them (4 digits):

| Run | Port 1 | Port 2 |
|---|---|---|
| `adaptive`, `no-surface-epr` | 4, 9, 6.555, 8.044, 7.355, 4.883, 7.357, 4.337, 8.654 | 4, 9, 6.555, 8.045 |
| `add-to-prom` | 4, 9, 7.357 (f0), 6.554, 5.077 | 4, 9, 7.357 (f0), 6.554, 5.077 |
| `tolerance-1e-5` | 4, 9, 6.555, 8.044, 7.355, 4.883, 7.357, 4.337, 8.654, 7.458, 8.391, 6.053, 7.162, 5.220, 7.693, 5.837, 7.403, 7.630, 7.325, 7.379 | 4, 9, 6.555, 8.045, 7.355, 4.883, 7.357, 4.337, 8.654, 7.158, 7.606, 7.322, 7.236, 5.807, 5.623, 7.359, 8.863, 7.807, 7.339, 7.345 |
| `uniform-2500` | as `adaptive` | as `adaptive` |
| `tolerance-1e-5-max-40` | as `tolerance-1e-5`, then 8.937, 5.090, 7.345, 7.330, 7.401, 6.944, 5.294, 7.814, 7.156, 7.357, 5.481, 4.564 (32) | as `tolerance-1e-5`'s first 15 (15) |

The two ports share one reduced model, so port 2 starts from port 1's solves.

### The Eigenmode runs

| Run | f0 (GHz) | Q = Re f / (2 Im f) | Q_ext[1] | Q_ext[2] | Q_ext together | kappa_ext[1], [2] (GHz) | Linear solves | GMRES iterations (per solve) | Total (s) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| eigenmode | 7.356647804 | 13947.6 | 22548.5 | 22484.5 | 11258.2 | -3.2626e-04, +3.2719e-04 | 28 | 406 ([11, 28]) | 1431.3 | 29,475 |
| eigenmode-linear-1e-9 | 7.356663931 | 11141.7 | 22555.5 | 22495.7 | 11262.8 | -3.2616e-04, +3.2703e-04 | 28 | 754 ([23, 50]) | 2074.0 | 29,459 |
| eigenmode-linear-1e-9-eigen-1e-9 | 7.356663931 | 11141.7 | 22555.5 | 22495.7 | 11262.8 | -3.2616e-04, +3.2703e-04 | 37 | 983 ([23, 50]) | 2649.0 | 29,463 |

Mode 1 is the resonator in every run: 99.75 % of its electric energy is above y = 200 µm (modes 2 to 4: 95.7,
69 to 71 and 36 to 45 %). With the linear solves inside the eigensolver at 1e-9, mode 1 moves to
f0 = 7.356663931 GHz and Q = 11,141.7: 20 Hz and 0.008 % from the notch's f0 and Q_l below, against 16.1 kHz and
25 % for the first run. Its ports' `Q_ext` together, 11,262.8, are 0.009 % from the notch's abs(Q_c), and
1/Q − 1/Q_ext[1] − 1/Q_ext[2] gives Q_i = 1.036e6, the notch's. Tightening the eigensolver's own tolerance to
1e-9 as well changes mode 1 in none of its printed digits (it takes 37 linear solves instead of 28). In all three
runs `kappa_ext[1]` is negative and `kappa_ext[2]` positive: the Eigenmode config writes port 1's `Direction` as
+x and port 2's as −x (attributes 11 and 12), where the Driven config writes port 1 as −x and port 2 as +x; the
`Q_ext` values are positive either way.

### The notch

| Run | Points fitted | rms residual | f0 (GHz) | Q_l | abs(Q_c) | Q_i | f0 - f0_eig (Hz) | Q_l / Q_eig - 1 | max abs(S21 - S21 of full-solves) | max abs(S21 - S21 of full-solves-linear-1e-9) | max abs(S21 - S21 of adaptive) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.537e-03 | 2.437e-06 | - |
| add-to-prom | 2003 | 1.75e-06 | 7.356663907 | 11140.7 | 11261.8 | 1.035e+06 | 1.61e+04 | -2.012e-01 | 4.535e-03 | 1.306e-05 | 1.430e-05 |
| tolerance-1e-5 (port-S.csv) | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.540e-03 | 2.535e-06 | 3.076e-06 |
| tolerance-1e-5-max-40 | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.540e-03 | 2.530e-06 | 3.174e-06 |
| no-surface-epr | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.537e-03 | 2.437e-06 | 1.604e-09 |
| full-solves | 9 | 1.12e-03 | 7.356663554 | 11056.4 | 11228.2 | 7.219e+05 | 1.575e+04 | -2.073e-01 | - | 4.538e-03 | 4.537e-03 |
| full-solves-linear-1e-9 | 9 | 1.30e-06 | 7.356663906 | 11140.8 | 11261.8 | 1.036e+06 | 1.61e+04 | -2.012e-01 | 4.538e-03 | - | 2.437e-06 |
| uniform-2500 | 10 | 9.88e-07 | 7.356663916 | 11140.8 | 11261.9 | 1.036e+06 | 1.611e+04 | -2.012e-01 | - | - | 7.855e-15 |

`f0`, `Q_l`, `abs(Q_c)` and `Q_i` are the hanger fit (`notch_fit.py`) of S21 on the dense window's points
(the 9 points of a full-solve run), with the line's delay taken from the band's phase away from the resonance,
14.64 ps (the full-solve runs use `adaptive`'s). `rms residual` is the fit's, in units of S21. The last columns
are the largest `abs(S21 − S21_reference)` over the frequencies a run shares with the reference; a dash, none
shared (`uniform-2500` shares only 4 and 9 GHz with `adaptive`). `f0_eig` and `Q_eig` are the first Eigenmode
run's, at SCGSim's linear tolerance of 1e-6; the Eigenmode runs above give the mode at a tighter one.

### The full solves and the linear solver

`full-solves`, solved with SCGSim's linear tolerance of 1e-6, differ from every adaptive run by up to 4.5e-3 in
S21, largest at f0, and their 9 points sit on the fitted notch with an rms residual of 1.1e-3. The same 9
frequencies solved with `set_numerical(tolerance=1e-9)` (`full-solves-linear-1e-9`) agree with `adaptive` to at
most 2.4e-6 at every point, and fit to the same notch as `adaptive` (rms residual 1.3e-6). All runs share the
mesh, the order and the linear solver, so the tolerance is the one setting that differs. At f0 with port 1
driven, Palace prints:

| Solve at f0 = 7.3566478 GHz | GMRES iterations | ‖E‖ | Field energy (J) | S21 |
|---|---|---|---|---|
| `full-solves`, linear tolerance 1e-6 | 21 | 1.446067e4 | 4.707e-7 | −1.868e-2 − 5.173e-2i |
| `full-solves-linear-1e-9` | 37 | 1.452735e4 | 4.751e-7 | −2.226e-2 − 4.895e-2i |
| `adaptive`, its reduced model | – | 1.452733e4 | 4.751e-7 | −2.226e-2 − 4.896e-2i |
| `add-to-prom`, its full solve at f0 (linear tolerance 1e-6) | 25 | – | 4.728e-7 | – |
| `add-to-prom`, its reduced model | – | 1.452729e4 | 4.751e-7 | −2.225e-2 − 4.895e-2i |

The nearest full solve to f0 in `adaptive` is the one its log prints as 7.357 GHz; in `add-to-prom` it is at f0
itself. Against the 1e-6 full solves the gap at f0 is 4.537e-3 in `adaptive` and 4.535e-3 in `add-to-prom`;
against the 1e-9 ones it is 2.0e-6 and 1.3e-5. A full solve at 1e-9 took 118 s, at 1e-6 87 s (29 to 39 GMRES
iterations against 14 to 23).

## What the runs show

Numbers only; the agreement a result must reach is the Human's.

- **A band with a dense window, adaptive at 1e-3** (`adaptive`): 9 and 4 full solves, 1,190 s to build the
  model, then 0.548 and 0.555 s per frequency and port for 2,502 frequencies; 3,948 s in all. Its S21 is within
  2.4e-6 of the full solves at a linear tolerance of 1e-9 at all 9 of their frequencies.
- **The same with `AddToPROM` at f0** (`add-to-prom`): 5 and 5 full solves, f0 among them, 921 s to build the
  model and 3,588 s in all; its S21 is within 1.4e-5 of `adaptive` and 1.3e-5 of the 1e-9 full solves.
- **Tolerance 1e-5**: the default cap of 20 full solves per port was reached on both ports (3,710 s to build,
  7,428 s in all) and `driven_result()` refuses the run; Palace's S21 is still within 3.1e-6 of `adaptive`. With
  the cap at 40 it converged with 32 and 15 full solves (4,370 s to build, 0.79 and 0.80 s per frequency and
  port, 8,352 s in all), within 3.2e-6 of `adaptive` and 2.5e-6 of the 1e-9 full solves.
- **Surface EPR off** (`no-surface-epr`): 0.528 and 0.532 s per frequency and port against 0.548 and 0.555 with
  the 7 Surface-EPR rows; Palace's `Postprocessing` timer 2,157 s against 2,254 s of the 3,948 s run. The model
  and S21 are the same (largest difference 1.6e-9).
- **A uniform grid, adaptive at 1e-3** (`uniform-2500`, 2.0 MHz apart): the same 9 and 4 full solves as
  `adaptive`, 0.544 and 0.556 s per frequency and port, 3,940 s in all. No grid frequency is in the notch: the
  nearest is 0.68 MHz above f0, where abs(S21) is −0.94 dB. The fit on the window's 10 grid points still gives
  f0 within 5 Hz and Q_l within 0.03 of `adaptive`'s. At 25,000 points the same costs give
  1,190 + 25,000 × (0.544 + 0.556) = 28,690 s (7.97 h).
- **Full solves**: 87 to 88 s per frequency and port at the default linear tolerance of 1e-6, 117 to 119 s at
  1e-9; the 1e-6 ones are off by up to 4.5e-3 near f0 (below).
- **The notch against the Eigenmode mode**: every run but the 1e-6 full solves gives f0 from 7.356663906 to
  7.356663916 GHz, Q_l from 11,140.75 to 11,140.82 and abs(Q_c) from 11,261.80 to 11,261.86; Q_i, from the small
  difference 1/Q_l − cos φ/abs(Q_c), is about 1.04e6. The first Eigenmode run, with SCGSim's linear tolerance of
  1e-6, gives a mode 16.1 kHz lower with Q = 13,947.6, 25 % above Q_l; with the linear tolerance at 1e-9 the mode
  is at 7.356663931 GHz with Q = 11,141.7 (2,074 s against 1,431 s).

## What failed

- **The Eigenmode runs are refused by `resolve_palace_result`.** Palace ended with exit code 0 and wrote
  `eig.csv`, `port-Q.csv`, `port-I.csv`, `port-V.csv`, `domain-E.csv` and `surface-Q.csv`, but the run's receipt
  says `failed`, because SCGSim's Eigenmode run-folder contract expects `port-EPR.csv` whenever the config has
  lumped ports, and Palace writes it only for ports with an inductance. `readback` records the refusal and reads
  the modes from `eig.csv`, `port-Q.csv` and the ParaView fields; the issue is
  [#311](https://github.com/NCUPHY-LTLab/scgsim/issues/311).
- **`tolerance-1e-5` reached Palace's default cap of 20 full solves on both ports**, with last errors of
  1.98e-4 and 3.71e-4, and `driven_result()` refuses it as unconverged. `fit` reads its S21 from Palace's
  `port-S.csv` to compare it, and the table marks it so.
