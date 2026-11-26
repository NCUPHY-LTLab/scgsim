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
# ... the same three stages for adaptive, add-to-prom, tolerance-1e-5, tolerance-1e-5-max-40,
# no-surface-epr, full-solves and uniform-2500 (uniform-25000 is defined too)
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

Each Driven run varies one setting of `adaptive`. f0 and Q are the Eigenmode run's resonator mode,
f0 = 7.356647804 GHz and Q = 13,947.6, so f0/Q = 527.4 kHz: the dense window runs from 7.346098839 to
7.367196768 GHz, 10.55 kHz apart, and the full solves from 7.35506546 to 7.35823015 GHz, about 395 kHz apart.

| Run | What it solves |
|---|---|
| `eigenmode` | `EigenmodeSim`, 4 modes from 6.5 GHz, the two ports as 50 Ω loads; every mode saved as a ParaView field |
| `adaptive` | `[Linear(4e9, 9e9, points=501), Linear(f0 − 20·f0/Q, f0 + 20·f0/Q, points=2001)]`, `adaptive_tolerance=1e-3`: 2,502 frequencies |
| `add-to-prom` | `adaptive`, with `{"Type": "Point", "Freq": [f0], "AddToPROM": true}` added to the written `config.json`'s `Solver.Driven.Samples` before the handoff: a full solve at f0 that SCGSim does not expose |
| `tolerance-1e-5` | `adaptive` with `adaptive_tolerance=1e-5` |
| `tolerance-1e-5-max-40` | `tolerance-1e-5` with `adaptive_max_samples=40`: `tolerance-1e-5` reached Palace's default cap of 20 |
| `no-surface-epr` | `adaptive` on a plan without `set_surface_epr` |
| `full-solves` | `[Points([...])]`, 9 frequencies of the window's own grid 0.75·f0/Q apart (±3 f0/Q), solved in full |
| `uniform-2500` | `[Linear(4e9, 9e9, points=2500)]`, `adaptive_tolerance=1e-3`, 2.0 MHz apart |

The issue's uniform 25,000-point sweep was not run. `adaptive`'s measured cost predicts 1,190 s to build the
model plus 25,000 × (0.548 + 0.555) s online, 28,765 s (8.0 h), past the study's limit of about 7 h; so
`uniform-2500` was run and its online cost scaled to 25,000 frequencies below.

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
| adaptive | 501 band + 2001 window, adaptive 0.001 | 521,337 | 3,677,860 | 9, 4 | 1190.0 | 0.548, 0.555 | 3947.6 | 35,738 |
| add-to-prom | 501 band + 2001 window, adaptive 0.001, AddToPROM at f0 | 521,337 | 3,677,860 | 5, 5 | 921.0 | 0.527, 0.539 | 3587.8 | 34,996 |
| tolerance-1e-5 | 501 band + 2001 window, adaptive 1e-05 | 521,337 | 3,677,860 | 20 (cap; error 0.000198), 20 (cap; error 0.000371) | 3710.0 | 0.739, 0.747 | 7428.4 | 40,512 |
| no-surface-epr | 501 band + 2001 window, adaptive 0.001, no Surface EPR | 521,337 | 3,677,860 | 9, 4 | 1190.0 | 0.528, 0.532 | 3842.0 | 35,739 |
| full-solves | 9 full solves | 521,337 | 3,677,860 | - | - | 86.7, 88 | 1586.0 | 33,325 |

Where the adaptive sweeps placed their full solves, in GHz as Palace's log prints them (4 digits):

| Run | Port 1 | Port 2 |
|---|---|---|
| `adaptive`, `no-surface-epr` | 4, 9, 6.555, 8.044, 7.355, 4.883, 7.357, 4.337, 8.654 | 4, 9, 6.555, 8.045 |
| `add-to-prom` | 4, 9, 7.357 (f0), 6.554, 5.077 | 4, 9, 7.357 (f0), 6.554, 5.077 |
| `tolerance-1e-5` | 4, 9, 6.555, 8.044, 7.355, 4.883, 7.357, 4.337, 8.654, 7.458, 8.391, 6.053, 7.162, 5.220, 7.693, 5.837, 7.403, 7.630, 7.325, 7.379 | 4, 9, 6.555, 8.045, 7.355, 4.883, 7.357, 4.337, 8.654, 7.158, 7.606, 7.322, 7.236, 5.807, 5.623, 7.359, 8.863, 7.807, 7.339, 7.345 |

The two ports share one reduced model, so port 2 starts from port 1's solves.

### The notch

| Run | Points fitted | rms residual | f0 (GHz) | Q_l | abs(Q_c) | Q_i | f0 - f0_eig (Hz) | Q_l / Q_eig - 1 | max abs(S21 - S21 of full-solves) | max abs(S21 - S21 of full-solves-linear-1e-9) | max abs(S21 - S21 of adaptive) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.537e-03 | - | - |
| add-to-prom | 2003 | 1.75e-06 | 7.356663907 | 11140.7 | 11261.8 | 1.035e+06 | 1.61e+04 | -2.012e-01 | 4.535e-03 | - | 1.430e-05 |
| tolerance-1e-5 (port-S.csv) | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.540e-03 | - | 3.076e-06 |
| no-surface-epr | 2003 | 1.75e-06 | 7.356663911 | 11140.8 | 11261.8 | 1.036e+06 | 1.611e+04 | -2.012e-01 | 4.537e-03 | - | 1.604e-09 |
| full-solves | 9 | 1.12e-03 | 7.356663554 | 11056.4 | 11228.2 | 7.219e+05 | 1.575e+04 | -2.073e-01 | - | - | 4.537e-03 |

`f0`, `Q_l`, `abs(Q_c)` and `Q_i` are the hanger fit (`notch_fit.py`) of S21 on the dense window's points
(the 9 points of a full-solve run), with the line's delay taken from the band's phase away from the resonance,
14.64 ps (the full-solve runs use `adaptive`'s). `rms residual` is the fit's, in units of S21. The last columns
are the largest `abs(S21 − S21_reference)` over the frequencies a run shares with the reference. The Eigenmode
mode's Q from its complex frequency, 13,947.6, and the two ports' `Q_ext` from `port-Q.csv`, 22,548.5 and
22,484.5 (11,258.2 together), are both Palace's numbers for the same mode.

## What failed

- **The Eigenmode run is refused by `resolve_palace_result`.** Palace ended with exit code 0 and wrote
  `eig.csv`, `port-Q.csv`, `port-I.csv`, `port-V.csv`, `domain-E.csv` and `surface-Q.csv`, but the run's receipt
  says `failed`, because SCGSim's Eigenmode run-folder contract expects `port-EPR.csv` whenever the config has
  lumped ports, and Palace writes it only for ports with an inductance. `readback` records the refusal and reads
  the modes from `eig.csv`, `port-Q.csv` and the ParaView fields; the issue is
  [#311](https://github.com/NCUPHY-LTLab/scgsim/issues/311).
- **`tolerance-1e-5` reached Palace's default cap of 20 full solves on both ports**, with last errors of
  1.98e-4 and 3.71e-4, and `driven_result()` refuses it as unconverged. `fit` reads its S21 from Palace's
  `port-S.csv` to compare it, and the table marks it so.
