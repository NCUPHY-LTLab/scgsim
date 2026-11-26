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
# ... the same three stages for adaptive, add-to-prom, tolerance-1e-5, no-surface-epr, full-solves, uniform-25000
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

Each Driven run varies one setting of `adaptive`. f0 and Q below are the Eigenmode run's, which places the
dense window and the full solves.

| Run | What it solves |
|---|---|
| `eigenmode` | `EigenmodeSim`, 4 modes from 6.5 GHz, the two ports as 50 Ω loads; every mode saved as a ParaView field |
| `adaptive` | `[Linear(4e9, 9e9, points=501), Linear(f0 − 20·f0/Q, f0 + 20·f0/Q, points=2001)]`, `adaptive_tolerance=1e-3`: 2,502 frequencies |
| `add-to-prom` | `adaptive`, with `{"Type": "Point", "Freq": [f0], "AddToPROM": true}` added to the written `config.json`'s `Solver.Driven.Samples` before the handoff: a full solve at f0 that SCGSim does not expose |
| `tolerance-1e-5` | `adaptive` with `adaptive_tolerance=1e-5` |
| `no-surface-epr` | `adaptive` on a plan without `set_surface_epr` |
| `full-solves` | `[Points([...])]`, 9 frequencies of the window's own grid 0.75·f0/Q apart (±3 f0/Q), solved in full |
| `uniform-2500` | `[Linear(4e9, 9e9, points=2500)]`, `adaptive_tolerance=1e-3` (2 MHz apart) |

The resonator mode is the Eigenmode mode with the largest share of its electric energy above y = 200 µm, where
only the meander is: the energy density `U_e` of each mode's ParaView field, integrated cell by cell.
