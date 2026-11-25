# Palace Driven, launcher to launcher

`run_case.py` runs the Driven case of [issue 187](https://github.com/NCUPHY-LTLab/scgsim/issues/187)
through the public `DrivenSim` API and stock Palace 0.18.1, reads it back with
`resolve_palace_result(...).driven_result()`, writes its Touchstone file and records the checks of
section 6 of the issue's design plan in `report.json`. It records numbers and judges none: every
tolerance of section 6 is the Human's. It makes the public calls of E1, the first run of the case on
issue 187, in the same order, and reads the run back as the issue's later comments did.

## The case

The layout is the roster's `launcher_to_launcher` (`scripts/roster_cases.py`), composed from public
OrcaPen SC PDK cells until the PDK draws it as one cell
([orcapen-sc-pdk#67](https://github.com/NCUPHY-LTLab/orcapen-sc-pdk/issues/67)):

| Part | Value |
|---|---|
| CPW | `cpw_6_7_6`: signal 7 µm, gaps 6 µm, 900 µm long between the launcher necks |
| Launchers | PDK `launcher`: pad 150 × 150 µm, taper 150 µm, 85 µm of gap behind and beside the pad |
| Port sheets | 85 × 150 µm behind each pad, landing 10 nm into the pad and the ground |
| Die, stack | 1870 × 900 µm, no dicing frame; PDK single-die stack (Si εr 11.9, 500 µm; Al 0.1 µm) |
| Vacuum | padding (0, 0, 1000) µm |

The plan declares the two sheets as 50 Ω `LumpedPort`s to the `ground` Net, left first, so the left
launcher is port 1. Surface EPR uses the PDK's `LiteratureCentral` MA, MS and SA presets: 7
`Dielectric` rows without bands, 18 with `--bands` (margins 0.5 and 1.0 µm, as the nightly roster
check meshes the case). The run is on Route B with no mesh refinement.

## Run it

Use an environment with the OrcaPen SC PDK revision CI's `pdk-tests` job pins and the SCGSim wheel
of this checkout, installed as that job installs them. Then:

```bash
export PALACE_EXECUTABLE=/path/to/palace        # the stock Palace 0.18.1 wrapper script
export PALACE_MPI_BIN=/path/to/mpi/bin          # the folder holding the mpirun Palace was built with
python run_case.py prepare  OUT --order 2 --points 81 --adaptive-tolerance 1e-3
python run_case.py execute  OUT
python run_case.py readback OUT
```

`prepare` runs `simulation_drc`, `mesh()`, `write_config()` and
`prepare_handoff(profile="direct-local")` into `OUT/run` with 8 MPI processes (`--processes`), and
writes `OUT/prepare.json`. The sweep is 2 to 10 GHz: `--points` frequencies, solved in full at each
one unless `--adaptive-tolerance` asks for Palace's adaptive fast sweep. `--order` is the FEM order,
`--bands` cuts the Surface-EPR bands, `--route A` meshes metal as sheets. The handoff script puts
`PALACE_MPI_BIN` on `PATH`, because the Palace wrapper finds `mpirun` there. `execute` runs that
script and writes `OUT/execute.json`. `readback` writes `OUT/launcher_to_launcher.s2p` and
`OUT/report.json`. `OUT` holds the mesh and Palace output; keep it out of the repository.

Run one case at a time: Open MPI binds each job's ranks from the first core, so two Palace jobs
started side by side share the same cores.

## The runs

All on stock Palace 0.18.1 with 8 MPI ranks, one job at a time, on the mesh `mesh()` writes. E1 is the
run of [the issue's comment of 2026-11-25](https://github.com/NCUPHY-LTLab/scgsim/issues/187#issuecomment-7154335384),
read back through `driven_result()` in
[a later comment](https://github.com/NCUPHY-LTLab/scgsim/issues/187#issuecomment-7155412261).

| Run | Order | Sweep | Bands | Mesh | Unknowns | Palace time | Peak memory |
|---|---|---|---|---|---|---|---|
| E1 r1 | 1 | 5 points, full solve at each | no | 204,000 tetrahedra | 280,541 | 28.2 s | 5.96 GB |
| E1 r3 | 2 | 81 points, adaptive, tolerance 1e-3 | no | 204,000 tetrahedra | 1,433,702 | 335.4 s | 13.9 GB |
| repeat of r1 | 1 | 5 points, full solve at each | no | 204,000 tetrahedra | 280,541 | 28.8 s | 5.93 GB |
| r1 with bands | 1 | 5 points, full solve at each | 0.5, 1.0 µm | 355,921 tetrahedra | 494,036 | 52.1 s | 9.83 GB |

The repeat of r1 was made with `run_case.py` as it is here. Its S-parameters equal E1's r1 to
4.1e-14 (largest difference of a complex entry), the last digit Palace prints.

## The checks of section 6

Measured on E1's r3, the order-2 run; the proposed tolerances are the plan's, for the Human to set.

| # | Check | r3 (order 2) | Proposed tolerance |
|---|---|---|---|
| A1 | Public API only; the `.s2p` file read back equals the record | equal | exact |
| A2 | \|S21\| ≥ −0.1 dB and \|S11\| ≤ −20 dB over 2–10 GHz | \|S21\| ≥ −0.00582 dB; \|S11\| ≤ −29.81 dB | as stated |
| A3 | Reciprocity, max \|S12 − S21\| | 7.7e-15 | ≤ 1e-3 |
| A4 | Lossless box, max \|1 − \|S11\|² − \|S21\|²\| | 2.9e-4 | ≤ 1e-3 |
| A5 | Mirror symmetry, max \|S11 − S22\| | 1.6e-4 | ≤ 1e-3 |
| A8 | Surface EPR for every row, port and frequency; margin rows not above their baseline | 162 records × 7 rows, no margins | exact |
| A9 | Port order in Palace's index map, the record and the Touchstone file | `chip/o_lumped_left` is port 1 in all three | exact |

arg S21 at 2, 4, 6, 8 and 10 GHz is −11.1180°, −22.2353°, −33.3520°, −44.4691° and −55.5891° (r3),
a delay of 15.44 ps. Each port's adaptive sweep converged with 4 full solves, at 2, 10, 6.142 and
3.744 GHz. A6 (n_eff from two line lengths) needs the 1800 µm variant, and A7 (Palace against HFSS)
an HFSS run; neither is here.

**Order 1 is not enough for phase.** At order 1, arg S21 at 2 GHz is −11.512° and |S11| reaches
−21.2 dB at 10 GHz, where order 2 gives −29.8 dB; the plan's fact 12 measures the order-1 phase about
7 % off in n_eff.

**With bands** (order 1): every one of the 10 records (2 ports × 5 frequencies) carries 8
whole-surface rows and 10 margin rows, and no margin row is above its baseline (the largest ratio is
0.86). Cutting the bands refines the mesh at the gap edges, so the order-1 values move toward order 2:
arg S21 at 2 GHz is −11.402° and |S11| at 10 GHz −24.46 dB. A3 is 3.4e-6, A4 3.8e-4 and A5 1.4e-3.
