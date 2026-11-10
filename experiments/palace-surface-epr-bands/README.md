# Palace Surface-EPR inset bands, end to end

`run_case.py` runs one roster layout with Surface-EPR inset bands through the public
`EigenmodeSim` API and stock Palace 0.18.1, and writes the four checks of issue #84 to
`report.json`: the band groups SGB built, the `Dielectric` rows of the config and their index-map
entries, the `surface-Q.csv` columns Palace wrote, and the rows `resolve_palace_result` and
`show_epr` read back. It records numbers and judges none. The results are on issue #84.

## Cases

| `--case` | Layout (public OrcaPen SC PDK) | Nets | Port |
|---|---|---|---|
| `simplepad` | `square_pad_capacitor(200, 20, 1000, 2)` | `pad`, `ground` | `chip/o_junction_lumped`, 10 nH |
| `circular_pad` | `circular_pad_capacitor(100, 0, 20, 1000, 2)` | `pad`, `ground` | `chip/o_junction_lumped`, 10 nH |
| `airbridge` | `cpw_airbridge_coupon()`, stack with airbridges | `signal`, `ground` (both grounds and the bridge) | none |

`--route A` or `--route B` selects the Palace route. Every case uses MA, MS and SA films from the
PDK's `LiteratureCentral_*` presets with `inset_margins_um=[0.5, 1.0]`, one mode, FEM order 2 and up
to 2 AMR passes: settings that check the pipeline, not convergence. `--mesh-recipe uniform-boundary`
replaces the default `FeatureRecipe()` by `UniformBoundaryRecipe()`, as a diagnostic only.

## Run it

Use an environment with the OrcaPen SC PDK revision CI's `pdk-tests` job pins and the SCGSim wheel
of this checkout, installed as that job installs them. Then, for one case:

```bash
export PALACE_EXECUTABLE=/path/to/palace        # the stock Palace 0.18.1 wrapper script
export PALACE_MPI_BIN=/path/to/mpi/bin          # the folder holding the mpirun Palace was built with
python run_case.py prepare  OUT --case simplepad --route B --processes 16
python run_case.py execute  OUT
python run_case.py readback OUT
```

`prepare` runs `simulation_drc`, `mesh()`, `write_config()` and
`prepare_handoff(profile="direct-local")` into `OUT/run`, and writes `OUT/prepare.json`. The handoff
script puts `PALACE_MPI_BIN` on `PATH` through its `setup_commands`, because the Palace wrapper finds
`mpirun` there. `execute` runs that script and stops it after 30 minutes. `readback` writes
`OUT/report.json`. `OUT` holds the mesh and Palace output; keep it out of the repository.
