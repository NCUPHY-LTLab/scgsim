# SimplePad AMR comparison

This driver prepares one public SimplePad Layout and compares Conformal and
Nonconformal AMR in Thin Film (Route A) and Shell Solid (Route B). Each Route
builds one initial Mesh, shared by its two AMR runs. The [Research page](../../docs/research/simplepad-amr.qmd)
explains the model and measured results; the [AMR analysis reference](../../docs/amr-analysis.qmd)
defines the metrics.

Use the SCGSim checkout recorded with the study and a Palace build containing
[`SaveAdaptDiagnostics`](https://github.com/NCUPHY-LTLab/palace/blob/bbe970edb6fda1c59336b8351816dc6018635168/docs/src/features/amr_diagnostics.md).
The launcher must match that Palace build. The preset uses 30 MPI ranks with
two OpenMP threads each and explicitly binds hardware threads. Native logs
retain Open MPI's actual binding report.

## Prepare the Python environment

Run these commands from the SCGSim checkout. The separate environment preserves
the package's development environment and supplies the selected 100 nm PDK.

```bash
uv venv --python 3.12 .artifacts/amr-env
uv pip install --python .artifacts/amr-env/bin/python -e '.[palace]' \
  'orcapen-sc-pdk @ git+https://github.com/NCUPHY-LTLab/orcapen-sc-pdk.git@c75a1d4d2ee07e86a3b5d9b491609e5a1a5b850c' \
  'gdsfactory==9.48.0' 'gmsh==4.15.2' 'plotly==6.9.0' matplotlib
```

## Prepare, run and analyze

Set `PALACE_BIN` and `PALACE_MPIRUN` to the selected native binary and its matching
Open MPI launcher after loading that build's runtime environment. Select a new
study directory on the storage used for simulation data.

```bash
STUDY_DIR="$PWD/.artifacts/simplepad-amr-study"
.artifacts/amr-env/bin/python -m research.palace_amr_simplepad prepare \
  --palace-binary "$PALACE_BIN" --mpi-launcher "$PALACE_MPIRUN" \
  --study-dir "$STUDY_DIR"
.artifacts/amr-env/bin/python -m research.palace_amr_simplepad run "$STUDY_DIR"
.artifacts/amr-env/bin/python -m research.palace_amr_simplepad analyze "$STUDY_DIR"
.artifacts/amr-env/bin/python -m research.palace_amr_simplepad.figures \
  "$STUDY_DIR" "$STUDY_DIR/comparison"
```

Preparation seals all four input packages before native execution. The two
initial Meshes, input digests, independent handoff identities, source versions,
launcher identity, preparation times and resources remain in the study
manifest. Native solve times come from Palace's existing Benchmark.

`run` attempts the four cases sequentially and records their return codes and
receipts. `analyze` also handles failed attempts through explicit inspection,
retaining their status and available observations. A failed command returns
nonzero while preserving the corresponding files and diagnostic records.

The comparison exporter consumes the same public AMR analysis used by the
Benchmark. It writes CSV observations, refinement transitions, standalone SVG
figures and a compact summary. Original matrices and unavailable reasons remain
in the per-case exports. This first batch supplies no reference matrix.
