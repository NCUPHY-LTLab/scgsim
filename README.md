# SCGSim

**Simulate superconducting layouts with Python.**

[![License: Apache-2.0](docs/assets/badges/license.svg)](https://github.com/NCUPHY-LTLab/scgsim/blob/develop/LICENSE)
[![Python: 3.12–3.13](docs/assets/badges/python.svg)](docs/course/installation.qmd)
[![Tutorial: Layout EM](docs/assets/badges/tutorial.svg)](docs/examples.qmd)
[![AEDT: HFSS / Q3D / Q2D](docs/assets/badges/aedt.svg)](docs/backend-support.qmd)
[![Palace: Electrostatic / Eigenmode](docs/assets/badges/palace.svg)](docs/course/palace-eigenmode.qmd)
[![Gmsh: Meshing](docs/assets/badges/gmsh.svg)](docs/concepts/geometry.qmd)
[![gsim: Unofficial Downstream](docs/assets/badges/gsim.svg)](docs/goals-and-upstream.qmd)

![SCGSim orca and penguin introducing a chip layout, finite-element mesh and illustrative electromagnetic field.](docs/assets/scgsim-overview-orca-penguin.png)

SCGSim prepares Palace and Ansys AEDT simulations, then returns capacitance, resonant modes, S-parameters and per-length line results with units.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then create a study project. These commands install SCGSim `2.0.0rc15` and the OrPen SC PDK component library used below, each pinned to an exact source revision.

```bash
uv init --python 3.12 scgsim-study
cd scgsim-study
uv add 'scgsim[aedt] @ git+https://github.com/NCUPHY-LTLab/scgsim.git@e018c587d5fab1ed1fa6e32f2b00cc31663b941c' \
  'orpen-sc-pdk @ git+https://github.com/NCUPHY-LTLab/orcapen-sc-pdk.git@51985d9507025e8241205712e866c0b804de29b9'
```

`uv` creates `.venv` and records resolved versions in `uv.lock`. Keep `pyproject.toml` and `uv.lock` with the study; run `uv sync` on another computer to recreate the environment.

> **Execution requirements:** The example needs AEDT Q3D 2024.2 installed on this host with a working license. Python packages do not include the native solver. See the [installation guide](docs/course/installation.qmd) for other backend setups.

## Usage example: one script for SimplePad capacitance

This example places a 200 µm square metal pad, a 20 µm gap and surrounding Ground on a 1000 µm square die. The component library supplies the layout, materials and layer stack. The script declares the final `pad` and `ground` electrical membership, then asks Q3D for capacitance and conductance matrices.

![SimplePad layout: square pad, surrounding Ground and authored locator.](docs/assets/simple-pad-layout.png)

Save the complete program below as `simplepad_q3d.py`. Its four sections build the source, prepare solver inputs, execute, and read the result. No code from another page is required first.

```python
from pathlib import Path
from uuid import uuid4
import argparse
import subprocess
import sys

import gdsfactory as gf
import orpen_sc_pdk
from orpen_sc_pdk import get_material_records
from orpen_sc_pdk.tech import get_single_die_layer_stack
from scgsim.geometry import GeometryPlan
from scgsim.aedt import (
    AedtResources,
    MatrixRunControl,
    PdkMaterial,
    prepare_q3d_from_geometry,
    resolve_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-only", action="store_true")
    args = parser.parse_args()

    # 1. Build the layout and declare electrical ownership.
    orpen_sc_pdk.activate()
    component = gf.get_component(
        "square_pad_capacitor",
        pad_side_um=200.0,
        gap_um=20.0,
        die_side_um=1000.0,
        junction_width_um=2.0,
    )
    records = get_material_records()
    plan = GeometryPlan(
        component=component,
        root_id="chip",
        layer_stack=get_single_die_layer_stack(),
        material_records=records,
        coupon_padding_um=0.0,
    )
    plan.set_nets({"pad": ("chip/PAD",), "ground": ("chip/GROUND",)})
    plan.set_vacuum_region(padding=(250.0, 250.0, 1000.0))
    snapshot = plan.prepare()

    # 2. Prepare a new, portable Q3D input directory.
    materials = {
        name: PdkMaterial(
            name, records[name]["material_kind"],
            records[name]["is_superconducting"],
            records[name]["aedt_library_name"],
        )
        for name in ("Al", "Si", "vacuum")
    }
    handoff = prepare_q3d_from_geometry(
        snapshot,
        aedt_version="2024.2",
        modeling="solid",
        output_dir=Path(".artifacts/simplepad-q3d") / uuid4().hex[:12],
        project_name="SimplePad",
        design_name="Capacitance",
        materials=materials,
        net_types={"pad": "Signal", "ground": "Signal"},
        physical_ground_nets=("ground",),
        run_control=MatrixRunControl(
            setup_name="Setup1", frequency_ghz=5.0,
            maximum_passes=10, convergence_percent=1.0,
        ),
        region_padding_um=(250.0, 250.0, 250.0, 250.0, 1000.0, 1000.0),
        resources=AedtResources(cores=4, ram_limit_percent=80),
    )
    print("Inputs:", handoff.run_dir)
    if args.handoff_only:
        return

    # 3. Run AEDT on this licensed host using the current Python environment.
    subprocess.run(
        [sys.executable, "-m", "scgsim.aedt.run",
         "--handoff", str(handoff.metadata_path), "--execute"],
        cwd=handoff.run_dir,
        check=True,
    )

    # 4. Read the returned matrix entries with their names and exported units.
    result = resolve_results(handoff.run_dir)
    for row in result.physics_results():
        print(row)


if __name__ == "__main__":
    main()
```

Run it from the study project:

```bash
uv run simplepad_q3d.py
```

Each invocation creates a new run folder under `.artifacts/simplepad-q3d/`. Printed matrix entries retain the quantity, row, column, value and native exported unit; keep these fields together when using the result.

To inspect the inputs first, run `uv run simplepad_q3d.py --handoff-only`. This option prepares the inputs and portable archive without opening AEDT. The complete mode runs and reads results on the same host.

Both Nets use Q3D's `Signal` type so they remain in the matrix; `physical_ground_nets` records the physical Ground role separately. Continue with the [capacitance tutorial](docs/tutorials/capacitance.qmd) for dimensions, boundary choices and interpretation. Each full tutorial retains the environment and source versions listed on its own page.

## Choose your next simulation

| Goal | Start here |
|---|---|
| Capacitance | [Palace Electrostatic or AEDT Q3D](docs/tutorials/capacitance.qmd) |
| Resonance and EPR | [Palace or HFSS modes](docs/tutorials/resonance.qmd) · [Energy participation](docs/tutorials/resonance-epr.qmd) |
| Driven S-parameters | [HFSS Modal or Terminal](docs/tutorials/scattering.qmd) |
| Transmission-line parameters | [AEDT Q2D](docs/tutorials/transmission-line.qmd) |
