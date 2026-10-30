# SCGSim

[![License: Apache-2.0](docs/assets/badges/license.svg)](https://github.com/NCUPHY-LTLab/scgsim/blob/develop/LICENSE)
[![Python: 3.12–3.13](docs/assets/badges/python.svg)](docs/course/installation.qmd)
[![Tutorial: Layout EM](docs/assets/badges/tutorial.svg)](docs/examples.qmd)
[![AEDT: HFSS / Q3D / Q2D](docs/assets/badges/aedt.svg)](docs/backend-support.qmd)
[![Palace: Electrostatic / Eigenmode](docs/assets/badges/palace.svg)](docs/course/palace-eigenmode.qmd)
[![Gmsh: Meshing](docs/assets/badges/gmsh.svg)](docs/concepts/geometry.qmd)
[![gsim: Unofficial Downstream](docs/assets/badges/gsim.svg)](docs/goals-and-upstream.qmd)

![SCGSim orca and penguin introducing a chip layout, finite-element mesh and illustrative electromagnetic field.](docs/assets/scgsim-overview-orca-penguin.png)

SCGSim is a Python toolkit for electromagnetic simulation of superconducting
circuits. You describe a layout by its named conductors and materials; SCGSim
prepares the input for Palace or Ansys Electronics Desktop (AEDT), and then
loads the solver output into reports with units, conductor names and mode
numbers attached.

SCGSim started from [gsim](https://github.com/gdsfactory/gsim) and is now an
independent project with its own geometry builder and solver APIs. It is not
an official gsim distribution.

## What you can compute

<div class="table-responsive">

| Question | Workflow | Result |
|---|---|---|
| How strongly are conductors coupled electrically? | Palace Electrostatic or AEDT Q3D | Capacitance matrix (F) |
| Where are the resonances, and where is their energy? | Palace Eigenmode or HFSS Eigenmode | Mode frequencies (Hz), quality factors, energy participation |
| How does the circuit respond between ports? | HFSS Driven Modal or Driven Terminal | S-parameters versus frequency |
| What are the conductor matrices of a finite structure? | AEDT Q3D | C/G and R/L matrices |
| What are the line properties per unit length? | AEDT Q2D | C/G/L/R per unit length |

</div>

SCGSim does not support Palace Driven or Palace Magnetostatic simulations.
See [Supported calculations](docs/backend-support.qmd) for the full list.

## How it works

![Layout and materials define the physical question. SCGSim prepares a run
directory, you run the solver locally or on another host, and SCGSim reads
the returned results.](docs/assets/scgsim-workflow.svg)

1. **Describe the model.** Load a component from your PDK and assign its
   shapes to electrical conductors (Nets), such as `pad` and `ground`.
2. **Choose the calculation.** Set up an Electrostatic, Eigenmode, Driven,
   Q3D or Q2D problem with its mesh and solver settings.
3. **Prepare a run directory.** SCGSim writes the mesh, solver configuration
   and a launch script. It never starts the solver on its own.
4. **Run the solver** on the same computer, a workstation with AEDT, or an
   HPC cluster.
5. **Read the results.** Load the returned directory to get capacitance,
   frequencies, participation ratios or S-parameters in a report.

Other features:

- **Curved geometry.** Palace models keep arcs and splines drawn in the
  source component instead of turning them into polygons.
- **Energy participation and loss.** Compute surface, bulk and junction
  participation, and change loss assumptions without solving again.
- **Mesh control.** Choose the Palace mesher, its threads and the geometric
  order, and inspect a mesh summary before running.

## Get started

[Install SCGSim](docs/course/installation.qmd), then work through the first
tutorial, [Pad capacitance](docs/tutorials/capacitance/prepare.qmd). It
uses this model: a 200 µm square pad in a ground plane on a silicon die. Each
tutorial has three steps: prepare a run directory, run the solver on it, and
analyze the returned result.

![SimplePad: a square pad within Ground, and a short sheet across the east
gap where the resonance tutorial attaches a junction.](docs/assets/simple-pad-layout.png)

The [tutorial overview](docs/examples.qmd) lists all tutorials and how-to
guides.

### Requirements

- Python 3.12 or 3.13.
- For Palace calculations: a Palace executable on the computer that runs the
  solver.
- For AEDT calculations: AEDT 2024.2 with a licence, and PyAEDT 1.3.0.

Installing SCGSim does not install Palace or AEDT. You can prepare runs on a
laptop and run the solver elsewhere.

## Documentation

- [Tutorials and how-to guides](docs/examples.qmd): step-by-step
  calculations.
- Concepts: [choosing the physical problem](docs/concepts/problems.qmd),
  [describing geometry](docs/concepts/geometry.qmd),
  [energy participation and loss](docs/eigenmode-epr.qmd), and
  [boundaries and numerics](docs/concepts/model-and-numerics.qmd).
- [Reference](docs/backend-support.qmd): exact API and input rules.
- [Maintainer guide](docs/ownership.qmd): for contributors to SCGSim.

## Python API

```python
from scgsim.geometry import GeometryPlan, VacuumRegionSpec
from scgsim.palace import ElectrostaticSim, EigenmodeSim, resolve_palace_result
```

AEDT workflows live in `scgsim.aedt`. If you are upgrading an existing
project, see the [API migration guide](docs/api-migration.qmd).
