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

## Choose a simulation

| Goal | Start here |
|---|---|
| Capacitance | [Palace Electrostatic or AEDT Q3D](docs/tutorials/capacitance.qmd) |
| Resonance and EPR | [Palace or HFSS modes](docs/tutorials/resonance.qmd) · [Energy participation](docs/tutorials/resonance-epr.qmd) |
| Driven S-parameters | [HFSS Modal or Terminal](docs/tutorials/scattering.qmd) |
| Transmission-line parameters | [AEDT Q2D](docs/tutorials/transmission-line.qmd) |

## Prepare, configure, run, read

Define the geometry, configure the model, run on its solver host, then read the returned quantities with their units.

![SimplePad layout: square pad, surrounding Ground and authored locator.](docs/assets/simple-pad-layout.png)

**[Install SCGSim](docs/course/installation.qmd)** · **[Choose a simulation](docs/examples.qmd)**
