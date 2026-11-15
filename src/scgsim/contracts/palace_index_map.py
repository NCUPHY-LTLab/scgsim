"""The sections of the Palace index map: which part of ``config.json`` each index-map entry describes.

``palace/config`` writes one entry per configured terminal, lumped port, energy domain and Surface-EPR row,
under the section named here; ``palace/results`` and the Palace preview loader read the entries by the same
names.  The names are the paths of the matching blocks in Palace's ``config.json``.
"""

TERMINAL_SECTION = "Boundaries.Terminal"
LUMPED_PORT_SECTION = "Boundaries.LumpedPort"
ENERGY_SECTION = "Domains.Postprocessing.Energy"
DIELECTRIC_SECTION = "Boundaries.Postprocessing.Dielectric"
"""A Surface-EPR row: one ``Boundaries.Postprocessing.Dielectric`` entry of the config."""
