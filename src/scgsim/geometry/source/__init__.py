"""The source stage: from a layout and a stack to a prepared snapshot.

In: a layout (components, GDS) and a stack, with the declared grid, vacuum region, port sheets and Surface-EPR specs.
Out: a ``GeometryPlan`` and its snapshot, whose ``GeometryBuildInput`` and stack keep every input identity and carry
the hashes and source audits the later stages record. Each declared recipe returns its value with its audits
(``recipes``); ``grid_projection`` puts the read polygons on the layout's grid.
Next: ``interpretation`` reads the prepared input. ``contact_pads`` resolves the face metal each contact pad attaches
to. ``surface_epr_specs`` holds the Surface-EPR spec rules the plan checks by, which ``surface_epr``, Palace and AEDT
read the specs and their margins with.
"""
