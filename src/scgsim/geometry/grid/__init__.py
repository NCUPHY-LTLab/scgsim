"""The grid engine of the source stage: noding on the model grid, tagged regions, overlays, loops and polygon operations.

In: polygon rings with the build's ``GeometryGridContext``.
Out: rings noded on that grid, tagged regions and loop operations that keep source provenance; a change no operation can certify is refused where it happens.
Next: ``source`` assembles its snapshot from them; ``interpretation``, ``junctions`` and ``planning`` call the same operations.
"""
