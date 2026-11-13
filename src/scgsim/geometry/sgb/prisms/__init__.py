"""S2: the effective input as prisms (planar footprints extruded over Z intervals), levels and slabs.

In: the effective ``GeometryBuildInput`` and ``RepresentationContext``.
Out: the ``PrismCatalogue`` (sources, ports and the declared grid as prisms), the ``LevelStructure`` (levels, slabs and
active sets) and the bound rings of chosen conductors.
Next: ``bands`` adds the cut families the requests ask for.
"""
