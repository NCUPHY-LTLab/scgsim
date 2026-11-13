"""S2: the effective input as prisms (planar footprints extruded over Z intervals), levels and slabs.

In: the effective ``GeometryBuildInput`` and ``RepresentationContext``.
Out: the ``PrismCatalogue`` (sources, ports and the declared grid as prisms) and the ``LevelStructure`` (levels, slabs
and active sets).
Next: ``arrangement`` derives the arrangements of the catalogue (its physical prisms alone first, for ``bands``).
"""
