"""The exact vocabulary every stage shares: integer geometry, the records the stages hand to one another, and the boundary binding.

In: the effective ``GeometryBuildInput``, for the footprints and the boundary binding; this folder imports no other
part of SGB but ``errors``.
Out: the model grid, plane predicates, circle arithmetic, carriers and their lineage, labels and the interface type
of a face, prisms, cut-family, band-region and support records, content-addressed ids, the stage stopwatch, and
the footprints of the input with their boundary records bound to exact lines and arcs.
Next: every stage of SGB imports it; the junction partition (``geometry.junctions``) reads the bound rings of a
lumped port's terminals from it.

* ``grid``         the model grid, the grid an input declares and the conversion of declared lengths onto it
* ``plane``        exact integer plane geometry
* ``circle_math``  platform-independent circle arithmetic
* ``carriers``     the exact curves edges come from, and their lineage
* ``labels``       the labels arrangement cells receive
* ``interfaces``   the interface type of a face from the domains on its two sides
* ``footprints``   where each Entity's planar footprint comes from, on the model grid
* ``source_arcs``  the circular arcs of a boundary record on the grid
* ``boundaries``   the binding of authored boundaries and GDS reconstructions to exact lines and arcs
* ``prisms``       the prism, footprint and catalogue records
* ``families``     cut families, the record bands hand to the arrangement
* ``band_regions`` the band regions and required seams of an inset family
* ``support``      the support region of a surface partition
* ``identity``     content-addressed ids
* ``stopwatch``    wall time of the named stages of one build
"""
