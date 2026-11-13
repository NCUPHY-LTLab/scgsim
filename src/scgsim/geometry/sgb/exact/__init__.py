"""The exact vocabulary every stage shares: integer geometry and the records the stages hand to one another.

In: nothing; this folder imports no other part of SGB.
Out: the model grid, plane predicates, circle arithmetic, carriers and their lineage, labels, prisms, cut-family,
band-region and support records, content-addressed ids and the stage stopwatch.
Next: every stage imports it.

* ``grid``         the model grid and the conversion of declared lengths onto it
* ``plane``        exact integer plane geometry
* ``circle_math``  platform-independent circle arithmetic
* ``carriers``     the exact curves edges come from, and their lineage
* ``labels``       the labels arrangement cells receive
* ``prisms``       the prism, footprint and catalogue records
* ``families``     cut families, the record bands hand to the arrangement
* ``band_regions`` the band regions and required seams of an inset family
* ``support``      the support region of a surface partition
* ``identity``     content-addressed ids
* ``stopwatch``    wall time of the named stages of one build
"""
