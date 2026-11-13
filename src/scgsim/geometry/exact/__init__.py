"""Exact arithmetic on the model grid and the binding of boundary records: the kernel the geometry stages share.

In: the effective ``GeometryBuildInput`` (from ``models``), for the declared grid, the footprints and the boundary
binding; this folder reads ``models`` only.
Out: the model grid, exact plane predicates, circle arithmetic, carriers and their lineage, each Entity's footprints
on the grid with its boundary records bound to exact lines and circular arcs, and the refusals of the kernel.
Next: the junction partition (``junctions``) reads the bound rings of a lumped port's terminals, ``surface_epr`` and
Palace's Surface-EPR config read ``Grid``, and every SGB stage reads all of it.

* ``errors``       the base class of SGB's refusals and the two refusals this kernel raises
* ``grid``         the model grid, the grid an input declares and the conversion of declared lengths onto it
* ``plane``        exact integer plane geometry
* ``circle_math``  platform-independent circle arithmetic
* ``carriers``     the exact curves edges come from, and their lineage
* ``footprints``   where each Entity's planar footprint comes from, on the model grid
* ``source_arcs``  the circular arcs of a boundary record on the grid
* ``boundaries``   the binding of authored boundaries and GDS reconstructions to exact lines and arcs
"""
