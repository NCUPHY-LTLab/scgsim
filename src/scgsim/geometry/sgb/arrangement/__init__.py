"""S4 and S5: the planar arrangement of the curves on the grid, its cells labelled and dissolved.

In: a ``PrismCatalogue`` (the ``LevelStructure`` is built from it): the physical prisms alone for the pre-pass, then
the whole catalogue with the cut families of ``bands``.
Out: the ``LabelledArrangements``: every slab and level arrangement, dissolved, labelled and content-addressed.
Next: ``bands`` finds the inset support faces on the pre-pass; ``topology`` stacks the final arrangements into
vertices, edges, faces and volumes.

* ``noding``      iterated snap rounding on the integer grid
* ``dcel``, ``arcs`` the half-edge structure with its faces and holes, and the realisation of arc edges
* ``labelling``   cell labels, dissolution, membership and the links between slab and level arrangements
* ``derivation``  the slab and level arrangements as one labelled, content-addressed result
"""
