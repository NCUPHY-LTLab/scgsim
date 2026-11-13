"""S4 and S5: the planar arrangement of the curves on the grid, its cells labelled and dissolved.

In: the ``PrismCatalogue`` with its cut families (the ``LevelStructure`` is built from it).
Out: the ``LabelledArrangements``: every slab and level arrangement, dissolved, labelled and content-addressed.
Next: ``topology`` stacks them into vertices, edges, faces and volumes.

* ``noding``      iterated snap rounding on the integer grid
* ``dcel``, ``arcs`` the half-edge structure with its faces and holes, and the realisation of arc edges
* ``labelling``   cell labels, dissolution, membership and the links between slab and level arrangements
* ``derivation``  the slab and level arrangements as one labelled, content-addressed result
"""
