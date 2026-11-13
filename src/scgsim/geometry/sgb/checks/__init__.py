"""S9: the checks of the model and of the files written from it.

In: the model (``LabelledArrangements``, ``Topology``, ``RoutedModel``) and, for the file checks, a written XAO or BREP
with its sidecar.
Out: the ``ValidationReport`` (a failed invariant raises, so no failing model is returned), the sliver report, and the
load-side and OCCT mismatches of a written file.
Next: ``output`` writes the files of a model that passed; the file checks run on what it wrote.

* ``validation``    the structural invariants, fail closed and never disabled
* ``slivers``       the local feature size on exact carriers; report only
* ``verification``  the Gmsh load check and the OCCT checks of a written file
"""
