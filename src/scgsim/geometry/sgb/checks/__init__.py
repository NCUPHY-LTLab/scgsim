"""S9: the checks of the model before any file is written.

In: the model (``LabelledArrangements``, ``Topology``, ``RoutedModel``).
Out: the ``ValidationReport`` (a failed invariant raises, so no failing model is returned) and the sliver report.
Next: ``model`` returns the model that passed; ``output`` writes its files and the sliver report beside them.

* ``validation``    the structural invariants, fail closed and never disabled
* ``slivers``       the local feature size on exact carriers; report only
"""
