"""Shared grid operations: noding on the model grid, polygon Booleans, tagged regions, overlays, loops and polygon operations.

In: polygon rings and the ``GridScope`` they are read in (``scope``): the model grid, the occurrence frame, whether
the unrounded reference is evaluated, and the declared source a refusal names.
Out: rings noded on that grid (``arrangement``), Boolean results (``booleans``) and loop operations that keep source
provenance; a change no operation can certify is refused where it happens, in the scope's words.
Next: ``source`` builds its snapshot with them; ``interpretation``, ``junctions`` and ``planning`` call the same
operations, ``planning`` on no grid (``UNGRIDDED``). The records the stages share are in ``models``.
"""
