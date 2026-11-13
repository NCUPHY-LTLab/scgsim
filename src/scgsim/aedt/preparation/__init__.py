"""Write one portable AEDT handoff from a request spec or a geometry plan, without opening AEDT.

In: an AEDT request spec (``specs``) for ``prepare_handoff``, or a ``GeometryPlan`` snapshot from ``scgsim.geometry`` for the HFSS and Q3D geometry adapters and ``EigenmodeSim``.
Out: a handoff folder (``HandoffPlan``) holding the request payload, the prepared planar geometry and the hash cohort that binds its files, plus the native object names (``planar_source``) every later stage gives the same source entity.
Next: ``runtime`` executes the handoff; ``runtime/native`` and ``epr`` name native objects through ``planar_source``; ``results`` checks a run against the cohort.
"""
