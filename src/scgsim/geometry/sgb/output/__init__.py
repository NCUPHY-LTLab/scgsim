"""S10: the files of a validated model: BREP and XAO, the sidecar and the AEDT STEP export.

In: the validated ``ConformalModel``.
Out: the byte-deterministic BREP and XAO, the ``scgsim.sgb.sidecar.v3`` JSON that binds every planned id to its native
tag, and the STEP of the emitted solids with its guard.
Next: the mesher and the solver adapters read the files; ``checks.verification`` checks them.

* ``emission``  the numbering of the shapes and the direct BREP/XAO writer
* ``sidecar``   the versioned JSON sidecar and its JSON Schema
* ``exchange``  the AEDT STEP export and its import guard
"""
