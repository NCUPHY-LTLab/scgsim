"""S10: the files of a validated model: BREP and XAO, the sidecar and the AEDT STEP export.

In: the validated ``ConformalModel``.
Out: the byte-deterministic BREP and XAO, the ``scgsim.sgb.sidecar.v5`` JSON that binds every planned id to its native
tag, the load-side and OCCT mismatches of a written file, and the STEP of the emitted solids with its guard.
Next: the mesher and the solver adapters read the files.

* ``emission``      the numbering of the shapes and the direct BREP/XAO writer
* ``sidecar``       the versioned JSON sidecar and its JSON Schema
* ``verification``  the Gmsh load check and the OCCT checks of a written file (tests and diagnosis; the STEP export
  runs the OCCT comparison)
* ``exchange``      the AEDT STEP export and its import guard
"""
