"""Parse the native convergence reports of HFSS, Q3D and Q2D runs.

In: the convergence report files a run exported into its run folder.
Out: strict per-pass convergence records, refused when a report does not match its family's format.
Next: ``runtime/families`` binds them into the result the receipt records; the ``results`` readback reads them again offline.
"""
