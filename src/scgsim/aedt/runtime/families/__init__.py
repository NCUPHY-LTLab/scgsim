"""Build, solve and export one request family (HFSS, Q3D, Q2D) in the AEDT application the transaction owns.

In: the validated request spec and prepared geometry, with the open AEDT application, from ``runtime.transaction``; ``epr.native_workflow`` and ``epr.adaptive`` call the HFSS Eigenmode setup, export and history steps.
Out: the solved native design, its exported files in the run folder and the native evidence the receipt records.
Next: ``runtime.transaction`` writes the receipt; ``results`` reads the exported files.
"""
