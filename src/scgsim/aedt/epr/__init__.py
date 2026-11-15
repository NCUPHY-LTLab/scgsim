"""Energy participation ratio (EPR) analysis of an HFSS Eigenmode solution, at each stage it passes through.

In: an EPR request (``specs.epr``) with the SGB model from ``preparation``; in AEDT, the solved design from ``runtime/families``; offline, the stored result rows of a run.
Out: the Surface-EPR rows and selection sheets read from the model's sidecar, the field expressions, caches and integrals evaluated in AEDT, and the detached ``EprResult`` with its offline reanalysis.
Next: ``runtime.transaction`` runs the in-AEDT part; ``results.epr`` resolves the result; ``presentation`` plots it.
"""
