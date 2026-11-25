"""From a returned run folder to the checked records of the run.

In: the folder and the handoff it must match; for the trust report, also a folder an incomplete run returned.
Out: ``ResolvedRunData`` with its tables and convergence check, the detached result it yields (an Eigenmode
run's EPR, a Driven run's S-parameters), and the AMR snapshots, result selection, failure diagnosis and
convergence check of an inspection.  Next: ``reports``.
"""
