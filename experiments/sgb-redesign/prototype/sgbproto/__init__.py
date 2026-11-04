"""sgbproto: phase-1 prototype of the redesigned SGB (semantic geometry builder).

Stages map to modules (see README.md): ``records`` / ``levels`` (S2),
``snap_rounding`` / ``arrangement`` / ``layers`` (S4), ``labels`` /
``dissolve`` (S5), ``assembly`` (S6), ``routes`` (S7), ``identity`` (S8),
``validate`` (S9), ``emit`` (S10), ``tiling`` (parallel S4), ``pipeline``
(orchestration), ``scenes`` (example inputs), ``report`` (JSON outputs).
"""
