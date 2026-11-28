"""The PyAEDT calls that create, assign and read back native objects, below the request families.

In: PyAEDT application objects and the prepared records, from ``runtime/families``, ``epr`` and ``runtime.transaction``.
Out: native materials, setups, saved-design reads and the solver resources around one solve, each with the read-back evidence that checks it; no geometry is built here.
Next: ``runtime/families`` and ``epr`` put that evidence into their results; ``scgsim.visualization`` reads ``compatibility`` for the pinned PyAEDT transport.
"""
