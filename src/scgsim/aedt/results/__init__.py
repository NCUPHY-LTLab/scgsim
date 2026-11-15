"""Read a finished or preparation-only AEDT run offline, and hold the receipt checks its writers share.

In: a run folder with its receipt (written by ``runtime``) and the handoff it ran (written by ``preparation``).
Out: ``ResolvedRun``, ``PreparedModel`` and ``EprResult``, read only from a current receipt; also the receipt schema and runtime-source record (``provenance``, ``runtime_inventory``), the contained-path check and the matrix export parsers that ``preparation`` and ``runtime`` call while writing.
Next: notebooks through the ``scgsim.aedt`` facade, ``presentation``, ``scgsim.aedt.batch_stages`` and the AEDT loaders of ``scgsim.visualization``.
"""
