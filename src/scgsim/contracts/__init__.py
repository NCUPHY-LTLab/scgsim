"""Neutral contracts between SCGSim stages: the definitions of the files one stage writes and another reads.

A contract module imports nothing else from ``scgsim`` and no third-party package, so the writer and every
reader can import it whatever their own layer (``.importlinter``: ``contracts-import-nothing-else``).

``sgb_sidecar``
    the SGB sidecar: its record definitions, the JSON Schema derived from them, its version and the shared
    reader.
"""
