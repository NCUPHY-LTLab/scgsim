"""Neutral contracts between SCGSim stages: the definitions of the files one stage writes and another reads.

A contract module imports nothing else from ``scgsim`` and no third-party package, so the writer and every
reader can import it whatever their own layer (``.importlinter``: ``contracts-import-nothing-else``).

``palace_run_folder``
    the Palace run folder: the path of each file, the schema of each record, the problem records
    (``EIGENMODE``, ``ELECTROSTATIC``) and the file helpers the receipt recorder carries into the folder.
``sgb_sidecar``
    the SGB sidecar: its record definitions, the JSON Schema derived from them, its version and the shared
    reader.
"""
