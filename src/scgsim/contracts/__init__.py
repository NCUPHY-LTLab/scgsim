"""Neutral contracts between SCGSim stages: the definitions of the files one stage writes and another reads.

A contract module imports nothing else from ``scgsim`` and no third-party package, so the writer and every
reader can import it whatever their own layer (``.importlinter``: ``contracts-import-nothing-else``).

``palace_run_folder``
    the Palace run folder: the path of each file, the schema of each record, the problem records
    (``EIGENMODE``, ``ELECTROSTATIC``), the fields of the records the receipt recorder writes, and the file
    helpers and refusal wording the recorder carries into the folder.
``palace_convergence``
    the convergence plan ``palace/config`` writes into the run folder, and the one check that decides from
    the pass folders whether the run converged, which ``palace/results`` runs.
``palace_index_map``
    the sections of the Palace index map, which ``palace/config`` writes and ``palace/results`` and the
    Palace preview loader read.
``sgb_sidecar``
    the SGB sidecar: its record definitions, the JSON Schema derived from them, its version and the shared
    reader.
"""
