"""Write one portable AEDT handoff from a request spec or a geometry plan, without opening AEDT.

In: a Q3D or Q2D request spec (``specs``) for ``prepare_handoff``, or a ``GeometryPlan`` snapshot from ``scgsim.geometry`` that ``prepare_sgb_model`` builds into the SGB model the HFSS and Q3D adapters and ``EigenmodeSim`` carry.
Out: a handoff folder (``HandoffPlan``) holding the request payload, the SGB model's STEP, BREP, sidecar and guard in its ``model/`` folder, and the hash cohort that binds its files.
Next: ``runtime`` executes the handoff and imports the model; ``results`` checks a run against the cohort and the model's identity.
"""
