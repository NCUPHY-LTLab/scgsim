"""Exchange of the emitted solids with other tools: the AEDT STEP export and its import guard (design brief decision D4, experiment E4).

``step`` writes the STEP and its guard and needs OCP (the optional ``occt`` dependency group) at import: call
:func:`~scgsim.geometry.sgb.checks.verification.ocp.require_ocp` first to get the readable error.  ``guard`` reads
and checks a written guard without OCP.
"""
