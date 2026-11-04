"""Exchange of the emitted solids with other tools: the AEDT STEP export (design brief decision D4, experiment E4).

Only ``step`` lives here today.  It needs OCP (the optional ``occt`` dependency group) at import: call
:func:`~scgsim.geometry.sgb.verification.ocp.require_ocp` first to get the readable error.
"""
