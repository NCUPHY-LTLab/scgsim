"""Execute one prepared handoff inside AEDT and record what it produced.

In: a handoff folder from ``preparation``, started by ``python -m scgsim.aedt.run`` or a native batch (``scgsim.aedt.batch``).
Out: the run folder: the saved AEDT project, the native exports and the run receipt, which binds them to the handoff and the runtime source; also the Q3D stop and save requests a notebook sends to a running solve (``control``).
Next: ``results`` reads the receipt and the run folder offline.
"""
