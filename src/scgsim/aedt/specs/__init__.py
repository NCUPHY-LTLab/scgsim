"""The AEDT request records for HFSS, Q3D and Q2D runs, with their payload decoders.

In: a caller's request values, or a stored request payload.
Out: immutable spec records and their canonical payloads (``parse_aedt_spec`` picks the family); this file re-exports the wave ``TerminalPort`` record and ``HfssDrivenGeometrySpec``.  The lumped records (``LumpedRlc``, ``LumpedTerminalPort``) are built from the plan's lumped elements, not by a caller.
Next: ``preparation`` writes the specs into a handoff, ``runtime`` executes them and ``results`` checks a run's receipt against them.
"""

from .common import TerminalPort
from .hfss import HfssDrivenGeometrySpec

__all__ = ["TerminalPort", "HfssDrivenGeometrySpec"]
