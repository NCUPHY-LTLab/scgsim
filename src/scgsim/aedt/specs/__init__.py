"""The AEDT request records for HFSS, Q3D and Q2D runs, with their payload decoders.

In: a caller's request values, or a stored request payload.
Out: immutable spec records and their canonical payloads (``parse_aedt_spec`` picks the family); this file re-exports the lumped terminal records and ``HfssDrivenGeometrySpec``.
Next: ``preparation`` writes the specs into a handoff, ``runtime`` executes them and ``results`` checks a run's receipt against them.
"""

from .common import LumpedRlc, LumpedTerminalPort, TerminalPort
from .hfss import HfssDrivenGeometrySpec

__all__ = ["LumpedRlc", "LumpedTerminalPort", "TerminalPort", "HfssDrivenGeometrySpec"]
