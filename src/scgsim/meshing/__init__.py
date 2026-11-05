"""Standalone meshing of SGB geometry: sizes from geometry facts, a quality audit and a repair loop.

The module reads only the sidecar JSON (``scgsim.sgb.sidecar.v2``) and the XAO; it imports no SGB, AEDT or
Palace code.  Importing the package does not import Gmsh: every Gmsh call runs in a worker process.
"""

from .errors import MeshingError, MeshRepairError, SidecarError
from .policy import RULES_VERSION, RuleSet

__all__ = ["RULES_VERSION", "MeshRepairError", "MeshingError", "RuleSet", "SidecarError"]
