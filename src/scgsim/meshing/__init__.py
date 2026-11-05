"""Standalone meshing of SGB geometry: sizes from geometry facts, a quality audit and a repair loop.

``mesh_sgb(xao, sidecar, output_dir, threshold=...)`` meshes an SGB XAO with Gmsh so that no tetrahedron
is below the quality threshold, or raises :class:`MeshRepairError` carrying the quality report.  The module
reads only the sidecar JSON (``scgsim.sgb.sidecar.v2``) and the XAO; it imports no SGB, AEDT or Palace code.
It trusts that the two files are the pair one SGB build wrote and does not check them against each other.

Importing the package does not import Gmsh: every Gmsh call runs in a worker process.
"""

from .entry import MeshResult, mesh_sgb
from .errors import MeshingError, MeshRepairError, SidecarError
from .policy import RULES_VERSION, RuleSet
from .repair import Budgets

__all__ = [
    "RULES_VERSION",
    "Budgets",
    "MeshRepairError",
    "MeshResult",
    "MeshingError",
    "RuleSet",
    "SidecarError",
    "mesh_sgb",
]
