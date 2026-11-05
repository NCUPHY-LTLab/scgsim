"""Standalone meshing of SGB geometry: sizes from a recipe, a quality audit and a repair loop.

``mesh_sgb(xao, sidecar, output_dir, threshold=...)`` meshes an SGB XAO with Gmsh so that no tetrahedron
is below the quality threshold, or raises :class:`MeshRepairError` carrying the quality report.  One
pipeline serves every sizing recipe: a recipe (:class:`FeatureRecipe`, :class:`GlobalSizeRecipe`,
:class:`UniformBoundaryRecipe`) decides only the initial element sizes, while repair
(:class:`RepairPolicy`, :class:`Budgets`), the threshold check and the report are shared.
:class:`MesherOptions` sets the Gmsh algorithms and the order of the delivered mesh; the audit always
measures the linear mesh.  The module reads only the sidecar JSON (``scgsim.sgb.sidecar.v2``) and the XAO;
it imports no SGB, AEDT or Palace code.  It trusts that the two files are the pair one SGB build wrote and
does not check them against each other.

Importing the package does not import Gmsh: every Gmsh call runs in a worker process.
"""

from .entry import MeshResult, mesh_sgb
from .errors import MeshingError, MeshRepairError, SidecarError
from .mesher import DELAUNAY, HXT, MesherOptions
from .recipes import (
    FeatureRecipe,
    GlobalSizeRecipe,
    SizingRecipe,
    UniformBoundaryRecipe,
)
from .repair import Budgets, RepairPolicy

__all__ = [
    "DELAUNAY",
    "HXT",
    "Budgets",
    "FeatureRecipe",
    "GlobalSizeRecipe",
    "MeshRepairError",
    "MeshResult",
    "MesherOptions",
    "MeshingError",
    "RepairPolicy",
    "SidecarError",
    "SizingRecipe",
    "UniformBoundaryRecipe",
    "mesh_sgb",
]
