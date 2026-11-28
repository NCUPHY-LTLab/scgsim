"""The Gmsh-default recipe: no size of SCGSim's own, so Gmsh meshes the model with its own defaults."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from ..features import SidecarFeatures
from ..sizing import BoundingBox, SizeRequests

# Gmsh's own default of Mesh.MeshSizeMax (Gmsh 4.15.2): no maximum element size.  The worker always sets
# that option, so this value leaves it as Gmsh has it.
GMSH_DEFAULT_MAX_SIZE = 1e22


@dataclass(frozen=True)
class GmshDefaultRecipe:
    """No size field and no maximum size: Gmsh's own 3D meshing of the SGB model.

    The one report-only strategy, and never the default.  :func:`~scgsim.meshing.mesh_sgb` meshes it once,
    without repair, and delivers the mesh whatever its quality: the report records the minSICN and the
    count below the threshold, but the threshold refuses nothing.  It gives no quality guarantee.  The
    tetrahedron and time budgets still apply.
    """

    name: ClassVar[str] = "gmsh-default"
    version: ClassVar[str] = "scgsim.meshing.gmsh-default/1"

    def size_requests(
        self, features: SidecarFeatures, bounding_box: BoundingBox
    ) -> SizeRequests:
        """Gmsh's default maximum size alone; ``features`` and the extent are not read.

        The gradation grades nothing: no curve is sized and no repair round runs.
        """
        return SizeRequests(GMSH_DEFAULT_MAX_SIZE, 1.0, MappingProxyType({}))
