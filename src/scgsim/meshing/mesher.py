"""The Gmsh settings that do not depend on the sizing recipe: algorithms and the order of the delivered mesh.

Owns :class:`MesherOptions`.  It imports no Gmsh, so both sides of the process boundary read it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Gmsh Mesh.Algorithm3D values.
DELAUNAY = 1
HXT = 10


@dataclass(frozen=True)
class MesherOptions:
    """How Gmsh meshes, whatever the sizes.

    The quality audit, the threshold and the repair loop all see the linear mesh.  When
    ``geometry_order`` is above 1, the worker raises the order of the mesh the loop accepted and writes
    that one; ``high_order_optimize`` then moves the extra nodes of curved elements to keep them valid.
    """

    # Gmsh's surface algorithm, left at its default when None.
    algorithm_2d: int | None = None
    # 3D algorithm.  With the same size fields, Delaunay meshed the 1, 2 and 4 mm chips 9 to 10 times faster
    # than HXT (4 mm: 11 s against 115 s; HXT spends 98 % of its time in its optimiser, on one thread) and
    # stayed clean at 0.01 on all seven benchmark cases, with about 1.5 times the tetrahedra.  HXT is the
    # other choice (and the retry when the first fails).
    algorithm_3d: int = DELAUNAY
    # Order of the delivered mesh: 1 is linear, 2 and above are curved elements.
    geometry_order: int = 1
    # Gmsh's "HighOrder" optimisation after raising the order; it has no effect on a linear mesh.
    high_order_optimize: bool = True

    def __post_init__(self) -> None:
        """Reject an order below 1."""
        if self.geometry_order < 1:
            raise ValueError(
                f"geometry_order must be 1 or more, got {self.geometry_order!r}."
            )
