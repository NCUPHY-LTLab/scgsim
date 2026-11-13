"""Type aliases and constants that the geometry records share: routes, interface kinds and their order, conductor roles, and the metadata keys of a geometry reference.

The records themselves are in the neighbouring modules.
"""

from __future__ import annotations

from os import PathLike
from typing import Literal

PathInput = str | PathLike[str]


RouteLiteral = Literal["A", "B", "C"]


InterfaceKindLiteral = Literal["MS", "MA", "SA", "AA", "MM", "SS"]


Coordinate = tuple[float, float]


PolygonRing = tuple[Coordinate, ...]


Vector3D = tuple[float, float, float]


ConductorPartRoleLiteral = Literal[
    "face_metal",
    "contact_pad",
    "bump_body",
    "airbridge_post",
    "airbridge_deck",
]


HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES = frozenset(
    ("bump_body", "airbridge_post", "airbridge_deck")
)


ConductorRepresentationLiteral = Literal[
    "material_volume",
    "cutout_boundary_shell",
    "surface_sheet",
]


CurveKindLiteral = Literal[
    "line_segment", "circular_arc", "interpolation_spline", "bspline"
]


CurveOrientationLiteral = Literal[1, -1]


SolverUseLiteral = Literal["solver_active"]


DEFAULT_INTERFACE_SOLVER_USE: dict[InterfaceKindLiteral, SolverUseLiteral] = {
    "MS": "solver_active",
    "MA": "solver_active",
    "SA": "solver_active",
    "AA": "solver_active",
    "MM": "solver_active",
    "SS": "solver_active",
}


INTERFACE_KIND_ORDER = ("MM", "SS", "AA", "MS", "MA", "SA")
