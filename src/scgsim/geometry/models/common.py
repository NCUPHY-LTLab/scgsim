"""Canonical geometry records. Keep source facts distinct from compiler records; reexports share these exact definitions."""

from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import Literal, cast

PathInput = str | PathLike[str]


RUN_METADATA_DIR = "metadata"


SEMANTIC_GEOMETRY_METADATA_DIR = "semantic_geometry"


RouteLiteral = Literal["A", "B", "C"]


class GeometryRoute(str, Enum):
    """Physical construction choices with unchanged serialized route codes."""

    THIN_FILM = "A"
    PEC_SHELL = "B"
    RETAINED_SOLID = "C"

    def __str__(self) -> str:
        return self.value

    @property
    def display_name(self) -> str:
        return {
            "A": "Thin film",
            "B": "PEC shell",
            "C": "Retained solid",
        }[self.value]


def _normalize_geometry_route(value: str | GeometryRoute) -> RouteLiteral:
    """Normalize caller input while keeping persisted A/B/C codes unchanged."""
    if isinstance(value, str):
        name = value.strip().upper().replace("-", "_")
        names = {member.name: member.value for member in GeometryRoute}
        code = names.get(name, name)
        if code in {"A", "B", "C"}:
            return cast(RouteLiteral, code)
    raise ValueError(
        "geometry route must be thin_film (A), pec_shell (B), "
        "or retained_solid (C)"
    )



InterfaceKindLiteral = Literal["MS", "MA", "SA", "AA", "MM", "SS"]


DimensionLiteral = Literal[1, 2, 3]


SurfaceOrientationLiteral = Literal["forward", "reversed"]


Coordinate = tuple[float, float]


PolygonRing = tuple[Coordinate, ...]


Vector3D = tuple[float, float, float]


GmshDimTag = tuple[int, int]


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


ROUTE_ALLOWED_REPRESENTATIONS: dict[
    RouteLiteral,
    frozenset[ConductorRepresentationLiteral],
] = {
    "A": frozenset(("surface_sheet", "cutout_boundary_shell")),
    "B": frozenset(("cutout_boundary_shell",)),
    "C": frozenset(("material_volume",)),
}


CurveKindLiteral = Literal[
    "line_segment", "circular_arc", "interpolation_spline", "bspline"
]


CurveOrientationLiteral = Literal[1, -1]


SurfaceLoopRoleLiteral = Literal["outer", "hole"]


TagSourceKindLiteral = Literal["surface", "volume"]


SolverUseLiteral = Literal["solver_active"]


DEFAULT_INTERFACE_SOLVER_USE: dict[InterfaceKindLiteral, SolverUseLiteral] = {
    "MS": "solver_active",
    "MA": "solver_active",
    "SA": "solver_active",
    "AA": "solver_active",
    "MM": "solver_active",
    "SS": "solver_active",
}
