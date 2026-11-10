"""Match the faces AEDT reports for one imported STEP solid to the faces its guard planned, one to one.

AEDT keeps the per-solid OCC face order as its face-id order, but that is an observation (experiment E4),
not a promise, so the import never relies on it.  Each imported face is matched by geometry instead:

* **Planar face.**  Area and centre of mass.  PyAEDT reports a planar face's centre of mass.
* **Curved face.**  Area and vertex set.  PyAEDT's ``face.center`` of a curved face is the vertex centroid,
  not a centre of mass, so the vertices identify it and no centre is compared.

The coordinate tolerance is the sidecar's ``tolerance_um``.  The area tolerance is
``1e-9 * area + perimeter * tolerance_um`` (issue #67, decision 3): the relative part covers
floating-point noise, and the perimeter part bounds the area a boundary shifted by ``tolerance_um`` can
add or remove.  A guard face with no candidate or with two candidates inside tolerance, or an imported
face claimed by two guard faces, fails: the importer never guesses which face is which.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

RELATIVE_AREA_TOLERANCE = 1e-9


class FaceMatchError(RuntimeError):
    """The imported faces of a solid do not correspond one to one to its guarded faces."""


@dataclass(frozen=True)
class ImportedFace:
    """One face of an imported object as AEDT reports it, in micrometres."""

    face_id: int
    area_um2: float
    centre_um: tuple[float, float, float]
    vertices_um: tuple[tuple[float, float, float], ...]
    perimeter_um: float
    planar: bool


def match_solid_faces(
    object_name: str,
    guard_faces: Sequence[Mapping[str, Any]],
    imported: Sequence[ImportedFace],
    tolerance_um: float,
) -> dict[str, int]:
    """``{planned face id: AEDT face id}`` for one imported solid, or :class:`FaceMatchError`.

    ``guard_faces`` are the solid's ``faces`` records from the STEP guard.
    """
    if not math.isfinite(tolerance_um) or tolerance_um <= 0:
        raise ValueError("tolerance_um must be finite and positive")
    if len(imported) != len(guard_faces):
        raise FaceMatchError(
            f"{object_name}: AEDT reports {len(imported)} faces, the guard planned {len(guard_faces)}"
        )
    matched: dict[str, int] = {}
    owner: dict[int, str] = {}
    for planned in guard_faces:
        face_id = _only_candidate(object_name, planned, imported, tolerance_um)
        if face_id in owner:
            raise FaceMatchError(
                f"{object_name}: AEDT face {face_id} matches both {owner[face_id]} and {planned['face']}"
            )
        owner[face_id] = planned["face"]
        matched[planned["face"]] = face_id
    return matched


def area_tolerance_um2(
    area_um2: float, perimeter_um: float, tolerance_um: float
) -> float:
    """The area difference two faces of the same planned face may show (decision 3)."""
    return RELATIVE_AREA_TOLERANCE * abs(area_um2) + perimeter_um * tolerance_um


def _only_candidate(
    object_name: str,
    planned: Mapping[str, Any],
    imported: Sequence[ImportedFace],
    tolerance_um: float,
) -> int:
    candidates = [
        face.face_id for face in imported if _same_face(planned, face, tolerance_um)
    ]
    if len(candidates) != 1:
        found = "no AEDT face" if not candidates else f"AEDT faces {candidates}"
        raise FaceMatchError(
            f"{object_name}: {found} within tolerance of planned face {planned['face']} "
            f"(area {planned['area_um2']} um2, {'curved' if planned['curved'] else 'planar'})"
        )
    return candidates[0]


def _same_face(
    planned: Mapping[str, Any], face: ImportedFace, tolerance_um: float
) -> bool:
    if face.planar == planned["curved"]:
        return False
    limit = area_tolerance_um2(planned["area_um2"], face.perimeter_um, tolerance_um)
    if abs(face.area_um2 - planned["area_um2"]) > limit:
        return False
    if planned["curved"]:
        return _same_vertex_set(planned["vertices_um"], face.vertices_um, tolerance_um)
    return math.dist(planned["centre_um"], face.centre_um) <= tolerance_um


def _same_vertex_set(
    planned: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
    tolerance_um: float,
) -> bool:
    """Every planned vertex has its own observed vertex within tolerance, and there are no others."""
    if len(planned) != len(observed):
        return False
    unused = list(observed)
    for point in planned:
        near = [
            k
            for k, other in enumerate(unused)
            if math.dist(point, other) <= tolerance_um
        ]
        if len(near) != 1:
            return False
        unused.pop(near[0])
    return True
