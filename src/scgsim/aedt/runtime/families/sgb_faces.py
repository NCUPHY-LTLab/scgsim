"""Match the faces AEDT reports for one imported STEP solid to the faces its guard planned, one to one.

AEDT keeps the per-solid OCC face order as its face-id order, but that is an observation (experiment E4),
not a promise, so the import never relies on it.  Each imported face is matched by geometry instead:

* **Every face.**  Area and vertex set.  Two faces of equal area are told apart by where their vertices
  are, so a guard record whose centre was swapped with another face's still finds no face.
* **Planar face.**  Also its centre of mass, which PyAEDT's ``GetFaceCenter`` reports.
* **Curved face.**  No centre: PyAEDT's ``face.center`` of a curved face is the vertex centroid, not a
  centre of mass, so the vertices alone identify it.

The tolerances are those of the AEDT readback (issue #67, decision 9), not the sidecar's ``tolerance_um``:
AEDT reports positions about 1e-6 um and vertices up to about 1e-4 um away from the exact STEP.  The
coordinate tolerance is ``1e-3 * grid_um`` (:func:`import_tolerance_um`), three orders below the smallest
feature SGB draws.  The area tolerance is ``1e-9 * area + perimeter * coordinate tolerance``: the relative
part covers floating-point noise, and the perimeter part bounds the area a boundary shifted by the
coordinate tolerance can add or remove.  A guard face with no candidate or with two candidates inside
tolerance, or an imported face claimed by two guard faces, fails: the importer never guesses which face is
which.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

RELATIVE_AREA_TOLERANCE = 1e-9
IMPORT_TOLERANCE_PER_GRID = 1e-3


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

    ``guard_faces`` are the solid's ``faces`` records from the STEP guard; ``tolerance_um`` is the
    readback's coordinate tolerance (:func:`import_tolerance_um`).
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


def import_tolerance_um(grid_um: float) -> float:
    """The coordinate tolerance of the AEDT readback of a model on ``grid_um`` (decision 9)."""
    if not math.isfinite(grid_um) or grid_um <= 0:
        raise ValueError("grid_um must be finite and positive")
    return IMPORT_TOLERANCE_PER_GRID * grid_um


def area_tolerance_um2(
    area_um2: float, perimeter_um: float, tolerance_um: float
) -> float:
    """The area difference two readings of the same planned face may show (decision 9)."""
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
    if not _same_vertex_set(planned["vertices_um"], face.vertices_um, tolerance_um):
        return False
    return planned["curved"] or (
        math.dist(planned["centre_um"], face.centre_um) <= tolerance_um
    )


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
