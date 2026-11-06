"""Lower SGB primary groups to the disjoint solver attributes Palace's config lists.

SGB's primary groups are source identities: they partition each dimension, but one group can carry
several Surface-EPR rows, and a Route A sheet row integrates a different side of each of its faces.
Palace integrates plain attributes, so the faces of a surface group are split by *row signature* (the
rows, with their sides, that integrate the face).  Faces with one signature form one attribute, which is the
common case: such a group keeps one attribute under its own name.  The attributes partition the faces, so
no Gmsh group overlaps another and nothing is counted twice.  This module reads only the sidecar JSON.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

# One row on one attribute: the row id and the side of the face it integrates (None: a single-group row).
RowSignature = tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class SurfacePart:
    """The faces of one primary surface group that share a row signature, as one solver attribute."""

    attribute: int
    name: str
    faces: tuple[str, ...]
    tags: tuple[int, ...]
    rows: RowSignature

    def to_payload(self) -> dict[str, Any]:
        """The JSON form carried in the group dict and to the Gmsh worker."""
        payload = asdict(self)
        payload["rows"] = [{"row": row, "side": side} for row, side in self.rows]
        return payload


def lower_surface_group(
    name: str,
    record: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    first_attribute: int,
) -> list[SurfacePart]:
    """Split one primary surface group into attributes numbered from ``first_attribute``.

    ``rows`` are the sidecar ``surface_epr_rows`` whose ``group`` is ``name``.  The parts are ordered by
    signature, so numbering is the same for every run of the same sidecar.
    """
    faces_by_signature: dict[RowSignature, list[tuple[str, int]]] = defaultdict(list)
    for face, tag in zip(record["ids"], record["native_tags"], strict=True):
        faces_by_signature[_signature(face, rows)].append((face, tag))
    signatures = sorted(
        faces_by_signature,
        key=lambda signature: [(row, side or "") for row, side in signature],
    )
    return [
        SurfacePart(
            attribute=first_attribute + index,
            name=name if len(signatures) == 1 else f"{name}#{index + 1}",
            faces=tuple(face for face, _ in faces_by_signature[signature]),
            tags=tuple(tag for _, tag in faces_by_signature[signature]),
            rows=signature,
        )
        for index, signature in enumerate(signatures)
    ]


def _signature(face: str, rows: Mapping[str, Mapping[str, Any]]) -> RowSignature:
    """The rows that integrate ``face`` with the side of the face each integrates."""
    found: list[tuple[str, str | None]] = []
    for row_id, row in rows.items():
        face_sides = row["face_sides"]
        if face_sides is None:
            found.append((row_id, None))
        elif face in face_sides:
            found.append((row_id, face_sides[face]))
    return tuple(sorted(found))
