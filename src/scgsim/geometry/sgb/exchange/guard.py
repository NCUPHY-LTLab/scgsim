"""The durable import guard of an AEDT STEP: what an importer checks every time, written next to the STEP.

The STEP alone cannot say which planned face an imported face is: AEDT keeps the per-solid OCC face order as
its face-id order, but that is an observation (experiment E4), not a documented promise.  So the exporter
writes, beside the STEP, a guard file the importer reads again before it assigns anything:

* **Binding.**  The SHA-256 of the STEP, of the BREP or XAO it was written from and of the sidecar, the
  sidecar's model identity digest and the length unit.  :func:`read_step_guard` re-hashes the files and
  refuses a guard whose files changed, so a STEP is never paired with another model's sidecar.
* **Solids.**  Per exported solid its AEDT name and its faces in OCC solid face order, each with the planned
  face id, area, centre of mass, vertex positions, vertex centroid and whether the face is curved.  A planar
  face is identified by centre and area; a curved one by its vertices, because PyAEDT reports a vertex
  centroid, not a centre of mass, for a curved face.  Both centres are kept so neither stands in for the
  other.
* **Free faces.**  Faces of the model that bound no exported solid (a Route A sheet with no solid on either
  side, a port sheet that is not on a support plane).  STEP cannot carry them at micrometre scale (E4), so
  the guard keeps their loops as read from the written BREP -- lines and three-point circular arcs -- for the
  importer to rebuild them from SGB output, never from the layout source.

The guard is plain JSON: keys sorted, numbers rounded to twelve significant digits like the sidecar, file
paths relative to the guard's folder, no timestamps.  Reading it needs no OCP.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypedDict, cast

from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.sidecar.files import read_sidecar

GUARD_SCHEMA = "scgsim.sgb.aedt-step-guard.v1"
GUARD_SUFFIX = ".guard.json"


class StepGuardError(SgbError):
    """A guard file that is malformed or no longer matches the files it binds."""


class GuardedFile(TypedDict):
    """One file the guard binds: its path relative to the guard's folder and its SHA-256."""

    path: str
    sha256: str


class FaceGuard(TypedDict):
    """One face of an exported solid, in OCC solid face order (the AEDT face-id order)."""

    face: str
    area_um2: float
    centre_um: list[float]
    vertex_centroid_um: list[float]
    vertices_um: list[list[float]]
    curved: bool


class SolidGuard(TypedDict):
    """One exported solid: its planned id, its AEDT name and its faces."""

    solid: str
    aedt_name: str
    faces: list[FaceGuard]


class LoopSegment(TypedDict):
    """One edge of a free face's loop in loop order: a ``line`` (two points) or an ``arc`` (start, mid, end)."""

    kind: str
    points_um: list[list[float]]


class FreeFace(TypedDict):
    """A face that bounds no exported solid, with what an importer needs to rebuild it as a sheet."""

    face: str
    group: str | None
    area_um2: float
    normal: list[float]
    loops: list[list[LoopSegment]]  # the outer loop first, then the holes


class StepGuard(TypedDict):
    """The whole guard file."""

    schema: str
    unit: str
    route: str
    model: dict[str, str]
    files: dict[str, GuardedFile]  # ``step``, ``geometry``, ``sidecar``
    solids: list[SolidGuard]
    free_faces: list[FreeFace]


def guard_path_for(step_path: Path) -> Path:
    """Where the guard of ``step_path`` lives: next to it, ``<stem>.guard.json``."""
    return step_path.with_name(step_path.stem + GUARD_SUFFIX)


def guarded_file(path: Path, guard_dir: Path) -> GuardedFile:
    """The binding record of ``path`` as seen from a guard in ``guard_dir``."""
    return {
        "path": Path(os.path.relpath(path, guard_dir)).as_posix(),
        "sha256": file_sha256(path),
    }


def file_sha256(path: Path) -> str:
    """SHA-256 of the bytes of ``path``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_step_guard(path: Path, guard: StepGuard) -> None:
    """Write ``guard`` as deterministic JSON (sorted keys, no NaN)."""
    text = json.dumps(guard, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))


def read_step_guard(path: Path) -> StepGuard:
    """The guard in ``path``, after checking that every file it binds is unchanged and belongs to it.

    Raises :class:`StepGuardError` when the schema or unit is not the one written here, a bound file is
    missing or its bytes changed, or the sidecar's model identity is not the one the guard was written for.
    """
    path = Path(path)
    try:
        guard = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StepGuardError(f"{path} is not a readable guard: {error}") from error
    if not isinstance(guard, dict) or guard.get("schema") != GUARD_SCHEMA:
        raise StepGuardError(f"{path} is not a {GUARD_SCHEMA} guard")
    if guard.get("unit") != "UM":
        raise StepGuardError(f"{path} has unit {guard.get('unit')!r}, not 'UM'")
    files = guard.get("files")
    if not isinstance(files, dict) or set(files) != {"step", "geometry", "sidecar"}:
        raise StepGuardError(f"{path} must bind exactly a step, geometry and sidecar")
    for role, record in sorted(files.items()):
        _check_bound_file(path, role, record)
    sidecar = read_sidecar(path.parent / files["sidecar"]["path"])
    if sidecar.get("model") != guard.get("model") or sidecar.get("route") != guard.get(
        "route"
    ):
        raise StepGuardError(
            f"{path}: the sidecar's model {sidecar.get('model')!r} (route {sidecar.get('route')!r}) is not "
            f"the guarded {guard.get('model')!r} (route {guard.get('route')!r})"
        )
    return cast(StepGuard, guard)


def _check_bound_file(guard_path: Path, role: str, record: Any) -> None:
    if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
        raise StepGuardError(f"{guard_path}: the {role} record is malformed")
    target = guard_path.parent / record["path"]
    if not target.is_file():
        raise StepGuardError(
            f"{guard_path}: the {role} file {record['path']} is missing"
        )
    observed = file_sha256(target)
    if observed != record["sha256"]:
        raise StepGuardError(
            f"{guard_path}: the {role} file {record['path']} changed "
            f"(sha256 {observed}, guarded {record['sha256']})"
        )
