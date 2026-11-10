"""The durable import guard of an AEDT STEP: what an importer checks every time, written next to the STEP.

The STEP alone cannot say which planned face an imported face is: AEDT keeps the per-solid OCC face order as
its face-id order, but that is an observation (experiment E4), not a documented promise.  So the exporter
writes, beside the STEP, a guard file the importer reads again before it assigns anything:

* **Binding.**  The SHA-256 of the STEP, of the BREP or XAO it was written from and of the sidecar, the
  sidecar's model identity digest and the length unit.  :func:`read_step_guard` re-hashes the files and
  refuses a guard whose files changed, so a STEP is never paired with another model's sidecar.
* **Solids.**  Per exported solid its AEDT name and its faces in OCC solid face order, each with the planned
  face id, area, centre of mass, vertex positions, vertex centroid and whether the face is curved.  A planar
  face is identified by centre, area and vertices; a curved one by its vertices, because PyAEDT reports a
  vertex centroid, not a centre of mass, for a curved face.  Both centres are kept so neither stands in for
  the other.
* **Free faces.**  Faces of the model that bound no exported solid (a Route A sheet with no solid on either
  side, a port sheet that is not on a support plane).  STEP cannot carry them at micrometre scale (E4), so
  the guard keeps their loops as read from the written BREP -- lines and three-point circular arcs -- for the
  importer to rebuild them from SGB output, never from the layout source.

The guard is plain JSON: keys sorted, numbers rounded to twelve significant digits like the sidecar, file
paths relative to the guard's folder, no timestamps.  Reading it needs no OCP.  :func:`read_step_guard`
checks the whole structure and then the solids and free faces against the bound sidecar
(:func:`guard_mismatches`), down to each face record being the record of the face its id names (its
vertices and area are that planned face's, from the sidecar's ``expected`` block), and each free-face loop
being connected.  The exporter runs the same check before it writes anything, so it never writes a guard the
reader would refuse.

Two values have no planned counterpart in the sidecar, so this check cannot bind them: a planar face's
centre of mass, and the shape between the ends of a free-face edge (line or arc, and an arc's mid point).
So the guard is trusted as the exporter's output.  A consumer reads the guard of the export it ran, and
binds the guard's own bytes when it keeps it across steps (the AEDT importer of #67 binds all four files by
SHA-256).  An importer matches a planar face by centre, area *and* vertices, so a centre never decides a
face on its own.  A guard edited by hand in those two values, with every hash left valid, is not detected
here.

The guard of ``model.step`` is ``model.step.guard.json``: the STEP's full file name is kept, so ``model.step``
and ``model.stp`` in one folder have separate guards.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    """Where the guard of ``step_path`` lives: next to it, ``<STEP file name>.guard.json``."""
    return step_path.with_name(step_path.name + GUARD_SUFFIX)


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
    """The guard in ``path``, after checking it is well formed, its files are unchanged and it matches its sidecar.

    Raises :class:`StepGuardError` when the schema or unit is not the one written here, a key or value is
    missing or of the wrong kind, a bound file is missing or its bytes changed, the sidecar's model identity is
    not the one the guard was written for, or the solids and free faces disagree with the sidecar
    (:func:`guard_mismatches`).
    """
    path = Path(path)
    guard = _load_guard(path)
    for role, record in sorted(guard["files"].items()):
        _check_bound_file(path, role, record)
    sidecar = _load_bound_sidecar(path, guard)
    if sidecar.get("model") != guard["model"] or sidecar.get("route") != guard["route"]:
        raise StepGuardError(
            f"{path}: the sidecar's model {sidecar.get('model')!r} (route {sidecar.get('route')!r}) is not "
            f"the guarded {guard['model']!r} (route {guard['route']!r})"
        )
    try:
        mismatches = guard_mismatches(guard["solids"], guard["free_faces"], sidecar)
    except (KeyError, IndexError, TypeError) as error:
        raise StepGuardError(
            f"{path}: the sidecar lacks a record the guard needs: {error!r}"
        ) from error
    if mismatches:
        raise StepGuardError(
            f"{path} does not match its sidecar: " + "; ".join(mismatches)
        )
    return guard


def guard_mismatches(
    solids: Sequence[SolidGuard],
    free_faces: Sequence[FreeFace],
    sidecar: dict[str, Any],
) -> list[str]:
    """How the guarded solids and free faces disagree with the sidecar they were exported from; empty when they agree.

    The solids must be distinct planned solids in tag order, each with its sidecar ``aedt_name`` and exactly
    its planned faces (``expected.solid_faces``), each face listed once and flagged ``curved`` exactly when it
    is one of the sidecar's ``curved_wall_faces``.  The free faces must be exactly :func:`free_face_ids` of
    those solids, in tag order, each with its sidecar group.

    Each face record must also be the record *of the face its id names*: its vertices (for a free face, the
    ends of its loop edges) are that planned face's vertices and its area is that face's planned area, within
    the #67 decision 3 tolerances.  So a guard whose face ids were swapped is refused although every id is in
    its place.  Two faces the sidecar cannot tell apart that way (same planned vertices, areas within
    tolerance) in one solid, or among the free faces, are refused too, because their ids could not be bound.
    Each free-face loop must be connected: every edge ends where the next one starts.

    The centre of mass and the shape of a free-face edge between its ends have no planned counterpart and
    are not bound here (see the module docstring); the importer matches a planar face by centre, area and
    vertices together.
    """
    tag_of = _tags(sidecar["binding"]["ids_by_tag"]["3"])
    unknown = [s["solid"] for s in solids if s["solid"] not in tag_of]
    if unknown:
        return [f"solids {unknown} are not planned solids of the sidecar"]
    tags = [tag_of[s["solid"]] for s in solids]
    mismatches = []
    if tags != sorted(set(tags)):
        mismatches.append("the solids are not distinct planned solids in tag order")
    planned = _PlannedFaces.of(sidecar)
    for solid in solids:
        mismatches += _solid_mismatches(solid, sidecar, planned)
    mismatches += _free_face_mismatches(
        free_faces, [s["solid"] for s in solids], sidecar, planned
    )
    return mismatches


def free_face_ids(sidecar: dict[str, Any], solid_ids: Sequence[str]) -> list[str]:
    """The faces of the model that bound none of ``solid_ids``, in tag order: the faces the guard keeps as loops."""
    tag_of = _tags(sidecar["binding"]["ids_by_tag"]["3"])
    solid_faces = sidecar["expected"]["solid_faces"]
    bounding = {tag for vid in solid_ids for tag in solid_faces[tag_of[vid] - 1]}
    face_ids = sidecar["binding"]["ids_by_tag"]["2"]
    return [fid for tag, fid in enumerate(face_ids, start=1) if tag not in bounding]


def _solid_mismatches(
    solid: SolidGuard, sidecar: dict[str, Any], planned_faces: _PlannedFaces
) -> list[str]:
    vid = solid["solid"]
    mismatches = []
    planned_name = sidecar["volumes"][vid]["aedt_name"]
    if solid["aedt_name"] != planned_name:
        mismatches.append(
            f"solid {vid} is named {solid['aedt_name']!r}, the sidecar names it {planned_name!r}"
        )
    face_ids = sidecar["binding"]["ids_by_tag"]["2"]
    tag = _tags(sidecar["binding"]["ids_by_tag"]["3"])[vid]
    planned = {face_ids[t - 1] for t in sidecar["expected"]["solid_faces"][tag - 1]}
    listed = [f["face"] for f in solid["faces"]]
    if len(set(listed)) != len(listed) or set(listed) != planned:
        return mismatches + [
            f"solid {vid} lists faces {sorted(listed)}, the sidecar plans {sorted(planned)}, each once"
        ]
    curved = set(sidecar["curved_wall_faces"])
    misflagged = sorted(
        f["face"] for f in solid["faces"] if f["curved"] != (f["face"] in curved)
    )
    if misflagged:
        mismatches.append(f"solid {vid} flags curved wrongly on faces {misflagged}")
    for face in solid["faces"]:
        mismatches += planned_faces.binding_mismatches(
            f"face {face['face']} of solid {vid}",
            face["face"],
            face["vertices_um"],
            face["area_um2"],
            each_vertex_once=True,
        )
        if not _within(
            face["vertex_centroid_um"],
            _mean(face["vertices_um"]),
            planned_faces.tolerance_um,
        ):
            mismatches.append(
                f"face {face['face']} of solid {vid}: the vertex centroid is not the mean of its vertices"
            )
    mismatches += planned_faces.indistinct(f"solid {vid}", listed)
    return mismatches


def _free_face_mismatches(
    free_faces: Sequence[FreeFace],
    solid_ids: Sequence[str],
    sidecar: dict[str, Any],
    planned_faces: _PlannedFaces,
) -> list[str]:
    expected = free_face_ids(sidecar, solid_ids)
    listed = [f["face"] for f in free_faces]
    if listed != expected:
        return [
            f"the free faces are {listed}, the faces bounding no guarded solid are {expected}"
        ]
    planned = sidecar["faces"]
    mismatches = [
        f"free face {f['face']} has group {f['group']!r}, the sidecar gives {planned[f['face']]['group']!r}"
        for f in free_faces
        if f["group"] != planned[f["face"]]["group"]
    ]
    for face in free_faces:
        # the ends of the loop edges are the face's vertices; an arc's mid point is not a vertex
        ends = [segment["points_um"][0] for loop in face["loops"] for segment in loop]
        mismatches += planned_faces.binding_mismatches(
            f"free face {face['face']}",
            face["face"],
            ends,
            face["area_um2"],
            each_vertex_once=False,  # a hole touching the outer loop passes one vertex twice
        )
        mismatches += _loop_gaps(
            f"free face {face['face']}", face["loops"], planned_faces.tolerance_um
        )
    return mismatches + planned_faces.indistinct("the free faces", listed)


def _loop_gaps(
    where: str, loops: Sequence[Sequence[LoopSegment]], tolerance_um: float
) -> list[str]:
    """Each place where a loop of ``where`` breaks: an edge that does not end where the next one starts."""
    return [
        f"{where}: loop {k} edge {i} ends at {segment['points_um'][-1]}, "
        f"the next edge starts at {following['points_um'][0]}"
        for k, loop in enumerate(loops)
        for i, (segment, following) in enumerate(
            zip(loop, [*loop[1:], *loop[:1]], strict=True)
        )
        if not _within(
            segment["points_um"][-1], following["points_um"][0], tolerance_um
        )
    ]


@dataclass(frozen=True)
class _PlannedFaces:
    """The planned vertices and area of every face, from the sidecar's ``expected`` block, to bind face records to."""

    tag_of: dict[str, int]  # planned face id -> face tag
    face_vertices: list[frozenset[int]]  # per face tag - 1: its vertex tags
    vertex_positions: list[list[float]]  # per vertex tag - 1
    areas: list[float]  # per face tag - 1
    chord_perimeters: list[float]  # per face tag - 1: the sum of its edges' chords
    tolerance_um: float

    @classmethod
    def of(cls, sidecar: dict[str, Any]) -> _PlannedFaces:
        expected = sidecar["expected"]
        positions = expected["vertices_um"]
        ends = expected["edge_vertices"]
        chords = [math.dist(positions[a - 1], positions[b - 1]) for a, b in ends]
        return cls(
            tag_of=_tags(sidecar["binding"]["ids_by_tag"]["2"]),
            face_vertices=[
                frozenset(v for e in edges for v in ends[e - 1])
                for edges in expected["face_edges"]
            ],
            vertex_positions=positions,
            areas=expected["face_area_um2"],
            chord_perimeters=[
                sum(chords[e - 1] for e in edges) for edges in expected["face_edges"]
            ],
            tolerance_um=sidecar["tolerance_um"],
        )

    def binding_mismatches(
        self,
        where: str,
        face_id: str,
        vertices: list[list[float]],
        area: float,
        *,
        each_vertex_once: bool,
    ) -> list[str]:
        """Why the record at ``where`` is not the record of planned face ``face_id``; empty when it is."""
        tag = self.tag_of[face_id]
        planned = self.face_vertices[tag - 1]
        matched = [self._vertex_tag(point, planned) for point in vertices]
        same = None not in matched and set(matched) == planned
        if each_vertex_once:
            same = same and len(matched) == len(planned)
        mismatches = []
        if not same:
            mismatches.append(
                f"{where}: the recorded vertices are not those of planned face {face_id}"
            )
        if abs(area - self.areas[tag - 1]) > self._area_tolerance(tag):
            mismatches.append(
                f"{where}: the recorded area {area} is not the planned {self.areas[tag - 1]} of {face_id}"
            )
        return mismatches

    def indistinct(self, where: str, face_ids: Sequence[str]) -> list[str]:
        """Pairs among ``face_ids`` with the same planned vertices and areas within tolerance, as refusals."""
        tags = sorted(self.tag_of[fid] for fid in face_ids)
        return [
            f"{where}: faces {self._id(a)} and {self._id(b)} have the same planned vertices and area, "
            "so the guard cannot bind their ids"
            for i, a in enumerate(tags)
            for b in tags[i + 1 :]
            if self.face_vertices[a - 1] == self.face_vertices[b - 1]
            and abs(self.areas[a - 1] - self.areas[b - 1])
            <= max(self._area_tolerance(a), self._area_tolerance(b))
        ]

    def _vertex_tag(self, point: list[float], candidates: frozenset[int]) -> int | None:
        # the vertices of a planned face lie far apart compared with the tolerance (an edge shorter than ten
        # tolerances is refused), so in practice at most one matches
        return next(
            (
                v
                for v in sorted(candidates)
                if _within(point, self.vertex_positions[v - 1], self.tolerance_um)
            ),
            None,
        )

    def _area_tolerance(self, tag: int) -> float:
        # #67 decision 3: relative 1e-9 plus perimeter x tolerance_um.  The chord perimeter is at most the true
        # perimeter, so this never accepts more than the decision does; the exporter already holds every area
        # to relative 1e-9 of the plan, so a written guard always passes.
        return (
            1e-9 * abs(self.areas[tag - 1])
            + self.chord_perimeters[tag - 1] * self.tolerance_um
        )

    def _id(self, tag: int) -> str:
        return next(fid for fid, t in self.tag_of.items() if t == tag)


def _within(point: Sequence[float], target: Sequence[float], tolerance: float) -> bool:
    """Every coordinate of ``point`` within ``tolerance`` of ``target``."""
    return all(abs(a - b) <= tolerance for a, b in zip(point, target, strict=True))


def _mean(points: list[list[float]]) -> list[float]:
    return [sum(p[k] for p in points) / len(points) for k in range(3)]


def _tags(ids: list[str]) -> dict[str, int]:
    """Planned id to native tag (1 + position in ``ids_by_tag``)."""
    return {planned_id: tag for tag, planned_id in enumerate(ids, start=1)}


def _load_guard(path: Path) -> StepGuard:
    """The parsed guard of ``path`` once its schema, unit and complete structure are the ones written here."""
    try:
        guard = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StepGuardError(f"{path} is not a readable guard: {error}") from error
    if not isinstance(guard, dict) or guard.get("schema") != GUARD_SCHEMA:
        raise StepGuardError(f"{path} is not a {GUARD_SCHEMA} guard")
    if guard.get("unit") != "UM":
        raise StepGuardError(f"{path} has unit {guard.get('unit')!r}, not 'UM'")
    problems = _structure_problems(guard)
    if problems:
        raise StepGuardError(f"{path} is malformed: " + "; ".join(problems))
    return cast(StepGuard, guard)


def _load_bound_sidecar(path: Path, guard: StepGuard) -> dict[str, Any]:
    try:
        sidecar = read_sidecar(path.parent / guard["files"]["sidecar"]["path"])
    except ValueError as error:
        raise StepGuardError(
            f"{path}: the bound sidecar is not readable JSON: {error}"
        ) from error
    if not isinstance(sidecar, dict):
        raise StepGuardError(f"{path}: the bound sidecar is not a JSON object")
    return sidecar


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


# ---------------------------------------------------------------- the structure of a guard, key by key
# Exact key sets, so a dropped or added key is refused rather than read as a default.


def _structure_problems(guard: dict[str, Any]) -> list[str]:
    if set(guard) != StepGuard.__required_keys__:
        return [
            f"the guard has keys {sorted(guard)}, expected {sorted(StepGuard.__required_keys__)}"
        ]
    problems = []
    if not isinstance(guard["route"], str):
        problems.append("route is not a string")
    model = guard["model"]
    if not isinstance(model, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in model.items()
    ):
        problems.append("model is not a mapping of strings")
    files = guard["files"]
    if not isinstance(files, dict) or set(files) != {"step", "geometry", "sidecar"}:
        problems.append("files must bind exactly a step, geometry and sidecar")
    else:
        problems += [
            f"files.{role} is not a path and a SHA-256"
            for role, record in sorted(files.items())
            if not _is_record(record, GuardedFile.__required_keys__)
            or not all(isinstance(record[k], str) for k in ("path", "sha256"))
        ]
    problems += _list_problems("solids", guard["solids"], _solid_problems)
    problems += _list_problems(
        "free_faces", guard["free_faces"], _free_face_problems, allow_empty=True
    )
    return problems


def _list_problems(
    where: str,
    items: Any,
    check: Callable[[str, Any], list[str]],
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(items, list) or (not items and not allow_empty):
        return [f"{where} is not a {'' if allow_empty else 'non-empty '}list"]
    return [p for i, item in enumerate(items) for p in check(f"{where}[{i}]", item)]


def _solid_problems(where: str, solid: Any) -> list[str]:
    if not _is_record(solid, SolidGuard.__required_keys__):
        return [
            f"{where} does not have exactly the keys {sorted(SolidGuard.__required_keys__)}"
        ]
    problems = [
        f"{where}.{key} is not a string"
        for key in ("solid", "aedt_name")
        if not isinstance(solid[key], str)
    ]
    return problems + _list_problems(f"{where}.faces", solid["faces"], _face_problems)


def _face_problems(where: str, face: Any) -> list[str]:
    if not _is_record(face, FaceGuard.__required_keys__):
        return [
            f"{where} does not have exactly the keys {sorted(FaceGuard.__required_keys__)}"
        ]
    vertices = face["vertices_um"]
    checks = {
        "face": isinstance(face["face"], str),
        "area_um2": _is_positive(face["area_um2"]),
        "centre_um": _is_point(face["centre_um"]),
        "vertex_centroid_um": _is_point(face["vertex_centroid_um"]),
        "vertices_um": isinstance(vertices, list)
        and bool(vertices)
        and all(map(_is_point, vertices)),
        "curved": isinstance(face["curved"], bool),
    }
    return [f"{where}.{key} is malformed" for key, ok in checks.items() if not ok]


def _free_face_problems(where: str, face: Any) -> list[str]:
    if not _is_record(face, FreeFace.__required_keys__):
        return [
            f"{where} does not have exactly the keys {sorted(FreeFace.__required_keys__)}"
        ]
    checks = {
        "face": isinstance(face["face"], str),
        "group": face["group"] is None or isinstance(face["group"], str),
        "area_um2": _is_positive(face["area_um2"]),
        "normal": _is_point(face["normal"]),
    }
    problems = [f"{where}.{key} is malformed" for key, ok in checks.items() if not ok]
    return problems + _list_problems(f"{where}.loops", face["loops"], _loop_problems)


def _loop_problems(where: str, loop: Any) -> list[str]:
    return _list_problems(where, loop, _segment_problems)


_SEGMENT_POINTS = {
    "line": 2,
    "arc": 3,
}  # a line by its ends, an arc by start, mid and end


def _segment_problems(where: str, segment: Any) -> list[str]:
    if not _is_record(segment, LoopSegment.__required_keys__):
        return [
            f"{where} does not have exactly the keys {sorted(LoopSegment.__required_keys__)}"
        ]
    count = (
        _SEGMENT_POINTS.get(segment["kind"])
        if isinstance(segment["kind"], str)
        else None
    )
    points = segment["points_um"]
    if (
        count is None
        or not isinstance(points, list)
        or len(points) != count
        or not all(map(_is_point, points))
    ):
        return [f"{where} is not a two-point line or a three-point arc"]
    return []


def _is_record(value: Any, keys: frozenset[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_positive(value: Any) -> bool:
    return _is_number(value) and value > 0


def _is_point(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(map(_is_number, value))
