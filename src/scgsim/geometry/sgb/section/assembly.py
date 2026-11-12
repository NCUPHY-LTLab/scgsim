"""The 2D assembly of the traces: the section's cells and regions, then its vertices, edges, faces and loops.

The traces cut the section ``[0, length] x [lowest level, highest level]`` into cells between consecutive
breakpoints along the cut (every crossing) and consecutive levels, so a row of cells is one slab of the 3D
model.  Cells that no trace separates form one region, and the traces on a region's boundary say which 3D volume
fills it; they must agree.  A region filled by an emitted volume is a face of the section; the others (an
excluded conductor, the outside) are not.  Nothing is decided here that the 3D model did not decide.

Three cut rules can only be read on the regions, and are checked here before any record is made: an end of the
cut inside the model, where the model's boundary does not close the section; a stretch of the cut outside the
model; and excluded regions that touch only at a point.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from scgsim.geometry.sgb import identity
from scgsim.geometry.sgb.assembly.records import Loop
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.labels import Domain, SheetOwner
from scgsim.geometry.sgb.section.frame import CutFrame
from scgsim.geometry.sgb.section.records import (
    SectionEdge,
    SectionFace,
    SectionVertex,
)
from scgsim.geometry.sgb.section.rules import (
    refuse_open_end,
    refuse_outside_stretch,
    refuse_point_contact,
)
from scgsim.geometry.sgb.section.traces import Traces

Cell = tuple[int, int]  # (column i, slab j)
OUTSIDE = -1
"""The label of a cell beyond the section, or of a region that is no face, in the point-contact rule."""


@dataclass
class Region:
    """Cells of the section that no trace separates, and the 3D volumes the traces around them see there."""

    cells: list[Cell] = field(default_factory=list)
    seen: set[str | None] = field(default_factory=set)
    volume: str | None = (
        None  # the emitted volume that fills the region; None when it is no face
    )


@dataclass
class SectionGrid:
    """The cells of the section: cell ``(i, j)`` spans ``us[i] .. us[i + 1]`` along the cut and slab ``j``."""

    us: list[int]
    zs: tuple[Fraction, ...]
    region_of: dict[Cell, int]
    regions: list[Region]
    problems: list[str] = field(default_factory=list)

    @property
    def columns(self) -> int:
        """The number of cells along the cut."""
        return len(self.us) - 1

    @property
    def slabs(self) -> int:
        """The number of cells up the stack."""
        return len(self.zs) - 1

    def region(self, cell: Cell) -> Region | None:
        """The region of a cell, ``None`` for a cell beyond the section."""
        index = self.region_of.get(cell)
        return None if index is None else self.regions[index]

    def label(self, cell: Cell) -> int:
        """The region index of a cell that is part of a face, else :data:`OUTSIDE`."""
        region = self.region(cell)
        return (
            OUTSIDE if region is None or region.volume is None else self.region_of[cell]
        )


@dataclass
class AssembledSection:
    """The records of a section and the grid they were read from."""

    grid: SectionGrid
    vertices: dict[str, SectionVertex]
    edges: dict[str, SectionEdge]
    faces: dict[str, SectionFace]
    face_regions: dict[str, int]  # face id -> the index of the region it is


def section_grid(frame: CutFrame, model: ConformalModel, traces: Traces) -> SectionGrid:
    """The cells of the section, joined into regions, each with the volume that fills it."""
    us = sorted(
        {0, frame.length}
        | {t.u0 for t in traces.horizontal}
        | {t.u1 for t in traces.horizontal}
        | {w.u for w in traces.walls}
    )
    zs = model.topology.levels
    column = {u: i for i, u in enumerate(us)}
    columns, slabs = len(us) - 1, len(zs) - 1
    level_cuts: set[Cell] = set()  # cell (i, j) is cut from the cell below it
    column_cuts: set[Cell] = set()  # cell (i, j) is cut from the cell before it
    seen: dict[Cell, set[str | None]] = defaultdict(set)
    faces = model.topology.faces
    for trace in traces.horizontal:
        face, j = faces[trace.face], trace.level
        for i in range(column[trace.u0], column[trace.u1]):
            level_cuts.add((i, j))
            seen[(i, j)].add(face.pos_volume)
            seen[(i, j - 1)].add(face.neg_volume)
    for wall in traces.walls:
        face, i = faces[wall.face], column[wall.u]
        before, after = (
            (face.neg_volume, face.pos_volume)
            if wall.positive_after
            else (face.pos_volume, face.neg_volume)
        )
        for j in range(*wall.levels):
            column_cuts.add((i, j))
            seen[(i - 1, j)].add(before)
            seen[(i, j)].add(after)
    region_of = _connect(columns, slabs, level_cuts, column_cuts)
    regions = [Region() for _ in range(max(region_of.values()) + 1)]
    for cell, index in sorted(region_of.items()):
        regions[index].cells.append(cell)
        regions[index].seen.update(seen.get(cell, ()))
    grid = SectionGrid(us, zs, region_of, regions)
    _fill_regions(grid, model)
    return grid


def _connect(
    columns: int, slabs: int, level_cuts: set[Cell], column_cuts: set[Cell]
) -> dict[Cell, int]:
    """Cell -> region index: the cells no trace separates, numbered in the order of their first cell."""
    parent = list(range(columns * slabs))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(columns):
        for j in range(slabs):
            if i + 1 < columns and (i + 1, j) not in column_cuts:
                join(i * slabs + j, (i + 1) * slabs + j)
            if j + 1 < slabs and (i, j + 1) not in level_cuts:
                join(i * slabs + j, i * slabs + j + 1)
    numbers: dict[int, int] = {}
    return {
        (i, j): numbers.setdefault(find(i * slabs + j), len(numbers))
        for i in range(columns)
        for j in range(slabs)
    }


def _fill_regions(grid: SectionGrid, model: ConformalModel) -> None:
    """The emitted volume of every region whose bounding traces all see that one volume in it."""
    emitted = model.routed.emitted_volumes
    for region in grid.regions:
        volumes = {v for v in region.seen if v is not None and v in emitted}
        if not volumes:
            continue
        if len(region.seen) == 1:
            region.volume = volumes.pop()
            continue
        i, j = region.cells[0]
        grid.problems.append(
            f"the region from cell ({grid.us[i]}, {grid.zs[j]}) is bounded by traces of different volumes "
            f"{sorted(map(str, region.seen))}"
        )


# ---------------------------------------------------------------------------- rules read on the regions


def check_regions(
    frame: CutFrame,
    model: ConformalModel,
    curves: Sequence[str],
    traces: Traces,
    grid: SectionGrid,
) -> None:
    """Refuse a cut whose ends do not lie on the model's boundary, that leaves the model, or that pinches it."""
    _check_ends(frame, model, curves, traces, grid)
    _check_stretches_inside(frame, model, curves, grid)
    _check_point_contacts(frame, model, grid)


def _check_ends(
    frame: CutFrame,
    model: ConformalModel,
    curves: Sequence[str],
    traces: Traces,
    grid: SectionGrid,
) -> None:
    """Each end of the cut lies on the model's boundary: only the outside lies beyond it, and walls close it."""
    faces, volumes = model.topology.faces, model.topology.volumes
    for u, i, beyond_is_before in (
        (0, 0, True),
        (frame.length, grid.columns - 1, False),
    ):
        at_end = [w for w in traces.walls if w.u == u]
        for wall in at_end:
            face = faces[wall.face]
            outer_positive = wall.positive_after == (not beyond_is_before)
            beyond = face.pos_volume if outer_positive else face.neg_volume
            if beyond is not None:
                span = (grid.zs[wall.levels[0]], grid.zs[wall.levels[1]])
                refuse_open_end(frame, model, curves, u, span, volumes[beyond].domain)
        if not at_end:
            refuse_open_end(frame, model, curves, u, (grid.zs[0], grid.zs[-1]), None)
        closed = {j for w in at_end for j in range(*w.levels)}
        for j in range(grid.slabs):
            region = grid.region((i, j))
            if region is not None and region.volume is not None and j not in closed:
                span = (grid.zs[j], grid.zs[j + 1])
                domain = volumes[region.volume].domain
                refuse_open_end(frame, model, curves, u, span, domain)


def _check_stretches_inside(
    frame: CutFrame, model: ConformalModel, curves: Sequence[str], grid: SectionGrid
) -> None:
    """No stretch of the cut lies outside the model at every level."""
    for i in range(grid.columns):
        outside = all(
            (region := grid.region((i, j))) is not None
            and region.volume is None
            and region.seen <= {None}
            for j in range(grid.slabs)
        )
        if outside:
            refuse_outside_stretch(frame, model, curves, grid.us[i], grid.us[i + 1])


def _check_point_contacts(
    frame: CutFrame, model: ConformalModel, grid: SectionGrid
) -> None:
    """At every corner of the cells, the faces round it, and what is no face, each lie in one run."""
    for i, u in enumerate(grid.us):
        for j, z in enumerate(grid.zs):
            ring = [(i, j), (i - 1, j), (i - 1, j - 1), (i, j - 1)]
            labels = [grid.label(cell) for cell in ring]
            runs = [label for k, label in enumerate(labels) if label != labels[k - 1]]
            repeated = [label for label in runs if runs.count(label) > 1]
            if not repeated:
                continue
            if repeated[0] == OUTSIDE:
                excluded = _excluded_domains(model, grid, ring)
                refuse_point_contact(frame, model, u, z, excluded)
            volume = grid.regions[repeated[0]].volume
            assert volume is not None
            itself = model.topology.volumes[volume].domain
            refuse_point_contact(frame, model, u, z, [], itself)


def _excluded_domains(
    model: ConformalModel, grid: SectionGrid, ring: list[Cell]
) -> list[Domain | None]:
    """What fills the cells of ``ring`` that are no face: conductor domains, and ``None`` for the outside."""
    found: set[Domain | None] = set()
    for cell in ring:
        if grid.label(cell) != OUTSIDE:
            continue
        region = grid.region(cell)
        seen = {None} if region is None else region.seen or {None}
        found.update(
            None if v is None else model.topology.volumes[v].domain for v in seen
        )
    return sorted(found, key=lambda d: (d is None, d.name if d else ""))


# ---------------------------------------------------------------------------- the records


@dataclass(frozen=True)
class _EdgeDraft:
    """An edge before the faces beside it have ids: its regions are region indices (``None``: beyond the section)."""

    id: str
    kind: Literal["horizontal", "vertical"]
    start: tuple[int, Fraction]
    end: tuple[int, Fraction]
    trace: str
    left: int | None
    right: int | None
    left_domain: Domain | None
    right_domain: Domain | None
    owner: SheetOwner | None
    bands: tuple[tuple[str, int], ...]
    curves: tuple[str, ...]


def assemble_section(
    model: ConformalModel, traces: Traces, grid: SectionGrid
) -> AssembledSection:
    """The vertices, edges and faces of a section whose grid passed the rules."""
    drafts = _edge_drafts(model, traces, grid)
    face_ids = _face_ids(grid, drafts)
    vertices: dict[str, SectionVertex] = {}
    edges: dict[str, SectionEdge] = {}
    for draft in drafts:
        v0, v1 = (_vertex(vertices, *point) for point in (draft.start, draft.end))
        edges[draft.id] = SectionEdge(
            draft.id,
            draft.kind,
            v0,
            v1,
            draft.trace,
            _group(model, draft.trace),
            None if draft.left is None else face_ids.get(draft.left),
            None if draft.right is None else face_ids.get(draft.right),
            draft.left_domain,
            draft.right_domain,
            draft.owner,
            draft.bands,
            draft.curves,
        )
    beside: dict[str, list[SectionEdge]] = defaultdict(list)
    for edge in edges.values():
        for face_id in {edge.left, edge.right}:
            if face_id is not None:
                beside[face_id].append(edge)
    faces = {
        face_id: _face(model, grid, index, face_id, beside[face_id], vertices)
        for index, face_id in sorted(face_ids.items(), key=lambda item: item[1])
    }
    return AssembledSection(
        grid,
        dict(sorted(vertices.items())),
        dict(sorted(edges.items())),
        faces,
        {face_id: index for index, face_id in face_ids.items()},
    )


def _edge_drafts(
    model: ConformalModel, traces: Traces, grid: SectionGrid
) -> list[_EdgeDraft]:
    """One edge per trace, with the region on each side; a side that holds two regions is a structural failure."""
    faces = model.topology.faces
    column = {u: i for i, u in enumerate(grid.us)}
    drafts = []
    for trace in traces.horizontal:
        face, j, z = faces[trace.face], trace.level, grid.zs[trace.level]
        span = range(column[trace.u0], column[trace.u1])
        drafts.append(
            _EdgeDraft(
                identity.digest("se", [trace.face, trace.u0, trace.u1, str(z), str(z)]),
                "horizontal",
                (trace.u0, z),
                (trace.u1, z),
                trace.face,
                _side(grid, [(i, j) for i in span], trace.face),
                _side(grid, [(i, j - 1) for i in span], trace.face),
                face.pos_domain,
                face.neg_domain,
                face.owner,
                face.bands,
                (),
            )
        )
    for wall in traces.walls:
        face, i = faces[wall.face], column[wall.u]
        z0, z1 = grid.zs[wall.levels[0]], grid.zs[wall.levels[1]]
        rows = range(*wall.levels)
        before, after = (
            (face.neg_domain, face.pos_domain)
            if wall.positive_after
            else (face.pos_domain, face.neg_domain)
        )
        drafts.append(
            _EdgeDraft(
                identity.digest("se", [wall.face, wall.u, wall.u, str(z0), str(z1)]),
                "vertical",
                (wall.u, z0),
                (wall.u, z1),
                wall.face,
                _side(grid, [(i - 1, j) for j in rows], wall.face),
                _side(grid, [(i, j) for j in rows], wall.face),
                before,
                after,
                None,
                (),
                wall.curves,
            )
        )
    return sorted(drafts, key=lambda d: d.id)


def _side(grid: SectionGrid, cells: list[Cell], trace: str) -> int | None:
    """The one region of the cells on one side of an edge (``None`` beyond the section)."""
    regions = {grid.region_of.get(cell) for cell in cells}
    if len(regions) != 1:
        grid.problems.append(
            f"the trace of {trace} has {len(regions)} regions on one side"
        )
    return min(regions, key=lambda r: -1 if r is None else r)


def _face_ids(grid: SectionGrid, drafts: list[_EdgeDraft]) -> dict[int, str]:
    """Region index -> face id of every region that is a face: its volume and the edges that bound it."""
    bounding: dict[int, list[str]] = defaultdict(list)
    for draft in drafts:
        for side in {draft.left, draft.right}:
            if side is not None:
                bounding[side].append(draft.id)
    return {
        index: identity.digest("sf", [region.volume, sorted(bounding[index])])
        for index, region in enumerate(grid.regions)
        if region.volume is not None
    }


def _vertex(vertices: dict[str, SectionVertex], u: int, z: Fraction) -> str:
    vertex_id = identity.digest("sv", [u, str(z)])
    vertices.setdefault(vertex_id, SectionVertex(vertex_id, u, z))
    return vertex_id


def _group(model: ConformalModel, trace: str) -> str:
    """The primary group of the 3D face an edge traces (every emitted face has one)."""
    group = model.routed.face_group[trace]
    assert group is not None, trace
    return group


def _face(  # noqa: PLR0913 - the model, the grid, the region and the records its loops are read from
    model: ConformalModel,
    grid: SectionGrid,
    index: int,
    face_id: str,
    beside: list[SectionEdge],
    vertices: dict[str, SectionVertex],
) -> SectionFace:
    """A face of the section: its loops with the face on their left, outer loop first, and its embedded edges."""
    region = grid.regions[index]
    assert region.volume is not None
    oriented: list[tuple[str, int]] = []
    embedded = []
    for edge in beside:
        if edge.left == face_id and edge.right == face_id:
            embedded.append(edge.id)
        elif edge.left == face_id:
            oriented.append((edge.id, 1))
        else:
            oriented.append((edge.id, -1))
    edges = {edge.id: edge for edge in beside}
    loops = _loops(face_id, oriented, edges, grid.problems)
    areas = [loop_twice_area(loop, edges, vertices) for loop in loops]
    order = sorted(range(len(loops)), key=lambda k: (areas[k] <= 0, loops[k]))
    return SectionFace(
        face_id,
        region.volume,
        model.topology.volumes[region.volume].domain,
        tuple(loops[k] for k in order),
        tuple(sorted(embedded)),
        sum(areas, Fraction(0)),
    )


def _loops(
    face_id: str,
    oriented: list[tuple[str, int]],
    edges: dict[str, SectionEdge],
    problems: list[str],
) -> list[Loop]:
    """The closed loops of a face's oriented boundary edges; a vertex left or entered twice is a failure."""

    def ends(entry: tuple[str, int]) -> tuple[str, str]:
        edge = edges[entry[0]]
        return (edge.v0, edge.v1) if entry[1] > 0 else (edge.v1, edge.v0)

    leaving: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for entry in oriented:
        leaving[ends(entry)[0]].append(entry)
    if any(len(out) != 1 for out in leaving.values()):
        problems.append(
            f"face {face_id}: a vertex of its boundary is left more than once"
        )
    unused = set(oriented)
    loops: list[Loop] = []
    for entry in sorted(oriented):
        if entry not in unused:
            continue
        loop, start, current = [], ends(entry)[0], entry
        while current in unused:
            unused.discard(current)
            loop.append(current)
            following = leaving.get(ends(current)[1], [])
            if not following:
                break
            current = following[0]
        if ends(loop[-1])[1] != start:
            problems.append(f"face {face_id}: a boundary loop does not close")
        loops.append(tuple(loop))
    return loops


def loop_twice_area(
    loop: Loop, edges: dict[str, SectionEdge], vertices: dict[str, SectionVertex]
) -> Fraction:
    """Twice the signed area a loop encloses in the ``(u, z)`` plane (grid units x um), positive counter-clockwise."""
    total = Fraction(0)
    for edge_id, sign in loop:
        edge = edges[edge_id]
        a, b = (edge.v0, edge.v1) if sign > 0 else (edge.v1, edge.v0)
        pa, pb = vertices[a], vertices[b]
        total += pa.u * pb.z - pb.u * pa.z
    return total
