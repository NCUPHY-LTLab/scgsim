"""Invariants of a section, checked in 2D before it is returned (the counterpart of ``validation.validate``).

* the traces pair up: every stretch of a horizontal face starts and ends at a crossing, and a wall's top and
  bottom chains are crossed at the same places;
* the regions are consistent: the traces around each region see one volume in it, and each side of an edge is
  one region;
* the section matches the 3D model's sources: in every cell, the domain the slab's prisms give at the middle of
  the cell's stretch of the cut is the domain of the face there (or no face, where it is not emitted);
* every edge bounds a face, with the domain it says on each side; no vertex lies inside an edge;
* every face's loops close, each boundary edge is used once, the outer loop comes first and is the only
  counter-clockwise one, and the loops enclose exactly the area of the face's cells;
* every edge and face is in exactly one primary group; unions and Surface-EPR rows name existing groups.

Everything is recomputed from the records and the 3D model's slabs, not from how the section was assembled.
A failed invariant raises :class:`~scgsim.geometry.sgb.errors.ValidationError`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction

from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.topology.routes.groups import PhysicalGroup
from scgsim.geometry.sgb.section.assembly import (
    AssembledSection,
    SectionGrid,
    loop_twice_area,
)
from scgsim.geometry.sgb.section.frame import CutFrame
from scgsim.geometry.sgb.section.traces import Traces
from scgsim.geometry.sgb.checks.validation.report import ValidationReport

_SHOWN = 5  # problems a failed check lists


def validate_section(  # noqa: PLR0913 - the cut, the 3D model and what the section was built from
    frame: CutFrame,
    model: ConformalModel,
    traces: Traces,
    section: AssembledSection,
    groups: dict[str, PhysicalGroup],
    unions: dict[str, list[str]],
    rows: dict[str, str],
) -> ValidationReport:
    """Every invariant of the module docstring, as a report; the caller raises on a failure."""
    report = ValidationReport()
    _add(report, "section_traces_paired", traces.problems, "every trace pairs up")
    _add(
        report,
        "section_regions_consistent",
        section.grid.problems,
        f"{len(section.grid.regions)} regions, each filled by one volume",
    )
    _add(
        report,
        "section_matches_slabs",
        _slab_mismatches(frame, model, section.grid),
        f"{section.grid.columns * section.grid.slabs} cells match the slabs' prisms",
    )
    _add(
        report,
        "section_edges_bound_faces",
        _edge_problems(section),
        f"{len(section.edges)} edges",
    )
    _add(
        report,
        "section_loops_closed",
        _loop_problems(section),
        f"{len(section.faces)} faces",
    )
    _add(
        report,
        "section_groups_partition",
        _group_problems(section, groups, unions, rows),
        f"{len(groups)} groups",
    )
    return report


def _add(report: ValidationReport, name: str, problems: list[str], passed: str) -> None:
    report.add(name, not problems, "; ".join(problems[:_SHOWN]) or passed)


def _slab_mismatches(
    frame: CutFrame, model: ConformalModel, grid: SectionGrid
) -> list[str]:
    """Cells whose face (or absence of one) differs from the domain the slab's own prisms give there."""
    slabs = model.labelled.slabs
    if len(slabs) != grid.slabs:
        return [f"{len(slabs)} slabs for {grid.slabs} rows of cells"]
    policy, volumes = model.policy, model.topology.volumes
    problems = []
    for i in range(grid.columns):
        point = frame.midpoint(grid.us[i], grid.us[i + 1])
        for j, slab in enumerate(slabs):
            prisms = slab.cells
            if prisms is None:
                return [f"slab {slab.plane} keeps no prisms"]
            expected = prisms.domain_at(point)
            region = grid.region((i, j))
            assert region is not None
            found = None if region.volume is None else volumes[region.volume].domain
            if (expected if policy.is_emitted(expected) else None) != found:
                problems.append(
                    f"cell ({grid.us[i]}..{grid.us[i + 1]}, {slab.plane}): the prisms give "
                    f"{expected.name if expected else None}, the section {found.name if found else None}"
                )
    return problems


def _edge_problems(section: AssembledSection) -> list[str]:
    """Edges that bound no face, disagree with the face beside them, or have a vertex inside them."""
    problems = []
    faces, vertices = section.faces, section.vertices
    at_level: dict[Fraction, list[int]] = defaultdict(list)
    at_u: dict[int, list[Fraction]] = defaultdict(list)
    for vertex in vertices.values():
        at_level[vertex.z].append(vertex.u)
        at_u[vertex.u].append(vertex.z)
    for edge in section.edges.values():
        if edge.left is None and edge.right is None:
            problems.append(f"edge {edge.id} bounds no face")
        for face_id, domain in (
            (edge.left, edge.left_domain),
            (edge.right, edge.right_domain),
        ):
            if face_id is not None and faces[face_id].domain != domain:
                problems.append(f"edge {edge.id}: face {face_id} is not {domain}")
        a, b = vertices[edge.v0], vertices[edge.v1]
        if edge.kind == "horizontal":
            inside = [u for u in at_level[a.z] if a.u < u < b.u]
        else:
            inside = [z for z in at_u[a.u] if a.z < z < b.z]
        if (edge.kind == "horizontal") != (a.z == b.z) or not (a.u, a.z) < (b.u, b.z):
            problems.append(f"edge {edge.id} does not run along its kind's direction")
        if inside:
            problems.append(f"edge {edge.id} has a vertex inside it")
    return problems


def _loop_problems(section: AssembledSection) -> list[str]:
    """Faces whose loops do not close, reuse or miss a boundary edge, or enclose the wrong area."""
    problems = []
    edges, vertices, grid = section.edges, section.vertices, section.grid
    for face_id, face in section.faces.items():
        uses = Counter(entry[0] for loop in face.loops for entry in loop)
        boundary = {
            e.id for e in edges.values() if (e.left == face_id) != (e.right == face_id)
        }
        if set(uses) != boundary or any(count != 1 for count in uses.values()):
            problems.append(
                f"face {face_id}: its loops do not use each boundary edge once"
            )
        for loop in face.loops:
            ends = [
                (edges[e].v0, edges[e].v1) if sign > 0 else (edges[e].v1, edges[e].v0)
                for e, sign in loop
            ]
            if any(
                ends[k][1] != ends[(k + 1) % len(ends)][0] for k in range(len(ends))
            ):
                problems.append(f"face {face_id}: a loop does not close")
        areas = [loop_twice_area(loop, edges, vertices) for loop in face.loops]
        if not areas or areas[0] <= 0 or any(area >= 0 for area in areas[1:]):
            problems.append(f"face {face_id}: its first loop is not its one outer loop")
        region = grid.regions[section.face_regions[face_id]]
        cells = sum(
            (
                2 * (grid.us[i + 1] - grid.us[i]) * (grid.zs[j + 1] - grid.zs[j])
                for i, j in region.cells
            ),
            Fraction(0),
        )
        if face.twice_area != sum(areas, Fraction(0)) or face.twice_area != cells:
            problems.append(
                f"face {face_id}: its loops enclose {face.twice_area / 2}, its cells {cells / 2}"
            )
    return problems


def _group_problems(
    section: AssembledSection,
    groups: dict[str, PhysicalGroup],
    unions: dict[str, list[str]],
    rows: dict[str, str],
) -> list[str]:
    """Edges and faces outside exactly one group of their dimension, and names that point at no group."""
    problems = []
    members: Counter[tuple[int, str]] = Counter(
        (group.dim, member) for group in groups.values() for member in group.ids
    )
    expected = [(1, e) for e in section.edges] + [(2, f) for f in section.faces]
    if sorted(members) != sorted(expected) or any(c != 1 for c in members.values()):
        problems.append("the groups do not partition the edges and the faces")
    problems += [f"group {name} is empty" for name, g in groups.items() if not g.ids]
    problems += [
        f"union {name} names {member}, no surface group"
        for name, names in unions.items()
        for member in names
        if member not in groups or groups[member].dim != 1
    ]
    problems += [
        f"row {row} names {group}, no surface group"
        for row, group in rows.items()
        if group not in groups or groups[group].dim != 1
    ]
    return problems
