"""Writing the BREP and the XAO file (design brief decision D8, stage S10).

Owns the file layout and the chunked rendering of a
:class:`~scgsim.geometry.sgb.emission.plan.BrepPlan`:

* **BREP** -- one ``CASCADE Topology V1`` compound of the solids, then the free faces and edges, every
  shared sub-shape written once and referenced by index;
* **XAO 1.0** -- that BREP in a CDATA block, the ``TopExp::MapShapes`` index of every sub-shape and the
  planned primary physical groups.

Each section is cut into index ranges of ``chunk_items`` that depend only on the read-only plan and are
rendered and written in file order, so no section is held in memory whole and the bytes are identical
for every chunk size (tested).

Does not own the plan (``plan``), the records (``records``) or the sidecar (``sidecar``).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO
from xml.sax.saxutils import quoteattr

from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.emission.records import SECTION_TEXT, compound_text
from scgsim.geometry.sgb.stopwatch import elapsed_seconds

CHUNK_ITEMS = 40_000  # items of one section per text-generation task
BREP_HEAD = "CASCADE Topology V1, (c) Matra-Datavision\nLocations 0\nCurve2ds %d\n"
BREP_PREFIX = "DBRep_DrawableShape\n\n"  # first line of a BREP file; the XAO's CDATA block starts after it
Task = tuple[str, int, int]  # (section, first index, one past the last index)
Piece = (
    str | Task | Path
)  # a literal, a chunk task, or a BREP file whose body is copied


def write_brep_file(
    plan: BrepPlan,
    path: Path,
    *,
    chunk_items: int = CHUNK_ITEMS,
) -> float:
    """Write ``path`` as an ASCII BREP and return the seconds the rendering took."""
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _write_pieces(path, plan, list(_brep_pieces(plan, BREP_PREFIX, chunk_items)))
    return elapsed_seconds(started)


def write_xao_file(
    plan: BrepPlan,
    path: Path,
    *,
    name: str | None = None,
    chunk_items: int = CHUNK_ITEMS,
    brep: Path | None = None,
) -> float:
    """Write ``path`` as XAO 1.0 (BREP, sub-shape references, physical groups) and return the seconds it took.

    ``brep`` is a file this plan was already written to by :func:`write_brep_file`: its body is copied
    into the XAO instead of being rendered a second time (the bytes are the same).

    The ``reference`` of each sub-shape is its ``TopExp::MapShapes`` index, through which Gmsh resolves
    the groups.  Groups are the planned primary groups: face or solid elements by 0-based index in
    tag order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _write_pieces(
        path, plan, list(_xao_pieces(plan, name or path.stem, chunk_items, brep))
    )
    return elapsed_seconds(started)


def _xao_pieces(
    plan: BrepPlan, name: str, chunk_items: int, brep: Path | None
) -> Iterator[Piece]:
    c = plan.counts
    yield (
        '<?xml version="1.0" encoding="UTF-8"?>\n<XAO version="1.0" author="scgsim.sgb">\n'
        f'  <geometry name={quoteattr(name)}>\n    <shape format="BREP"><![CDATA[\n'
    )
    if brep is None:
        yield from _brep_pieces(plan, "", chunk_items)
    else:
        yield brep
    yield "]]></shape>\n    <topology>\n"
    for kind, section, count in (
        ("vertices", "xao_vertices", c["vertices"]),
        ("edges", "xao_edges", c["edges"]),
        ("faces", "xao_faces", c["faces"]),
    ):
        yield f'      <{kind} count="{count}">\n'
        yield from _tasks(section, count, chunk_items)
        yield f"      </{kind}>\n"
    solids = "".join(
        f'        <solid index="{i}" name="" reference="{ref}"/>\n'
        for i, ref in enumerate(plan.found.solid_ref)
    )
    yield f'      <solids count="{c["solids"]}">\n{solids}      </solids>\n    </topology>\n  </geometry>\n'
    yield _groups_xml(plan)
    yield "</XAO>\n"


def _brep_pieces(plan: BrepPlan, prefix: str, chunk_items: int) -> Iterator[Piece]:
    """The BREP as an ordered stream of literal strings and (section, lo, hi) chunk tasks."""
    c = plan.counts
    yield prefix + BREP_HEAD % c["pcurves"]
    yield from _tasks("curve2ds", c["pcurves"], chunk_items)
    yield "Curves %d\n" % c["edges"]
    yield from _tasks("curves", c["edges"], chunk_items)
    yield "Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces %d\n" % c["surfaces"]
    yield from _tasks("surfaces", c["surfaces"], chunk_items)
    yield "Triangulations 0\n\nTShapes %d\n" % plan.section_bases()["total"]
    for section in ("vertices", "edges", "wires", "faces", "shells", "solids"):
        yield from _tasks(section, c[section], chunk_items)
    yield compound_text(plan)


def _tasks(section: str, count: int, chunk_items: int) -> Iterator[Task]:
    for lo in range(0, count, chunk_items):
        yield (section, lo, min(count, lo + chunk_items))


def _groups_xml(plan: BrepPlan) -> str:
    lines = [f'  <groups count="{len(plan.groups)}">\n']
    for group in plan.groups:
        dimension = "face" if group.dim == 2 else "solid"
        lines.append(
            f'    <group name={quoteattr(group.name)} dimension="{dimension}" tag="{group.tag}" '
            f'count="{len(group.elements)}">\n'
        )
        lines.append("".join(f'      <element index="{i}"/>\n' for i in group.elements))
        lines.append("    </group>\n")
    lines.append('  </groups>\n  <fields count="0"/>\n')
    return "".join(lines)


def _write_pieces(path: Path, plan: BrepPlan, pieces: list[Piece]) -> None:
    """Write literals, rendered chunk tasks and copied BREP bodies in order."""
    with path.open("wb") as out:
        for piece in pieces:
            if isinstance(piece, tuple):
                section, lo, hi = piece
                out.write(SECTION_TEXT[section](plan, lo, hi).encode("ascii"))
            elif isinstance(piece, Path):
                _copy_brep_body(out, piece)
            else:
                out.write(piece.encode("ascii"))


def _copy_brep_body(out: BinaryIO, source: Path) -> None:
    with source.open("rb") as handle:
        handle.seek(len(BREP_PREFIX))
        shutil.copyfileobj(handle, out)
