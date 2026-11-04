"""Writing the BREP and the XAO file (design brief decision D8, stage S10).

Owns the file layout and the chunked, parallel rendering of a
:class:`~scgsim.geometry.sgb.emission.plan.BrepPlan`:

* **BREP** -- one ``CASCADE Topology V1`` compound of the solids, then the free faces and edges, every
  shared sub-shape written once and referenced by index;
* **XAO 1.0** -- that BREP in a CDATA block, the ``TopExp::MapShapes`` index of every sub-shape and the
  planned primary physical groups.

Each section is cut into index ranges of ``chunk_items`` that depend only on the read-only plan.  With
``workers > 1`` a forked pool renders the ranges and returns encoded text in file order, so the bytes
are identical for every worker count and every chunk size (tested); with one worker the same
functions run in-process.  Where the platform has no ``fork`` the writer renders in-process.

Does not own the plan (``plan``), the records (``records``) or the sidecar (``sidecar``).
"""

from __future__ import annotations

import multiprocessing.pool
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO
from xml.sax.saxutils import quoteattr

from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.emission.records import SECTION_TEXT, compound_text
from scgsim.geometry.sgb.forking import fork_context, quiet_fork_warning
from scgsim.geometry.sgb.stopwatch import elapsed_seconds

CHUNK_ITEMS = 40_000  # items of one section per text-generation task
BREP_HEAD = "CASCADE Topology V1, (c) Matra-Datavision\nLocations 0\nCurve2ds %d\n"
BREP_PREFIX = "DBRep_DrawableShape\n\n"  # first line of a BREP file; the XAO's CDATA block starts after it
Task = tuple[str, int, int]  # (section, first index, one past the last index)
Piece = (
    str | Task | Path
)  # a literal, a chunk task, or a BREP file whose body is copied

_PLAN: BrepPlan | None = (
    None  # inherited by forked workers; set before a pool is created
)


def write_brep_file(
    plan: BrepPlan,
    path: Path,
    *,
    workers: int = 1,
    chunk_items: int = CHUNK_ITEMS,
) -> float:
    """Write ``path`` as an ASCII BREP and return the seconds the rendering took."""
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _write_pieces(
        path, plan, list(_brep_pieces(plan, BREP_PREFIX, chunk_items)), workers
    )
    return elapsed_seconds(started)


def write_xao_file(  # noqa: PLR0913 - the plan, the target and the rendering options
    plan: BrepPlan,
    path: Path,
    *,
    name: str | None = None,
    workers: int = 1,
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
        path,
        plan,
        list(_xao_pieces(plan, name or path.stem, chunk_items, brep)),
        workers,
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


def _run_task(task: Task) -> bytes:
    section, lo, hi = task
    assert _PLAN is not None  # set before the pool forks
    return SECTION_TEXT[section](_PLAN, lo, hi).encode("ascii")


def _write_pieces(
    path: Path, plan: BrepPlan, pieces: list[Piece], workers: int
) -> None:
    """Write literals and chunk tasks in order; tasks run in a forked pool when ``workers > 1``."""
    global _PLAN  # noqa: PLW0603 - forked workers inherit the plan; nothing is pickled
    _PLAN = plan
    tasks = [p for p in pieces if isinstance(p, tuple)]
    context = fork_context()
    pool: multiprocessing.pool.Pool | None = None
    try:
        if workers > 1 and len(tasks) > 1 and context is not None:
            with quiet_fork_warning():
                pool = context.Pool(workers)
            results = pool.imap(_run_task, tasks, chunksize=1)
        else:
            results = map(_run_task, tasks)
        with path.open("wb") as out:
            for piece in pieces:
                if isinstance(piece, tuple):
                    out.write(next(results))
                elif isinstance(piece, Path):
                    _copy_brep_body(out, piece)
                else:
                    out.write(piece.encode("ascii"))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _PLAN = None


def _copy_brep_body(out: BinaryIO, source: Path) -> None:
    with source.open("rb") as handle:
        handle.seek(len(BREP_PREFIX))
        shutil.copyfileobj(handle, out)
