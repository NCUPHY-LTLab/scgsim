"""Official OCCT validity checks of a written BREP or XAO (design brief section 14, "Validation layers").

Owns running OpenCASCADE's own checkers, through OCP (the ``occt`` dependency group; end-user installs do
not pull it), over the shape the direct writer produced, and returning a structured report:

1. ``BRepCheck_Analyzer`` with geometry controls on: every unique vertex, edge, wire, face, shell and solid
   is asked ``IsValid``; each faulty one is reported with its own status list and its status in each
   context (an edge in a face, a face in a shell ...), not just one boolean.
2. ``BOPAlgo_ArgumentAnalyzer`` -- the Boolean-operation pre-check -- with these modes on (all are off by
   default): argument type, self-interference, small edges, face rebuild, tangent faces, merge of vertices
   and of edges that coincide but are not shared (a conformity defect), continuity, and curve on surface
   (pcurve against the 3D curve).  Self-interference is the expensive one and can be switched off.
3. Cheap ``ShapeAnalysis`` checks: per shell, free edges and bad (non-manifold or inconsistently oriented)
   edges, and edges shorter than ten tolerances.
4. A cross-check against the plan: the volume of every solid and the area of every face against the planned
   values, matched by the writer's ``TopExp::MapShapes`` numbering.

In strict mode any finding raises :class:`OcctCheckError`.  This module needs OCP at import: call
:func:`~scgsim.geometry.sgb.verification.ocp.require_ocp` first to get the readable error naming the dependency group.

Does not own writing (``emission``) or SGB's own structural invariants (``validation``).
"""

# pyright: reportAttributeAccessIssue=false
# OCP is an untyped native binding (its modules expose no stubs), so pyright cannot see its symbols.
from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from OCP.BOPAlgo import BOPAlgo_ArgumentAnalyzer
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.ShapeAnalysis import ShapeAnalysis_Shell
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape

from scgsim.contracts.sgb_sidecar import Sidecar, views
from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.emission.measures import (
    planned_face_area_um2,
    planned_volume_um3,
)
from scgsim.geometry.sgb.emission.plan import BrepPlan
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.verification.ocp import require_ocp
from scgsim.geometry.sgb.verification.tolerances import (
    RELATIVE_TOLERANCE,
    VERTEX_TOLERANCE_UM,
    relative_error,
)

ARGUMENT_MODES = (
    "ArgumentTypeMode", "SelfInterMode", "SmallEdgeMode", "RebuildFaceMode", "TangentMode",
    "MergeVertexMode", "MergeEdgeMode", "ContinuityMode", "CurveOnSurfaceMode",
)  # fmt: skip
SHORT_EDGE_TOLERANCES = 10


class OcctCheckError(SgbError):
    """The official OCCT checks found a problem (strict mode)."""


@dataclass
class OcctReport:
    """Structured result of :func:`check_shape`; ``problems`` is empty exactly when everything passed."""

    source: str
    counts: dict[str, int] = field(default_factory=dict)
    brepcheck: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    shape_analysis: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether no check found anything."""
        return not self.problems

    def raise_if_failed(self) -> None:
        """Raise :class:`OcctCheckError` listing the first findings."""
        if self.problems:
            raise OcctCheckError(
                f"{self.source}: {len(self.problems)} OCCT finding(s): "
                + "; ".join(self.problems[:6])
            )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready form for a report file."""
        return {
            "source": self.source,
            "ok": self.ok,
            "counts": self.counts,
            "brepcheck": self.brepcheck,
            "bop_argument_analyzer": self.arguments,
            "shape_analysis": self.shape_analysis,
            "properties": self.properties,
            "problems": self.problems,
            "timings_s": {k: round(v, 4) for k, v in self.timings.items()},
        }


# --------------------------------------------------------------------------- reading


BREP_END = b"+1 0 \n"  # the last record of a complete BREP: the root compound


def read_brep(path: str | Path) -> Any:
    """Load an ASCII BREP file into a ``TopoDS_Shape``.

    OCCT's reader does not terminate on every truncated file (measured: it spins on a BREP cut in the middle
    of the shape table), so a file that does not end with the root record is refused before OCCT sees it.
    """
    require_ocp()
    with Path(path).open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - len(BREP_END)))
        if handle.read() != BREP_END:
            raise OcctCheckError(
                f"{path}: the BREP is truncated (it does not end with the root compound record)"
            )
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise OcctCheckError(f"{path}: OCCT could not read the BREP")
    return shape


def read_xao_brep(path: str | Path) -> Any:
    """The BREP inside an XAO file (``geometry/shape`` CDATA, format BREP), loaded the same way."""
    require_ocp()
    root = ET.parse(str(path)).getroot()
    node = root.find("geometry/shape")
    if node is None or node.get("format") != "BREP":
        raise OcctCheckError(f"{path}: no BREP shape in the XAO")
    with tempfile.TemporaryDirectory() as tmp:
        inner = Path(tmp) / "inner.brep"
        inner.write_text(node.text or "")
        return read_brep(inner)


def read_shape(path: str | Path) -> Any:
    """The shape of a ``.brep`` or ``.xao`` file."""
    return read_xao_brep(path) if str(path).endswith(".xao") else read_brep(path)


# --------------------------------------------------------------------------- the checks


def _unique(shape: Any, kind: Any) -> Any:
    out = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, out)
    return out


def _kinds() -> tuple[tuple[str, Any], ...]:
    return (
        ("vertex", TopAbs_VERTEX), ("edge", TopAbs_EDGE), ("wire", TopAbs_WIRE),
        ("face", TopAbs_FACE), ("shell", TopAbs_SHELL), ("solid", TopAbs_SOLID),
    )  # fmt: skip


def _status_names(statuses: Any) -> list[str]:
    return [s.name.replace("BRepCheck_", "") for s in statuses]


def brep_check(shape: Any, *, parallel: bool = True, limit: int = 50) -> dict[str, Any]:
    """``BRepCheck_Analyzer`` per sub-shape: counts and, for every faulty one, its status lists.

    ``histogram`` counts every non-``NoError`` status by sub-shape kind; faulty sub-shapes that carry a
    real status are listed first (a shape can be faulty only because something below it is).
    """
    analyzer = BRepCheck_Analyzer(shape, True, parallel)
    out: dict[str, Any] = {
        "valid": bool(analyzer.IsValid()),
        "checked": {},
        "faulty": [],
        "faulty_count": 0,
        "histogram": {},
    }
    rows = []
    for name, kind in _kinds():
        shapes = _unique(shape, kind)
        out["checked"][name] = shapes.Extent()
        for i in range(1, shapes.Extent() + 1):
            sub = shapes.FindKey(i)
            if analyzer.IsValid(sub):
                continue
            out["faulty_count"] += 1
            row = _faulty_row(analyzer, name, i, sub)
            real = [
                s
                for s in row["statuses"]
                + [t for c in row["contexts"] for t in c["statuses"]]
                if s != "NoError"
            ]
            for status in real:
                key = f"{name}:{status}"
                out["histogram"][key] = out["histogram"].get(key, 0) + 1
            rows.append((0 if real else 1, len(rows), row))
    out["faulty"] = [row for _, _, row in sorted(rows, key=lambda r: r[:2])][:limit]
    return out


def _faulty_row(analyzer: Any, name: str, index: int, sub: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": name, "index": index, "statuses": [], "contexts": []}
    result = analyzer.Result(sub)
    if result is not None:
        row["statuses"] = _status_names(result.Status())
        result.InitContextIterator()
        while result.MoreShapeInContext():
            row["contexts"].append(
                {
                    "in": result.ContextualShape()
                    .ShapeType()
                    .name.replace("TopAbs_", ""),
                    "statuses": _status_names(result.StatusOnShape()),
                }
            )
            result.NextShapeInContext()
    return row


def without_free_shapes(shape: Any) -> tuple[Any, int, int]:
    """The shape minus compound-level free edges and free faces, and how many of each were dropped.

    A ``refinement_only`` curve is written as a free edge of the compound that lies *inside* a face of a
    solid, an embedded sheet (a port or a floating Route A sheet) as a free face inside a solid (the mesher
    embeds both).  To a Boolean pre-check that is a self-interference by design, so they are excluded from
    ``BOPAlgo_ArgumentAnalyzer`` (the other checks still see them; the free faces are analysed on their own
    by :func:`argument_check`).
    """
    kept, edges, faces = [], 0, 0
    for child in _children(shape):
        if child.ShapeType() == TopAbs_EDGE:
            edges += 1
        elif child.ShapeType() == TopAbs_FACE:
            faces += 1
        else:
            kept.append(child)
    if not (edges or faces):
        return shape, 0, 0
    return _compound(kept), edges, faces


def _compound(children: list[Any]) -> Any:
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    for child in children:
        builder.Add(compound, child)
    return compound


def _children(shape: Any) -> list[Any]:
    out, it = [], TopoDS_Iterator(shape)
    while it.More():
        out.append(it.Value())
        it.Next()
    return out


def _analyze(
    shape: Any, *, self_interference: bool, parallel: bool
) -> tuple[Any, dict[str, bool]]:
    analyzer = BOPAlgo_ArgumentAnalyzer()
    analyzer.SetShape1(shape)
    modes = {m: True for m in ARGUMENT_MODES}
    modes["SelfInterMode"] = self_interference
    for mode, value in modes.items():
        setattr(analyzer, mode, value)
    analyzer.SetRunParallel(parallel)
    analyzer.Perform()
    return analyzer, modes


def _result_rows(analyzer: Any, where: str | None = None) -> list[dict[str, Any]]:
    rows = [
        {
            "status": r.GetCheckStatus().name.replace("BOPAlgo_", ""),
            "faulty_shapes_1": r.GetFaultyShapes1().Size(),
        }
        for r in analyzer.GetCheckResult()
    ]
    return [{**row, "in": where} for row in rows] if where else rows


def argument_check(
    shape: Any,
    *,
    self_interference: bool = True,
    parallel: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    """``BOPAlgo_ArgumentAnalyzer`` with the modes listed in the module docstring.

    The solids are analysed without the compound-level free edges and faces (they interfere with their
    host by design); the free faces are analysed as a compound of their own, so a defect among them is
    still found.
    """
    solids, dropped_edges, dropped_faces = without_free_shapes(shape)
    analyzer, modes = _analyze(
        solids, self_interference=self_interference, parallel=parallel
    )
    faulty = bool(analyzer.HasFaulty())
    results = _result_rows(analyzer)
    if dropped_faces:
        free = _compound([c for c in _children(shape) if c.ShapeType() == TopAbs_FACE])
        extra, _ = _analyze(
            free, self_interference=self_interference, parallel=parallel
        )
        faulty = faulty or bool(extra.HasFaulty())
        results += _result_rows(extra, "free faces")
    return {
        "faulty": faulty,
        "flags": modes,
        "results": results[:limit],
        "result_count": len(results),
        "excluded_free_edges": dropped_edges,
        "excluded_free_faces": dropped_faces,
    }


def shape_analysis(shape: Any, tolerance: float) -> dict[str, Any]:
    """Free and bad edges per shell of every solid, and edges shorter than ten tolerances (cheap checks)."""
    shells = []
    solid_index = 0
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid_index += 1
        inner = TopExp_Explorer(explorer.Current(), TopAbs_SHELL)
        shell_index = 0
        while inner.More():
            shell_index += 1
            shell = TopoDS.Shell_s(inner.Current())
            analysis = ShapeAnalysis_Shell()
            analysis.LoadShells(shell)
            bad = analysis.CheckOrientedShells(shell, True, False)
            free = _edge_count(analysis.FreeEdges()) if analysis.HasFreeEdges() else 0
            shells.append(
                {
                    "solid": solid_index,
                    "shell": shell_index,
                    "problem": bool(bad),
                    "free_edges": free,
                    "bad_edges": _edge_count(analysis.BadEdges()),
                }
            )
            inner.Next()
        explorer.Next()
    edges = _unique(shape, TopAbs_EDGE)
    short, shortest = 0, float("inf")
    props = GProp_GProps()
    for i in range(1, edges.Extent() + 1):
        BRepGProp.LinearProperties_s(edges.FindKey(i), props)
        shortest = min(shortest, props.Mass())
        short += props.Mass() < SHORT_EDGE_TOLERANCES * tolerance
    return {
        "shells": shells,
        "shells_with_free_edges": sum(1 for s in shells if s["free_edges"]),
        "shells_with_bad_edges": sum(
            1 for s in shells if s["bad_edges"] or s["problem"]
        ),
        "edges_below_10x_tolerance": short,
        "shortest_edge_um": shortest if edges.Extent() else None,
    }


def _edge_count(shape: Any) -> int:
    if shape is None or shape.IsNull():
        return 0
    seen = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, seen)
    return seen.Extent()


def plan_properties(
    shape: Any,
    topology: Topology,
    plan: BrepPlan,
    *,
    volume_rtol: float = 1e-7,
    area_rtol: float = 1e-7,
) -> dict[str, Any]:
    """OCCT volume of every solid and area of every face against the plan (matched by ``MapShapes`` order)."""
    out: dict[str, Any] = {"solids": [], "faces": {}, "mismatches": []}
    props = GProp_GProps()
    solids = _unique(shape, TopAbs_SOLID)
    if solids.Extent() != len(plan.solid_ids):
        out["mismatches"].append(
            f"{solids.Extent()} solids in the shape, {len(plan.solid_ids)} planned"
        )
    for i, vid in enumerate(plan.solid_ids, start=1):
        if i > solids.Extent():
            break
        BRepGProp.VolumeProperties_s(solids.FindKey(i), props)
        planned = planned_volume_um3(topology, topology.volumes[vid])
        err = abs(props.Mass() - planned) / abs(planned)
        out["solids"].append(
            {
                "id": vid,
                "occ_volume_um3": props.Mass(),
                "planned_um3": planned,
                "relative_error": err,
            }
        )
        if props.Mass() <= 0 or err > volume_rtol:
            out["mismatches"].append(
                f"solid {vid}: OCCT volume {props.Mass():.9g} vs planned {planned:.9g}"
            )
    faces = _unique(shape, TopAbs_FACE)
    if faces.Extent() != len(plan.face_ids):
        out["mismatches"].append(
            f"{faces.Extent()} faces in the shape, {len(plan.face_ids)} planned"
        )
    worst, worst_id = 0.0, None
    for i, fid in enumerate(plan.face_ids, start=1):
        if i > faces.Extent():
            break
        BRepGProp.SurfaceProperties_s(faces.FindKey(i), props)
        planned = planned_face_area_um2(topology, topology.faces[fid])
        err = abs(props.Mass() - planned) / max(planned, 1e-300)
        if err > worst:
            worst, worst_id = err, fid
        if err > area_rtol:
            out["mismatches"].append(
                f"face {fid}: OCCT area {props.Mass():.9g} vs planned {planned:.9g}"
            )
    out["faces"] = {
        "checked": min(faces.Extent(), len(plan.face_ids)),
        "max_relative_error": worst,
        "worst_face": worst_id,
    }
    return out


def sidecar_mismatches(shape: Any, sidecar: Sidecar, *, limit: int = 3) -> list[str]:
    """Where a loaded shape differs from the sidecar's load-side view (``views.expected``), by tag order; empty when it belongs to it.

    The OCP twin of :func:`~scgsim.geometry.sgb.verification.loaded.verify_loaded`'s counts, vertex
    placement, face-area and solid-volume checks, with the same tolerances
    (:mod:`~scgsim.geometry.sgb.verification.tolerances`) and the same tag order (``TopExp::MapShapes``).
    Counts come first: when they differ the tag order means nothing, so only those are reported.  Each
    other quantity lists at most ``limit`` of its mismatches, named by planned id.
    """
    require_ocp()
    ids, expected = sidecar["binding"]["ids_by_tag"], views.expected(sidecar)
    kinds = (
        ("0", TopAbs_VERTEX),
        ("1", TopAbs_EDGE),
        ("2", TopAbs_FACE),
        ("3", TopAbs_SOLID),
    )
    found = {dim: _unique(shape, kind) for dim, kind in kinds}
    wrong_counts = [
        f"{what}: the shape has {found[dim].Extent()}, the sidecar plans {len(ids[dim])}"
        for (dim, _), what in zip(
            kinds, ("vertices", "edges", "faces", "solids"), strict=True
        )
        if found[dim].Extent() != len(ids[dim])
    ]
    if wrong_counts:
        return wrong_counts
    return [
        *_vertex_mismatches(found["0"], ids["0"], expected["vertices_um"], limit),
        *_measure_mismatches(
            found["2"], ids["2"], expected["face_area_um2"], "face", "area", limit
        ),
        *_measure_mismatches(
            found["3"], ids["3"], expected["solid_volume_um3"], "solid", "volume", limit
        ),
    ]


def _vertex_mismatches(
    vertices: Any, planned_ids: list[str], positions: list[list[float]], limit: int
) -> list[str]:
    out: list[str] = []
    for tag, (planned_id, want) in enumerate(
        zip(planned_ids, positions, strict=True), start=1
    ):
        point = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertices.FindKey(tag)))
        got = [point.X(), point.Y(), point.Z()]
        if (
            max(abs(g - w) for g, w in zip(got, want, strict=True))
            > VERTEX_TOLERANCE_UM
        ):
            out.append(f"vertex {planned_id}: at {_um(got)} um, planned {_um(want)} um")
    return _first(out, limit, "vertices not where the plan puts them")


def _measure_mismatches(  # noqa: PLR0913 - a shape map, its planned ids and values, and how to name them
    shapes: Any,
    planned_ids: list[str],
    planned: list[float],
    what: str,
    measure: str,
    limit: int,
) -> list[str]:
    props, out = GProp_GProps(), []
    for tag, (planned_id, want) in enumerate(
        zip(planned_ids, planned, strict=True), start=1
    ):
        shape = shapes.FindKey(tag)
        if measure == "volume":
            BRepGProp.VolumeProperties_s(shape, props)
        else:
            BRepGProp.SurfaceProperties_s(shape, props)
        if props.Mass() <= 0 or relative_error(props.Mass(), want) > RELATIVE_TOLERANCE:
            out.append(
                f"{what} {planned_id}: {measure} {props.Mass():.9g}, planned {want:.9g}"
            )
    return _first(out, limit, f"{what} {measure}s differing from the plan")


def _first(rows: list[str], limit: int, what: str) -> list[str]:
    """The first ``limit`` rows of one kind of mismatch, with how many there are."""
    if not rows:
        return []
    return [f"{len(rows)} {what}: " + "; ".join(rows[:limit])]


def _um(values: list[float]) -> str:
    return "(" + ", ".join(f"{v:.9g}" for v in values) + ")"


# --------------------------------------------------------------------------- entry points


def check_shape(  # noqa: PLR0913 - a check configuration
    shape: Any,
    topology: Topology | None = None,
    plan: BrepPlan | None = None,
    *,
    source: str = "<shape>",
    tolerance: float = 1e-7,
    self_interference: bool = True,
    parallel: bool = True,
    strict: bool = False,
) -> OcctReport:
    """Run every check on a loaded shape; with ``topology`` and ``plan`` also compare volumes and areas with the plan."""
    require_ocp()
    report = OcctReport(source)
    report.counts = {name: _unique(shape, kind).Extent() for name, kind in _kinds()}
    report.brepcheck = _timed(
        report, "brep_check", lambda: brep_check(shape, parallel=parallel)
    )
    report.arguments = _timed(
        report,
        "bop_argument_analyzer",
        lambda: argument_check(
            shape, self_interference=self_interference, parallel=parallel
        ),
    )
    report.shape_analysis = _timed(
        report, "shape_analysis", lambda: shape_analysis(shape, tolerance)
    )
    if topology is not None and plan is not None:
        report.properties = _timed(
            report, "plan_properties", lambda: plan_properties(shape, topology, plan)
        )
    _collect_problems(report)
    if strict:
        report.raise_if_failed()
    return report


def _timed(report: OcctReport, name: str, run: Any) -> Any:
    started = time.perf_counter()
    result = run()
    report.timings[name] = time.perf_counter() - started
    return result


def _collect_problems(report: OcctReport) -> None:
    problems = report.problems
    if not report.brepcheck["valid"]:
        rows = report.brepcheck["faulty"][:3]
        problems.append(
            f"BRepCheck_Analyzer: {report.brepcheck['faulty_count']} faulty sub-shape(s) "
            f"{report.brepcheck['histogram']}, e.g. "
            + "; ".join(
                f"{r['kind']} #{r['index']} {r['statuses'] or r['contexts']}"
                for r in rows
            )
        )
    if report.arguments["faulty"]:
        problems.append(
            "BOPAlgo_ArgumentAnalyzer: "
            + ", ".join(sorted({r["status"] for r in report.arguments["results"]}))
        )
    sa = report.shape_analysis
    if sa["shells_with_free_edges"] or sa["shells_with_bad_edges"]:
        problems.append(
            f"ShapeAnalysis_Shell: {sa['shells_with_free_edges']} shell(s) with free edges, "
            f"{sa['shells_with_bad_edges']} with bad edges"
        )
    if sa["edges_below_10x_tolerance"]:
        problems.append(
            f"{sa['edges_below_10x_tolerance']} edge(s) shorter than 10x the tolerance"
        )
    problems.extend(report.properties.get("mismatches", []))


def check_file(
    path: str | Path,
    topology: Topology | None = None,
    plan: BrepPlan | None = None,
    **options: Any,
) -> OcctReport:
    """Read a ``.brep`` or ``.xao`` file with OCP and check it (see :func:`check_shape`)."""
    started = time.perf_counter()
    shape = read_shape(path)
    report = check_shape(shape, topology, plan, source=str(path), **options)
    report.timings["read"] = (
        time.perf_counter() - started - sum(report.timings.values())
    )
    return report
