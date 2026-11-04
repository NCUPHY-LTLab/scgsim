"""Official OCCT validity checks of a written BREP / XAO (brief section 14, "Validation layers").

Owns running OpenCASCADE's own checkers (through OCP, the cadquery OCCT binding; a *development-only* dependency)
over the shape the direct writer produced and returning a structured report:

1. ``BRepCheck_Analyzer`` (geometry controls on): every unique vertex, edge, wire, face, shell and solid is
   asked ``IsValid``; each faulty one is reported with its own status list and the status in each context
   (an edge in a face, a face in a shell ...), not just one boolean.
2. ``BOPAlgo_ArgumentAnalyzer`` -- the Boolean-operation pre-check -- with these modes switched on (all are off by
   default): ``ArgumentTypeMode`` (arguments are valid shapes), ``SelfInterMode`` (self-interference),
   ``SmallEdgeMode`` (edges shorter than the fuzzy tolerance), ``RebuildFaceMode`` (faces that cannot be rebuilt
   from their wires), ``TangentMode`` (tangent faces), ``MergeVertexMode`` / ``MergeEdgeMode`` (vertices or edges
   that coincide but are not shared: a conformity defect), ``ContinuityMode`` (C0 continuity of edges and faces) and
   ``CurveOnSurfaceMode`` (pcurve vs 3-D curve). ``OperationType`` stays at its default (no operation is planned),
   so the operation-specific type constraints do not apply. Self-interference is the expensive one and can be
   switched off with ``self_interference=False``.
3. Cheap ``ShapeAnalysis`` checks: per shell ``ShapeAnalysis_Shell`` free edges and bad (non-manifold /
   inconsistently oriented) edges, and edges shorter than ``10 x tolerance`` (``BRepGProp`` length).
4. Cross-check against the plan: ``BRepGProp`` volume of every solid and area of every face against the planned
   signed volume and ``emit.planned_face_area_um2`` (faces and solids are matched by the writer's
   ``TopExp::MapShapes`` numbering, which is also the Gmsh tag order).

In strict mode any finding raises :class:`OcctCheckError`. The module needs OCP; importing it without OCP raises
:class:`OcctUnavailable` with the instruction to install the dev extra.

Does not own writing (``brep_writer``) or SGB's own invariants (``validate``).
"""

from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from sgbproto.errors import SgbError


class OcctUnavailable(SgbError):
    """OCP is not installed."""


class OcctCheckError(SgbError):
    """The official OCCT checks found a problem (strict mode)."""


try:
    from OCP.BOPAlgo import BOPAlgo_ArgumentAnalyzer
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepTools import BRepTools
    from OCP.GProp import GProp_GProps
    from OCP.ShapeAnalysis import ShapeAnalysis_Shell
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
    from OCP.TopTools import TopTools_IndexedMapOfShape

    OCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dev extra
    OCP_AVAILABLE = False

ARGUMENT_MODES = ("ArgumentTypeMode", "SelfInterMode", "SmallEdgeMode", "RebuildFaceMode", "TangentMode",
                  "MergeVertexMode", "MergeEdgeMode", "ContinuityMode", "CurveOnSurfaceMode")


def require_ocp() -> None:
    if not OCP_AVAILABLE:
        raise OcctUnavailable(
            "the official OCCT checks need OCP (cadquery-ocp): install the dev extra (e.g. `pip install "
            "'sgbproto[dev]'`); end-user installs do not pull it"
        )


@dataclass
class OcctReport:
    """Structured result of :func:`check_shape`; ``problems`` is empty exactly when everything passed."""

    source: str
    counts: dict[str, int] = field(default_factory=dict)
    brepcheck: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=dict)
    shape_analysis: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_if_failed(self) -> None:
        if self.problems:
            raise OcctCheckError(f"{self.source}: {len(self.problems)} OCCT finding(s): " + "; ".join(self.problems[:6]))

    def to_dict(self) -> dict:
        return {"source": self.source, "ok": self.ok, "counts": self.counts, "brepcheck": self.brepcheck,
                "bop_argument_analyzer": self.arguments, "shape_analysis": self.shape_analysis,
                "properties": self.properties, "problems": self.problems,
                "timings_s": {k: round(v, 4) for k, v in self.timings.items()}}


# --------------------------------------------------------------------------- reading


def read_brep(path: str | Path):
    """Load an ASCII/binary BREP file into a ``TopoDS_Shape`` (``BRepTools.Read_s`` with a ``BRep_Builder``)."""
    require_ocp()
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise OcctCheckError(f"{path}: OCCT could not read the BREP")
    return shape


def read_xao_brep(path: str | Path):
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


def read_shape(path: str | Path):
    return read_xao_brep(path) if str(path).endswith(".xao") else read_brep(path)


# --------------------------------------------------------------------------- the checks


def _unique(shape, kind) -> "TopTools_IndexedMapOfShape":
    out = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, out)
    return out


def _kinds():
    return (("vertex", TopAbs_VERTEX), ("edge", TopAbs_EDGE), ("wire", TopAbs_WIRE), ("face", TopAbs_FACE),
            ("shell", TopAbs_SHELL), ("solid", TopAbs_SOLID))


def _status_names(statuses) -> list[str]:
    return [s.name.replace("BRepCheck_", "") for s in statuses]


def brep_check(shape, parallel: bool = True, limit: int = 50) -> dict:
    """``BRepCheck_Analyzer`` per sub-shape: counts and, for every faulty one, its status lists (on the shape and
    in each context). ``histogram`` counts every non-``NoError`` status by sub-shape kind; faulty sub-shapes that
    carry a real status are listed first (a shape can be faulty only because something below it is)."""
    analyzer = BRepCheck_Analyzer(shape, True, parallel)
    out: dict = {"valid": bool(analyzer.IsValid()), "checked": {}, "faulty": [], "faulty_count": 0, "histogram": {}}
    rows = []
    for name, kind in _kinds():
        shapes = _unique(shape, kind)
        out["checked"][name] = shapes.Extent()
        for i in range(1, shapes.Extent() + 1):
            sub = shapes.FindKey(i)
            if analyzer.IsValid(sub):
                continue
            out["faulty_count"] += 1
            row = {"kind": name, "index": i, "statuses": [], "contexts": []}
            result = analyzer.Result(sub)
            if result is not None:
                row["statuses"] = _status_names(result.Status())
                result.InitContextIterator()
                while result.MoreShapeInContext():
                    row["contexts"].append({"in": result.ContextualShape().ShapeType().name.replace("TopAbs_", ""),
                                            "statuses": _status_names(result.StatusOnShape())})
                    result.NextShapeInContext()
            real = [s for s in row["statuses"] + [t for c in row["contexts"] for t in c["statuses"]] if s != "NoError"]
            for status in real:
                key = f"{name}:{status}"
                out["histogram"][key] = out["histogram"].get(key, 0) + 1
            rows.append((0 if real else 1, len(rows), row))
    out["faulty"] = [row for _, _, row in sorted(rows, key=lambda r: r[:2])][:limit]
    return out


def without_free_edges(shape):
    """The shape minus compound-level free edges and free faces, and how many of each were dropped.

    A ``refinement_only`` curve is written as a free edge of the compound that lies *inside* a face of a solid, an
    embedded sheet (a port or a floating Route A sheet) as a free face inside a solid (the mesher embeds both).
    To a Boolean pre-check that is a self-interference by design, so they are excluded from
    ``BOPAlgo_ArgumentAnalyzer`` (``BRepCheck`` and the other checks still see them; the free faces are analysed on
    their own by :func:`argument_check`).
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


def _compound(children):
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    for child in children:
        builder.Add(compound, child)
    return compound


def _children(shape) -> list:
    out, it = [], TopoDS_Iterator(shape)
    while it.More():
        out.append(it.Value())
        it.Next()
    return out


def _analyze(shape, self_interference: bool, parallel: bool):
    analyzer = BOPAlgo_ArgumentAnalyzer()
    analyzer.SetShape1(shape)
    modes = {m: True for m in ARGUMENT_MODES}
    modes["SelfInterMode"] = self_interference
    for mode, value in modes.items():
        setattr(analyzer, mode, value)
    analyzer.SetRunParallel(parallel)
    analyzer.Perform()
    return analyzer, modes


def argument_check(shape, self_interference: bool = True, parallel: bool = True, limit: int = 50) -> dict:
    """``BOPAlgo_ArgumentAnalyzer`` with the modes listed in the module docstring.

    The solids are analysed without the compound-level free edges / faces (they interfere with their host by design);
    the free faces are analysed as a compound of their own, so a defect among them is still found.
    """
    solids, dropped_edges, dropped_faces = without_free_edges(shape)
    analyzer, modes = _analyze(solids, self_interference, parallel)
    faulty = bool(analyzer.HasFaulty())
    results = [{"status": r.GetCheckStatus().name.replace("BOPAlgo_", ""), "faulty_shapes_1": r.GetFaultyShapes1().Size()}
               for r in analyzer.GetCheckResult()]
    if dropped_faces:
        free = _compound([c for c in _children(shape) if c.ShapeType() == TopAbs_FACE])
        extra, _ = _analyze(free, self_interference, parallel)
        faulty = faulty or bool(extra.HasFaulty())
        results += [{"status": r.GetCheckStatus().name.replace("BOPAlgo_", ""), "faulty_shapes_1": r.GetFaultyShapes1().Size(),
                     "in": "free faces"} for r in extra.GetCheckResult()]
    return {"faulty": faulty, "flags": modes, "results": results[:limit], "result_count": len(results),
            "excluded_free_edges": dropped_edges, "excluded_free_faces": dropped_faces}


def shape_analysis(shape, tolerance: float) -> dict:
    """Free / bad edges per shell of every solid and edges shorter than ``10 x tolerance`` (cheap checks)."""
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
            free = _count(analysis.FreeEdges()) if analysis.HasFreeEdges() else 0
            non_manifold = _count(analysis.BadEdges())
            shells.append({"solid": solid_index, "shell": shell_index, "problem": bool(bad), "free_edges": free,
                           "bad_edges": non_manifold})
            inner.Next()
        explorer.Next()
    edges = _unique(shape, TopAbs_EDGE)
    short, shortest = 0, float("inf")
    props = GProp_GProps()
    for i in range(1, edges.Extent() + 1):
        BRepGProp.LinearProperties_s(edges.FindKey(i), props)
        length = props.Mass()
        shortest = min(shortest, length)
        short += length < 10 * tolerance
    return {"shells": shells, "shells_with_free_edges": sum(1 for s in shells if s["free_edges"]),
            "shells_with_bad_edges": sum(1 for s in shells if s["bad_edges"] or s["problem"]),
            "edges_below_10x_tolerance": short, "shortest_edge_um": shortest if edges.Extent() else None}


def _count(shape) -> int:
    if shape is None or shape.IsNull():
        return 0
    seen = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, seen)
    return seen.Extent()


def plan_properties(shape, topo, plan, volume_rtol: float = 1e-7, area_rtol: float = 1e-7) -> dict:
    """OCP volume of every solid and area of every face against the plan (matched by ``MapShapes`` order)."""
    from sgbproto.emit import planned_face_area_um2  # lazy: the helper lives with the gmsh-API oracle

    g2 = topo.grid_g**2
    out: dict = {"solids": [], "faces": {}, "mismatches": []}
    props = GProp_GProps()
    solids = _unique(shape, TopAbs_SOLID)
    if solids.Extent() != len(plan.solid_ids):
        out["mismatches"].append(f"{solids.Extent()} solids in the shape, {len(plan.solid_ids)} planned")
    for i, vid in enumerate(plan.solid_ids, start=1):
        if i > solids.Extent():
            break
        BRepGProp.VolumeProperties_s(solids.FindKey(i), props)
        planned = float(sum(s.signed_volume for s in topo.volumes[vid].shells)) * g2
        err = abs(props.Mass() - planned) / abs(planned)
        out["solids"].append({"id": vid, "occ_volume_um3": props.Mass(), "planned_um3": planned, "relative_error": err})
        if props.Mass() <= 0 or err > volume_rtol:
            out["mismatches"].append(f"solid {vid}: OCP volume {props.Mass():.9g} vs planned {planned:.9g}")
    faces = _unique(shape, TopAbs_FACE)
    if faces.Extent() != len(plan.face_ids):
        out["mismatches"].append(f"{faces.Extent()} faces in the shape, {len(plan.face_ids)} planned")
    worst, worst_id = 0.0, None
    for i, fid in enumerate(plan.face_ids, start=1):
        if i > faces.Extent():
            break
        BRepGProp.SurfaceProperties_s(faces.FindKey(i), props)
        planned = planned_face_area_um2(topo, topo.faces[fid])
        err = abs(props.Mass() - planned) / max(planned, 1e-300)
        if err > worst:
            worst, worst_id = err, fid
        if err > area_rtol:
            out["mismatches"].append(f"face {fid}: OCP area {props.Mass():.9g} vs planned {planned:.9g}")
    out["faces"] = {"checked": min(faces.Extent(), len(plan.face_ids)), "max_relative_error": worst, "worst_face": worst_id}
    return out


# --------------------------------------------------------------------------- entry points


def check_shape(
    shape,
    topo=None,
    plan=None,
    *,
    source: str = "<shape>",
    tolerance: float = 1e-7,
    self_interference: bool = True,
    parallel: bool = True,
    strict: bool = False,
) -> OcctReport:
    """Run every check on a loaded shape; with ``topo`` and ``plan`` also compare volumes/areas with the plan."""
    require_ocp()
    report = OcctReport(source)
    report.counts = {name: _unique(shape, kind).Extent() for name, kind in _kinds()}
    t0 = time.perf_counter()
    report.brepcheck = brep_check(shape, parallel)
    report.timings["brep_check"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    report.arguments = argument_check(shape, self_interference, parallel)
    report.timings["bop_argument_analyzer"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    report.shape_analysis = shape_analysis(shape, tolerance)
    report.timings["shape_analysis"] = time.perf_counter() - t0
    if topo is not None and plan is not None:
        t0 = time.perf_counter()
        report.properties = plan_properties(shape, topo, plan)
        report.timings["plan_properties"] = time.perf_counter() - t0

    p = report.problems
    if not report.brepcheck["valid"]:
        rows = report.brepcheck["faulty"][:3]
        p.append(f"BRepCheck_Analyzer: {report.brepcheck['faulty_count']} faulty sub-shape(s) {report.brepcheck['histogram']}, e.g. "
                 + "; ".join(f"{r['kind']} #{r['index']} {r['statuses'] or r['contexts']}" for r in rows))
    if report.arguments["faulty"]:
        p.append("BOPAlgo_ArgumentAnalyzer: " + ", ".join(sorted({r["status"] for r in report.arguments["results"]})))
    sa = report.shape_analysis
    if sa["shells_with_free_edges"] or sa["shells_with_bad_edges"]:
        p.append(f"ShapeAnalysis_Shell: {sa['shells_with_free_edges']} shell(s) with free edges, "
                 f"{sa['shells_with_bad_edges']} with bad edges")
    if sa["edges_below_10x_tolerance"]:
        p.append(f"{sa['edges_below_10x_tolerance']} edge(s) shorter than 10x the tolerance")
    p.extend(report.properties.get("mismatches", []))
    if strict:
        report.raise_if_failed()
    return report


def check_file(path: str | Path, topo=None, plan=None, **options) -> OcctReport:
    """Read a ``.brep`` or ``.xao`` file with OCP and check it (see :func:`check_shape`)."""
    t0 = time.perf_counter()
    shape = read_shape(path)
    report = check_shape(shape, topo, plan, source=str(path), **options)
    report.timings["read"] = time.perf_counter() - t0 - sum(report.timings.values())
    return report
