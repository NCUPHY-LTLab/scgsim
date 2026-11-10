"""STEP export of the emitted solids for AEDT (design brief decision D4, experiment E4).

Owns writing the solids of a written BREP or XAO as an OCCT STEP **XDE assembly**, one named component
per solid, the way AEDT 2024.2 imports it with its object names intact.  Export only: nothing here starts
AEDT.  What E4 measured and this module applies:

* **Names.** Names travel only as the name of an assembly component; a flat list of shapes loses them.
  The names are the sanitised, collision-checked ``aedt_name`` of each volume in the sidecar (AEDT
  keeps only ``[A-Za-z0-9_]``), so the adapter can assert that the objects AEDT created are exactly
  those names.
* **Unit.** OCCT scales the numbers by ``xstep.cascade.unit`` and ``write.step.unit`` only changes the
  *label*: ``cascade=MM`` with ``write=UM`` leaves the numbers unchanged (they are micrometres) and
  labels them micrometre.  The parameters are set after ``STEPControl_Controller.Init_s()`` and
  ``STEPCAFControl_Controller.Init_s()``, otherwise setting them silently fails for the first file.
* **Partitions are pre-split.** Inset bands and every other partition are faces of the solids already,
  so no AEDT Imprint is needed.  Shared faces are written once per solid (AEDT duplicates them).
* **Face identity** is the per-solid OCC face order, which AEDT keeps as its face-id order (observed on
  four cases, not documented).  So the export also writes a durable guard next to the STEP
  (:mod:`~scgsim.geometry.sgb.exchange.guard`): per solid its faces in that order with area, centre of mass,
  vertices and a curved flag, bound to the STEP, BREP/XAO and sidecar by SHA-256.  The importer reads it
  again every time and fails closed.
* **Free faces** (faces no exported solid carries) cannot travel in STEP at micrometre scale; the guard
  records their loops from the written BREP so the importer rebuilds them from SGB output.
"""

# pyright: reportAttributeAccessIssue=false
# OCP is an untyped native binding (its modules expose no stubs), so pyright cannot see its symbols.
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.Interface import Interface_Static
from OCP.Message import Message, Message_Gravity
from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Controller
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import (
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool

from scgsim.geometry.sgb.emission.measures import rounded
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.exchange.guard import (
    GUARD_SCHEMA,
    FaceGuard,
    FreeFace,
    LoopSegment,
    SolidGuard,
    StepGuard,
    free_face_ids,
    guard_mismatches,
    guard_path_for,
    guarded_file,
    write_step_guard,
)
from scgsim.geometry.sgb.sidecar.files import read_sidecar
from scgsim.geometry.sgb.verification.occt import read_shape, sidecar_mismatches
from scgsim.geometry.sgb.verification.ocp import require_ocp

LENGTH_UNIT = (
    "UM"  # the label of the written numbers; the numbers themselves are micrometres
)
ASSEMBLY_NAME = "SGB_MODEL"


class StepExportError(SgbError):
    """OCCT refused to write the STEP file."""


@dataclass(frozen=True)
class StepExport:
    """The written STEP file and its guard (also written, next to it, as :attr:`guard_path`)."""

    path: Path
    guard_path: Path
    guard: StepGuard

    @property
    def unit(self) -> str:
        """The length unit the STEP numbers are labelled with."""
        return self.guard["unit"]

    @property
    def solids(self) -> list[SolidGuard]:
        """The exported solids in tag order, each with its faces in OCC solid face order."""
        return self.guard["solids"]

    @property
    def object_names(self) -> set[str]:
        """The AEDT object names the import must create, exactly."""
        return {s["aedt_name"] for s in self.solids}


def export_aedt_step(
    geometry_file: Path, sidecar_file: Path, step_path: Path
) -> StepExport:
    """Write the solids of ``geometry_file`` (BREP or XAO) as a named STEP assembly in micrometres, and its guard.

    Solids are matched to the sidecar in ``sidecar_file`` by tag order (the ``TopExp::MapShapes`` order of
    the file), so each gets the ``aedt_name`` of its planned volume.  A file that does not belong to the
    sidecar (counts, vertex positions, face areas or solid volumes differ from its ``expected`` block) raises
    :class:`StepExportError` before any name, STEP or guard is written, and so does a file whose solids do
    not carry exactly their planned faces (:func:`~scgsim.geometry.sgb.exchange.guard.guard_mismatches`,
    the check the reader repeats).  The guard ``<STEP file name>.guard.json``
    (:mod:`~scgsim.geometry.sgb.exchange.guard`) is written last, binding the three files by SHA-256.

    One call at a time per output target (``step_path``), which the caller keeps and nothing checks: the
    guard hashes the STEP after writing it, so two calls at once for the same ``step_path`` may leave the
    STEP of one bound by a guard that lists the faces of the other.  Calls with different ``step_path``
    may run at once, also in one folder.
    """
    require_ocp()
    sidecar = read_sidecar(sidecar_file)
    shape = read_shape(geometry_file)
    mismatches = sidecar_mismatches(shape, sidecar)
    if mismatches:
        raise StepExportError(
            f"{geometry_file} does not belong to the sidecar: " + "; ".join(mismatches)
        )
    solids = _indexed(shape, TopAbs_SOLID)
    solid_ids = sidecar["binding"]["ids_by_tag"]["3"]
    names = [sidecar["volumes"][vid]["aedt_name"] for vid in solid_ids]
    faces = _FaceIndex(
        _indexed(shape, TopAbs_FACE), sidecar["binding"]["ids_by_tag"]["2"]
    )
    solid_guards: list[SolidGuard] = [
        {
            "solid": vid,
            "aedt_name": name,
            "faces": [faces.guard(f) for f in _faces_of(solids.FindKey(i + 1))],
        }
        for i, (vid, name) in enumerate(zip(solid_ids, names, strict=True))
    ]
    free_ids = set(free_face_ids(sidecar, solid_ids))
    free_faces = [
        faces.free_face(face, sidecar["faces"][fid]["group"])
        for face, fid in faces.in_tag_order()
        if fid in free_ids
    ]
    mismatches = guard_mismatches(solid_guards, free_faces, sidecar)
    if mismatches:
        raise StepExportError(
            f"{geometry_file} does not carry its planned faces: "
            + "; ".join(mismatches)
        )
    _write_assembly(
        [solids.FindKey(i + 1) for i in range(solids.Extent())], names, step_path
    )
    guard_path = guard_path_for(step_path)
    guard: StepGuard = {
        "schema": GUARD_SCHEMA,
        "unit": LENGTH_UNIT,
        "route": sidecar["route"],
        "model": sidecar["model"],
        "files": {
            "step": guarded_file(step_path, guard_path.parent),
            "geometry": guarded_file(geometry_file, guard_path.parent),
            "sidecar": guarded_file(sidecar_file, guard_path.parent),
        },
        "solids": solid_guards,
        "free_faces": free_faces,
    }
    write_step_guard(guard_path, guard)
    return StepExport(step_path, guard_path, guard)


def _indexed(shape: Any, kind: Any) -> Any:
    out = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, out)
    return out


def _faces_of(solid: Any) -> list[Any]:
    """The faces of one solid in OCC solid face order (the order AEDT keeps as its face ids)."""
    faces = _indexed(solid, TopAbs_FACE)
    return [faces.FindKey(i) for i in range(1, faces.Extent() + 1)]


@dataclass(frozen=True)
class _FaceIndex:
    """The faces of the file in tag order, with their planned ids."""

    faces: Any  # TopTools_IndexedMapOfShape of the whole file
    face_ids: list[str]

    def planned_id(self, face: Any) -> str:
        return self.face_ids[self.faces.FindIndex(face) - 1]

    def in_tag_order(self) -> list[tuple[Any, str]]:
        return [
            (self.faces.FindKey(i), self.face_ids[i - 1])
            for i in range(1, self.faces.Extent() + 1)
        ]

    def guard(self, face: Any) -> FaceGuard:
        """One face's identity record: planned id, area, centre of mass, vertices and curvature."""
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        centre = props.CentreOfMass()
        vertices = _vertices_um(face)
        return {
            "face": self.planned_id(face),
            "area_um2": rounded(props.Mass()),
            "centre_um": _point(centre),
            "vertex_centroid_um": [
                rounded(sum(v[k] for v in vertices) / len(vertices)) for k in range(3)
            ],
            "vertices_um": vertices,
            "curved": BRepAdaptor_Surface(TopoDS.Face_s(face)).GetType()
            != GeomAbs_Plane,
        }

    def free_face(self, face: Any, group: str | None) -> FreeFace:
        """A face no exported solid carries, with its plane normal and its loops for rebuilding it as a sheet."""
        fid = self.planned_id(face)
        topo_face = TopoDS.Face_s(face)
        surface = BRepAdaptor_Surface(topo_face)
        if surface.GetType() != GeomAbs_Plane:
            raise StepExportError(
                f"free face {fid} is curved; only planar sheets can be rebuilt"
            )
        normal = surface.Plane().Axis().Direction()
        sign = -1.0 if topo_face.Orientation() == TopAbs_REVERSED else 1.0
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        outer = BRepTools.OuterWire_s(topo_face)
        holes = [w for w in _wires(topo_face) if not w.IsSame(outer)]
        return {
            "face": fid,
            "group": group,
            "area_um2": rounded(props.Mass()),
            "normal": [rounded(sign * c) for c in (normal.X(), normal.Y(), normal.Z())],
            "loops": [_loop(w, topo_face, fid) for w in (outer, *holes)],
        }


def _vertices_um(face: Any) -> list[list[float]]:
    vertices = _indexed(face, TopAbs_VERTEX)
    return [
        _point(BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertices.FindKey(i))))
        for i in range(1, vertices.Extent() + 1)
    ]


def _wires(face: Any) -> list[Any]:
    wires = _indexed(face, TopAbs_WIRE)
    return [TopoDS.Wire_s(wires.FindKey(i)) for i in range(1, wires.Extent() + 1)]


def _loop(wire: Any, face: Any, fid: str) -> list[LoopSegment]:
    """The edges of ``wire`` in connection order, each as a line or a three-point arc in its walking direction."""
    segments: list[LoopSegment] = []
    explorer = BRepTools_WireExplorer(wire, face)
    while explorer.More():
        edge = explorer.Current()
        curve = BRepAdaptor_Curve(edge)
        first, last = curve.FirstParameter(), curve.LastParameter()
        if curve.GetType() == GeomAbs_Line:
            kind, params = "line", [first, last]
        elif curve.GetType() == GeomAbs_Circle:
            kind, params = "arc", [first, (first + last) / 2.0, last]
        else:
            raise StepExportError(
                f"free face {fid} has an edge that is neither a line nor a circular arc"
            )
        points = [_point(curve.Value(t)) for t in params]
        if edge.Orientation() == TopAbs_REVERSED:
            points.reverse()
        segments.append({"kind": kind, "points_um": points})
        explorer.Next()
    return segments


def _point(point: Any) -> list[float]:
    return [rounded(point.X()), rounded(point.Y()), rounded(point.Z())]


def _write_assembly(solids: list[Any], names: list[str], step_path: Path) -> None:
    STEPControl_Controller.Init_s()
    STEPCAFControl_Controller.Init_s()  # registers the write.step.* parameters (setting them earlier silently fails)
    accepted = [
        Interface_Static.SetCVal_s(
            "xstep.cascade.unit", "MM"
        ),  # numbers stay micrometres ...
        Interface_Static.SetCVal_s(
            "write.step.unit", LENGTH_UNIT
        ),  # ... and are labelled micrometre
        Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 0),
    ]
    if not all(accepted):
        raise StepExportError(f"Interface_Static refused a STEP parameter: {accepted}")
    application = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    application.InitDocument(document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    for solid in solids:
        builder.Add(compound, solid)
    root = shape_tool.AddShape(
        compound, True
    )  # an assembly: one component per solid, in insertion order
    _name(root, ASSEMBLY_NAME)
    components = TDF_LabelSequence()
    shape_tool.GetComponents_s(root, components)
    for i, name in enumerate(names, start=1):
        component = components.Value(i)
        referred = TDF_Label()
        shape_tool.GetReferredShape_s(component, referred)
        _name(component, name)
        _name(referred, name)
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    step_path.parent.mkdir(parents=True, exist_ok=True)
    with _quiet_occt():
        if not writer.Transfer(document, STEPControl_AsIs):
            raise StepExportError("STEP transfer failed")
        status = writer.Write(str(step_path))
    if int(status) != 1:
        raise StepExportError(f"STEP write failed: {status}")


@contextmanager
def _quiet_occt() -> Iterator[None]:
    """OCCT prints transfer statistics at ``Info`` level to stdout; raise the printers' level for the write."""
    printers = list(Message.DefaultMessenger_s().Printers())
    levels = [p.GetTraceLevel() for p in printers]
    for printer in printers:
        printer.SetTraceLevel(Message_Gravity.Message_Fail)
    try:
        yield
    finally:
        for printer, level in zip(printers, levels, strict=True):
            printer.SetTraceLevel(level)


def _name(label: Any, text: str) -> None:
    TDataStd_Name.Set_s(label, TCollection_ExtendedString(text))
