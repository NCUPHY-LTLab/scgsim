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
  four cases, not documented).  :class:`StepExport` therefore carries, per solid, its faces in that order
  with area and centre, as the guard the importer checks (counts plus area and centre, fail closed).
* **Free faces** (lumped-port sheets) cannot travel in STEP at micrometre scale; the adapter recreates
  them from the sidecar (``ports``, ``embedded_faces``).
"""

# pyright: reportAttributeAccessIssue=false
# OCP is an untyped native binding (its modules expose no stubs), so pyright cannot see its symbols.
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from OCP.BRep import BRep_Builder
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Interface import Interface_Static
from OCP.Message import Message, Message_Gravity
from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Controller
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS_Compound
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool

from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.verification.occt import read_shape
from scgsim.geometry.sgb.verification.ocp import require_ocp

LENGTH_UNIT = (
    "UM"  # the label of the written numbers; the numbers themselves are micrometres
)
ASSEMBLY_NAME = "SGB_MODEL"


class StepExportError(SgbError):
    """OCCT refused to write the STEP file."""


@dataclass
class SolidGuard:
    """One exported solid: its AEDT name and its faces in OCC solid face order (the AEDT face-id order)."""

    solid_id: str
    aedt_name: str
    faces: list[dict[str, Any]] = field(
        default_factory=list
    )  # {"face", "area_um2", "centre_um"}


@dataclass
class StepExport:
    """The written STEP file and the guard an importer checks object by object."""

    path: Path
    unit: str
    solids: list[SolidGuard]

    @property
    def object_names(self) -> set[str]:
        """The AEDT object names the import must create, exactly."""
        return {s.aedt_name for s in self.solids}


def export_aedt_step(
    geometry_file: Path, sidecar: Mapping[str, Any], step_path: Path
) -> StepExport:
    """Write the solids of ``geometry_file`` (BREP or XAO) as a named STEP assembly in micrometres.

    Solids are matched to the sidecar by tag order (the ``TopExp::MapShapes`` order of the file), so each
    gets the ``aedt_name`` of its planned volume.
    """
    require_ocp()
    shape = read_shape(geometry_file)
    solids = _indexed(shape, TopAbs_SOLID)
    solid_ids = sidecar["binding"]["ids_by_tag"]["3"]
    if solids.Extent() != len(solid_ids):
        raise StepExportError(
            f"{geometry_file}: {solids.Extent()} solids, the sidecar lists {len(solid_ids)}"
        )
    names = [sidecar["volumes"][vid]["aedt_name"] for vid in solid_ids]
    all_faces = _indexed(shape, TopAbs_FACE)
    face_ids = sidecar["binding"]["ids_by_tag"]["2"]
    guards = [
        SolidGuard(vid, name, _face_guard(solids.FindKey(i + 1), all_faces, face_ids))
        for i, (vid, name) in enumerate(zip(solid_ids, names, strict=True))
    ]
    _write_assembly(
        [solids.FindKey(i + 1) for i in range(solids.Extent())], names, step_path
    )
    return StepExport(step_path, LENGTH_UNIT, guards)


def _indexed(shape: Any, kind: Any) -> Any:
    out = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, out)
    return out


def _face_guard(
    solid: Any, all_faces: Any, face_ids: list[str]
) -> list[dict[str, Any]]:
    """The faces of one solid in OCC solid face order, with planned id, area and centre of mass."""
    props = GProp_GProps()
    out = []
    faces = _indexed(solid, TopAbs_FACE)
    for i in range(1, faces.Extent() + 1):
        face = faces.FindKey(i)
        BRepGProp.SurfaceProperties_s(face, props)
        centre = props.CentreOfMass()
        out.append(
            {
                "face": face_ids[all_faces.FindIndex(face) - 1],
                "area_um2": props.Mass(),
                "centre_um": [centre.X(), centre.Y(), centre.Z()],
            }
        )
    return out


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
