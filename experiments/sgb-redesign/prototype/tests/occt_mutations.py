"""Deliberate defects applied to a loaded BREP shape (negative controls for the official OCCT checks)."""

from __future__ import annotations

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepTools import BRepTools_ReShape
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Iterator


def children(shape) -> list:
    """Direct children with their orientation (as OCCT iterates them)."""
    out, it = [], TopoDS_Iterator(shape)
    while it.More():
        out.append(it.Value())
        it.Next()
    return out


def first(shape, kind, predicate=lambda s: True):
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        if predicate(ex.Current()):
            return ex.Current()
        ex.Next()
    raise LookupError("no such sub-shape")


def reversed_inner_wire(shape):
    """A face with a hole gets its inner wire in the same sense as the outer wire."""
    face = first(shape, TopAbs_FACE, lambda f: len(children(f)) > 1)
    wires = children(face)
    new = TopoDS.Face_s(face.EmptyCopied())
    builder = BRep_Builder()
    builder.Add(new, wires[0])
    for w in wires[1:]:
        builder.Add(new, w.Reversed())
    reshape = BRepTools_ReShape()
    reshape.Replace(face, new)
    return reshape.Apply(shape)


def flipped_face_in_one_shell(shape):
    """One face is referenced with the opposite orientation in a single shell (the neighbour keeps it)."""
    solid = first(shape, TopAbs_SOLID)
    shell = first(solid, TopAbs_SHELL)
    faces = children(shell)
    new = TopoDS.Shell_s(shell.EmptyCopied())
    builder = BRep_Builder()
    for i, f in enumerate(faces):
        builder.Add(new, f.Reversed() if i == 0 else f)
    reshape = BRepTools_ReShape()
    reshape.Replace(shell, new)
    return reshape.Apply(shape)


def vertex_off_curve(shape, distance_um: float = 1e-4):
    """A vertex shared by edges is moved by ``distance_um`` (its edges' curves do not follow)."""
    vertex = TopoDS.Vertex_s(first(shape, TopAbs_VERTEX))
    p = BRep_Tool.Pnt_s(vertex)
    BRep_Builder().UpdateVertex(vertex, gp_Pnt(p.X() + distance_um, p.Y(), p.Z()), BRep_Tool.Tolerance_s(vertex))
    return shape


def missing_face(shape):
    """A face is deleted from the model: its neighbours' shells are open."""
    face = first(shape, TopAbs_FACE)
    reshape = BRepTools_ReShape()
    reshape.Remove(face)
    return reshape.Apply(shape)


def wire_with_gap(shape):
    """A wire loses one of its edges (the loop no longer closes)."""
    wire = first(shape, TopAbs_WIRE, lambda w: len(children(w)) >= 4)
    edges = children(wire)
    new = TopoDS.Wire_s(wire.EmptyCopied())
    builder = BRep_Builder()
    for e in edges[:-1]:
        builder.Add(new, e)
    reshape = BRepTools_ReShape()
    reshape.Replace(wire, new)
    return reshape.Apply(shape)


CONTROLS = {
    "reversed_inner_wire": reversed_inner_wire,
    "flipped_shared_face_in_one_shell": flipped_face_in_one_shell,
    "vertex_moved_1e-4_um_off_its_curve": vertex_off_curve,
    "missing_face_open_shell": missing_face,
    "wire_with_a_gap": wire_with_gap,
}
