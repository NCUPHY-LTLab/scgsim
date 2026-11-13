"""The entry of SGB's 2D models: the plane section of a validated 3D model along a cut line, in memory.

:func:`build_section_model` takes a :class:`~scgsim.geometry.sgb.model.ConformalModel` that passed SGB's 3D
validation and a :class:`~scgsim.geometry.models.section.CutLine`, and returns the
:class:`~scgsim.geometry.sgb.section.records.SectionModel`: every edge the trace of one 3D face, every face the
trace of one emitted 3D volume, with the 3D groups, Surface-EPR rows and band families restricted to what the
cut crosses.  Priority, dissolution, contacts, route policy, naming and band geometry are therefore the 3D
model's; nothing is re-derived from the sources.

The steps run in order: the cut on the model grid, the cut within the model's extent, the crossings of every
curve of an emitted face (each refused where it breaks a cut rule), the traces of the faces and walls, the
section's cells and regions with the rules read on them, the records, the groups, and the 2D validation.  A
broken cut rule raises :class:`~scgsim.geometry.sgb.errors.InputError` with its place; a failed invariant
raises :class:`~scgsim.geometry.sgb.errors.ValidationError`.  Nothing is written: emitting the 2D model to
files is a later stage.  The section is a function of the 3D model alone.
"""

from __future__ import annotations

from scgsim.geometry.models.section import CutLine
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.section.assembly import (
    AssembledSection,
    assemble_section,
    check_regions,
    section_grid,
)
from scgsim.geometry.sgb.section.frame import cut_frame
from scgsim.geometry.sgb.section.records import SectionModel
from scgsim.geometry.sgb.section.rules import (
    check_within_extent,
    cross_curves,
    curves_of_emitted_faces,
)
from scgsim.geometry.sgb.section.traces import trace_faces
from scgsim.geometry.sgb.section.validation import validate_section
from scgsim.geometry.sgb.topology.routes.groups import PhysicalGroup
from scgsim.geometry.sgb.topology.routes.naming import volume_group


def build_section_model(model: ConformalModel, cut: CutLine) -> SectionModel:
    """The plane section of ``model`` along ``cut``, validated in 2D (see the module docstring)."""
    frame = cut_frame(cut, model.labelled.catalogue.grid)
    curves = curves_of_emitted_faces(model)
    check_within_extent(frame, model.topology, curves)
    crossings = cross_curves(frame, model, curves)
    traces = trace_faces(frame, model, crossings)
    grid = section_grid(frame, model, traces)
    check_regions(frame, model, curves, traces, grid)
    assembled = assemble_section(model, traces, grid)
    groups = _groups(assembled)
    unions = {
        name: kept
        for name, members in model.routed.unions.items()
        if (kept := [member for member in members if member in groups])
    }
    rows = {
        row: group
        for row, group in model.routed.surface_epr_rows.items()
        if group in groups
    }
    report = validate_section(frame, model, traces, assembled, groups, unions, rows)
    report.raise_if_failed()
    return SectionModel(
        model,
        cut,
        frame,
        assembled.vertices,
        assembled.edges,
        assembled.faces,
        groups,
        unions,
        rows,
        _band_families(model, assembled),
        report,
    )


def _groups(assembled: AssembledSection) -> dict[str, PhysicalGroup]:
    """The primary groups of the section: each edge its 3D face's group, each face its volume's."""
    groups: dict[str, PhysicalGroup] = {}
    for edge_id, edge in assembled.edges.items():
        groups.setdefault(edge.group, PhysicalGroup(1)).ids.append(edge_id)
    for face_id, face in assembled.faces.items():
        groups.setdefault(volume_group(face.domain), PhysicalGroup(2)).ids.append(
            face_id
        )
    return dict(sorted(groups.items()))


def _band_families(
    model: ConformalModel, assembled: AssembledSection
) -> tuple[InsetRequest, ...]:
    """The band families the cut crosses, as their concrete requests (with their margins), in family order."""
    crossed = {family for edge in assembled.edges.values() for family, _ in edge.bands}
    return tuple(r for r in model.inset_plan.requests if r.request_id in crossed)
