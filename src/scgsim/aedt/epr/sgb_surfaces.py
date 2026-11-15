"""Bind each Surface-EPR row and junction of an imported SGB model to one AEDT surface selection, and choose how it is integrated.

A Surface-EPR row of the sidecar (``surface_epr_rows``) is one interface type on one group: ``MA``
(metal-air), ``MS`` (metal-substrate) or ``SA`` (substrate-air).  Its faces are the group's planned faces.
A face two imported solids share imports once on each (issue #67), so each face is counted once, on the copy
owned by the solid on the type's dielectric side, the way Palace sides a shared sheet by its type:

* ``MS`` and ``SA``: the dielectric solid's copy;
* ``MA``: the only copy -- the conductor solid's (Route C, where the vacuum is not imported) or the substrate's
  under a Route A sheet;
* a Route A sheet face that a retained finite body (a bump or post, issue #67 decision 8) stands on imports on
  the body and on the substrate: every row reads the substrate's copy, the copy it has when the body is
  excluded, since a PEC body is an excluded volume with a PEC surface;
* a rebuilt sheet (a free face) has its one copy.

Any other number of copies fails rather than picking one.

The row's selection is a non-model sheet copied from those faces (``CreateObjectFromFaces``, united when the
row has several), named after the row.  A face list is not used: in AEDT 2024.2 the field calculator fails on a
face list of faces inside the model (a substrate face under a conductor or against the Region) and the
session ends with it, while a non-model sheet of the same faces integrates (issue #67 development check).
Being non-model, the sheet does not enter the mesh or the solve.

A junction's voltage is integrated the same way, on a non-model sheet copied from its port faces
(:func:`create_junction_selections`), read on their dielectric side like a sheet row.

The field each row integrates is read on its selection: ``surface`` enters the sheet itself
(``EnterSurface``), ``adjacent`` the surface just off it (``EnterAdjacentSurface``).  The choice is keyed by
sidecar group (``surface_field_selectors``), so a saved solve can be integrated again with another choice
without solving again.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from scgsim.aedt.epr.sgb_source import row_selection_names
from scgsim.aedt.runtime.families.sgb_import import (
    ImportedSgbModel,
    copy_faces_to_sheet,
)
from scgsim.contracts.sgb_sidecar import Sidecar, views
from scgsim.geometry.sgb.output.sidecar.aedt import sanitise_aedt_name

FieldSelector = Literal["surface", "adjacent"]
DIELECTRIC_SIDE_TYPES = ("MS", "SA", "SHEET")
_ENTER = {"surface": "EnterSurface", "adjacent": "EnterAdjacentSurface"}


class SurfaceBindingError(RuntimeError):
    """A Surface-EPR row has no single face copy to integrate, or AEDT did not create its selection."""


@dataclass(frozen=True)
class SurfaceSelection:
    """One sidecar Surface-EPR row and the AEDT non-model sheet that carries its faces."""

    row: str
    group: str
    interface_type: str
    entities: tuple[str, ...]
    band: int | None
    selection_name: str
    faces: tuple[
        tuple[str, int], ...
    ]  # (AEDT object, AEDT face id), one per planned face

    def to_payload(self) -> dict[str, Any]:
        """The canonical record of this binding."""
        return {
            "row": self.row,
            "group": self.group,
            "interface_type": self.interface_type,
            "entities": list(self.entities),
            "band": self.band,
            "selection_name": self.selection_name,
            "faces": [list(face) for face in self.faces],
        }


def create_surface_selections(
    app: Any,
    sidecar: Sidecar,
    imported: ImportedSgbModel,
    rows: Collection[str] | None = None,
) -> dict[str, SurfaceSelection]:
    """Create the non-model sheet of each Surface-EPR row in ``rows`` (``None``: every row); ``{row: binding}``."""
    bindings = surface_face_bindings(sidecar, imported)
    if rows is not None:
        unknown = sorted(set(rows) - set(bindings))
        if unknown:
            raise SurfaceBindingError(f"the model has no Surface-EPR rows {unknown}")
        bindings = {row: bindings[row] for row in sorted(rows)}
    for binding in bindings.values():
        create_selection_sheet(
            app, f"row {binding.row}", binding.faces, binding.selection_name
        )
    return bindings


def create_selection_sheet(
    app: Any, label: str, faces: Sequence[tuple[str, int]], name: str
) -> None:
    """Copy ``faces`` into one non-model sheet named ``name``; its area must be theirs.  ``label`` names the owner."""
    copy_faces_to_sheet(app, label, faces, name, non_model=True)


@dataclass(frozen=True)
class JunctionSelection:
    """One junction and the AEDT non-model sheet of its port faces its voltage is integrated on."""

    junction_id: str
    port_group: str
    selection_name: str
    faces: tuple[tuple[str, int], ...]


def junction_face_bindings(
    sidecar: Sidecar, junction_groups: Mapping[str, str], imported: ImportedSgbModel
) -> dict[str, JunctionSelection]:
    """``{junction id: binding}`` for ``junction_groups`` (junction id -> ``PORT::`` group), without AEDT.

    The voltage sheet copies the port faces, read on their dielectric side like a sheet row; its name is
    ``epr_junction_<id>``, and two junction ids that become one AEDT name are refused.
    """
    kinds = {
        name: sidecar["volume"][solid]["kind"]
        for solid, name in imported.objects.items()
    }
    ports = {port["group"]: port for port in sidecar["port"].values()}
    out: dict[str, JunctionSelection] = {}
    by_name: dict[str, str] = {}
    for junction_id, group in sorted(junction_groups.items()):
        name = junction_selection_name(junction_id)
        if name in by_name:
            raise SurfaceBindingError(
                f"junctions {by_name[name]!r} and {junction_id!r} both become the AEDT name {name!r}"
            )
        by_name[name] = junction_id
        faces = tuple(
            selected_copy(junction_id, "SHEET", face, imported, kinds)
            for face in ports[group]["faces"]
        )
        out[junction_id] = JunctionSelection(junction_id, group, name, faces)
    return out


def junction_selection_name(junction_id: str) -> str:
    """The AEDT name of the voltage sheet of junction ``junction_id``."""
    return sanitise_aedt_name(f"epr_junction_{junction_id}")


def create_junction_selections(
    app: Any,
    sidecar: Sidecar,
    junction_groups: Mapping[str, str],
    imported: ImportedSgbModel,
) -> dict[str, JunctionSelection]:
    """Create the voltage sheet of every junction on the imported model; ``{junction id: binding}``."""
    bindings = junction_face_bindings(sidecar, junction_groups, imported)
    for binding in bindings.values():
        create_selection_sheet(
            app,
            f"junction {binding.junction_id}",
            binding.faces,
            binding.selection_name,
        )
    return bindings


def surface_face_bindings(
    sidecar: Sidecar, imported: ImportedSgbModel
) -> dict[str, SurfaceSelection]:
    """The face-list binding of every Surface-EPR row, without touching AEDT."""
    rows = sidecar["surface_epr_row"]
    names = row_selection_names(sidecar)
    kinds = {
        name: sidecar["volume"][solid]["kind"]
        for solid, name in imported.objects.items()
    }
    groups = sidecar["group"]
    members = views.group_members(sidecar)
    bindings = {}
    for row, record in sorted(rows.items()):
        planned = members[record["group"]]
        # A sheet group's faces are read on their dielectric side whatever the row type (see the module).
        side = (
            "SHEET"
            if groups[record["group"]]["interface_type"] == "SHEET"
            else record["interface_type"]
        )
        faces = tuple(
            selected_copy(row, side, face, imported, kinds) for face in planned
        )
        bindings[row] = SurfaceSelection(
            row=row,
            group=record["group"],
            interface_type=record["interface_type"],
            entities=tuple(groups[record["group"]]["entities"]),
            band=record["band"],
            selection_name=names[row],
            faces=faces,
        )
    return bindings


def selected_copy(
    row: str,
    interface_type: str,
    face: str,
    imported: ImportedSgbModel,
    kinds: Mapping[str, str],
) -> tuple[str, int]:
    """The ``(object, face id)`` copy of ``face`` that row ``row`` integrates (see the module).

    ``interface_type`` is the side the row reads: its type, or ``SHEET`` for a row of a sheet group.
    """
    copies = imported.face_copies.get(face, {})
    if len(copies) == 1:
        ((name, face_id),) = copies.items()
        return name, face_id
    dielectric = [
        (name, face_id)
        for name, face_id in copies.items()
        if kinds.get(name) == "dielectric"
    ]
    if (
        len(copies) == 2
        and interface_type in DIELECTRIC_SIDE_TYPES
        and len(dielectric) == 1
    ):
        return dielectric[0]
    raise SurfaceBindingError(
        f"Surface-EPR row {row}: face {face} has imported copies {sorted(copies)}, "
        f"none of them the one {interface_type} reads"
    )


def surface_field_selections(
    bindings: Mapping[str, SurfaceSelection],
    selectors: Mapping[str, FieldSelector] | None,
) -> dict[str, str]:
    """``{row: native token}`` -- ``EnterSurface`` or ``EnterAdjacentSurface`` on the row's selection.

    ``selectors`` are keyed by sidecar group; a group without one reads its faces themselves (``surface``).
    A selector for a group no row carries, or of another value, is refused.
    """
    selectors = dict(selectors or {})
    groups = {binding.group for binding in bindings.values()}
    unknown = sorted(set(selectors) - groups)
    if unknown:
        raise ValueError(
            f"surface_field_selectors name groups no row carries: {unknown}"
        )
    wrong = sorted(g for g, value in selectors.items() if value not in _ENTER)
    if wrong:
        raise ValueError(
            f"surface_field_selectors for {wrong} must be 'surface' or 'adjacent'"
        )
    return {
        row: _ENTER[selectors.get(binding.group, "surface")]
        for row, binding in bindings.items()
    }
