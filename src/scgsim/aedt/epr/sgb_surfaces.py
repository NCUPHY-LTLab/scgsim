"""Bind each Surface-EPR row of an imported SGB model to one AEDT surface selection, and choose how it is integrated.

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

The field each row integrates is read on its selection: ``surface`` enters the sheet itself
(``EnterSurface``), ``adjacent`` the surface just off it (``EnterAdjacentSurface``).  The choice is keyed by
sidecar group (``surface_field_selectors``), so a saved solve can be integrated again with another choice
without solving again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scgsim.aedt.runtime.families.sgb_import import ImportedSgbModel
from scgsim.geometry.sgb.sidecar.aedt import aedt_names

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
    app: Any, sidecar: Mapping[str, Any], imported: ImportedSgbModel
) -> dict[str, SurfaceSelection]:
    """Create the non-model sheet of every Surface-EPR row on the imported model; ``{row: binding}``."""
    bindings = surface_face_bindings(sidecar, imported)
    for binding in bindings.values():
        _create_selection_sheet(app, binding)
    return bindings


def _create_selection_sheet(app: Any, binding: SurfaceSelection) -> None:
    """Copy the row's faces into one non-model sheet named ``binding.selection_name``; its area must be theirs."""
    areas = {
        (name, int(face.id)): float(face.area)
        for name in {name for name, _ in binding.faces}
        for face in app.modeler[name].faces
    }
    expected = sum(areas[face] for face in binding.faces)
    created = app.modeler.create_object_from_face(
        [face_id for _, face_id in binding.faces], non_model=True
    )
    sheets = created if isinstance(created, list) else [created]
    names = [sheet.name for sheet in sheets if sheet]
    if len(names) != len(binding.faces):
        raise SurfaceBindingError(
            f"AEDT copied {len(names)} of the {len(binding.faces)} faces of row {binding.row}"
        )
    if len(names) > 1 and not app.modeler.unite(names):
        raise SurfaceBindingError(f"AEDT did not unite the faces of row {binding.row}")
    sheet = app.modeler[names[0]]
    sheet.name = binding.selection_name
    area = sum(float(face.area) for face in sheet.faces)
    # The copies are AEDT's own faces, so their areas agree to floating-point noise.
    if sheet.name != binding.selection_name or abs(area - expected) > 1e-9 * expected:
        raise SurfaceBindingError(
            f"the selection of row {binding.row} is {sheet.name!r} of {area} um2, "
            f"not {binding.selection_name!r} of {expected} um2"
        )


def surface_face_bindings(
    sidecar: Mapping[str, Any], imported: ImportedSgbModel
) -> dict[str, SurfaceSelection]:
    """The face-list binding of every Surface-EPR row, without touching AEDT."""
    rows = sidecar["surface_epr_rows"]
    names = aedt_names([(row, f"epr_{row}") for row in sorted(rows)])
    kinds = {
        name: sidecar["volumes"][solid]["kind"]
        for solid, name in imported.objects.items()
    }
    groups = sidecar["physical_groups"]
    bindings = {}
    for row, record in sorted(rows.items()):
        planned = groups[record["group"]]["ids"]
        # A sheet group's faces are read on their dielectric side whatever the row type (see the module).
        side = (
            "SHEET"
            if groups[record["group"]].get("interface_type") == "SHEET"
            else record["interface_type"]
        )
        faces = tuple(
            selected_copy(row, side, face, imported, kinds) for face in planned
        )
        bindings[row] = SurfaceSelection(
            row=row,
            group=record["group"],
            interface_type=record["interface_type"],
            entities=tuple(record["entities"]),
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
