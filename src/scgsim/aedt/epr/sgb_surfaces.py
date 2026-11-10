"""Bind each Surface-EPR row of an imported SGB model to one AEDT face list, and choose how it is integrated.

A Surface-EPR row of the sidecar (``surface_epr_rows``) is one interface type on one group: ``MA``
(metal-air), ``MS`` (metal-substrate) or ``SA`` (substrate-air).  Its faces are the group's planned faces.
A face two imported solids share imports once on each (issue #67), so each face is counted once, on the copy
owned by the solid on the type's dielectric side, the way Palace sides a shared sheet by its type:

* ``MS`` and ``SA``: the dielectric solid's copy;
* ``MA``: the only copy -- the conductor solid's (Route C, where the vacuum is not imported) or the substrate's
  under a Route A sheet;
* a rebuilt sheet (a free face) has its one copy.

Any other number of copies fails rather than picking one.

The field each row integrates is read on its face list: ``surface`` enters the faces themselves
(``EnterSurface``), ``adjacent`` the surface just off them (``EnterAdjacentSurface``).  The choice is keyed by
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
DIELECTRIC_SIDE_TYPES = ("MS", "SA")
_ENTER = {"surface": "EnterSurface", "adjacent": "EnterAdjacentSurface"}


class SurfaceBindingError(RuntimeError):
    """A Surface-EPR row has no single face copy to integrate, or AEDT refused its face list."""


@dataclass(frozen=True)
class SurfaceFaceList:
    """One sidecar Surface-EPR row and the AEDT face list that carries its faces."""

    row: str
    group: str
    interface_type: str
    entities: tuple[str, ...]
    band: int | None
    face_list: str
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
            "face_list": self.face_list,
            "faces": [list(face) for face in self.faces],
        }


def create_surface_face_lists(
    app: Any, sidecar: Mapping[str, Any], imported: ImportedSgbModel
) -> dict[str, SurfaceFaceList]:
    """Create one face list per Surface-EPR row on the imported model; ``{row: binding}``."""
    bindings = surface_face_bindings(sidecar, imported)
    for binding in bindings.values():
        created = app.modeler.create_face_list(
            [face_id for _, face_id in binding.faces], name=binding.face_list
        )
        if not created or created.name != binding.face_list:
            raise SurfaceBindingError(
                f"AEDT did not create face list {binding.face_list}"
            )
    return bindings


def surface_face_bindings(
    sidecar: Mapping[str, Any], imported: ImportedSgbModel
) -> dict[str, SurfaceFaceList]:
    """The face-list binding of every Surface-EPR row, without touching AEDT."""
    rows = sidecar["surface_epr_rows"]
    names = aedt_names([(row, f"epr_{row}") for row in sorted(rows)])
    kinds = {
        name: sidecar["volumes"][solid]["kind"]
        for solid, name in imported.objects.items()
    }
    bindings = {}
    for row, record in sorted(rows.items()):
        planned = sidecar["physical_groups"][record["group"]]["ids"]
        faces = tuple(
            selected_copy(row, record["interface_type"], face, imported, kinds)
            for face in planned
        )
        bindings[row] = SurfaceFaceList(
            row=row,
            group=record["group"],
            interface_type=record["interface_type"],
            entities=tuple(record["entities"]),
            band=record["band"],
            face_list=names[row],
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
    """The ``(object, face id)`` copy of ``face`` that row ``row`` integrates (see the module)."""
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
    bindings: Mapping[str, SurfaceFaceList],
    selectors: Mapping[str, FieldSelector] | None,
) -> dict[str, str]:
    """``{row: native token}`` -- ``EnterSurface`` or ``EnterAdjacentSurface`` on the row's face list.

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
