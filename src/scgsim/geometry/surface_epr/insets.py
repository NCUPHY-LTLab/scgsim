"""Turn Surface-EPR inset margins into SGB inset requests.

Bands are geometry: SGB cuts them into the model, so the margins of the Surface-EPR specs must be known when
the geometry is built.  SGB owns every rule about which faces can carry bands (a Route A sheet, the top or
bottom of a finite conductor, the solution interface; never a sidewall; only where a gap edge exists), so a
backend states what it wants and lets SGB find the faces.

* **Default path, one build.**  A spec with ``inset_margins_um`` becomes one type-level request,
  ``InterfaceSurfaces(<type>, face_kinds=<the spec's face kinds>)``.
* **Custom path, two builds.**  A spec may customise conductors with ``inset_margins_by_entity_um``.  The
  faces of those conductors are found in a first, unbanded build of the same input (the sidecar lists every
  Surface-EPR row with its group and face kind) and each gets a concrete request with its own margins.  The
  spec's type-level request then leaves those conductors out (``exclude_entity_ids``), so a face never has
  two requests of one spec.

A Route A sheet carries both an ``MA`` and an ``MS`` row, so the requests of both specs can meet on one face.
One face is one band family with one set of margins, so those requests must ask the same margins (compared on
the model grid, so ``0.3`` and ``0.1 + 0.2`` agree); differing ones are an error, found here on the custom
path and by SGB on the default path.  Each spec selects its margins from the family (see
:func:`~scgsim.geometry.surface_epr.specs.margins_of`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from scgsim.geometry.sgb.cutfamilies.requests import (
    FACE_KINDS,
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    Side,
)
from scgsim.geometry.sgb.records.grid import Grid

# The kinds of face a conductor Entity has that SGB can band: a concrete request names one of these.
CONDUCTOR_FACE_KINDS: tuple[Side, ...] = ("sheet", "top", "bottom")


def check_bandable_face_kinds(interface_type: str, face_kinds: list[str]) -> None:
    """Refuse a face kind SGB cannot band.

    A spec names a face by the ``face_kind`` of the sidecar group it lies on, which is SGB's own vocabulary:
    ``sheet``, ``top``, ``bottom`` and ``interface`` map onto themselves.  The remaining kind a surface group
    has, ``sidewall``, has no SGB equivalent because SGB never bands sidewalls.
    """
    unbandable = [kind for kind in face_kinds if kind not in FACE_KINDS]
    if unbandable:
        raise ValueError(
            f"Surface EPR {interface_type} face_kinds {unbandable} cannot carry inset margins; "
            f"SGB bands only {list(FACE_KINDS)} (sidewalls are never banded)."
        )


def requests_wanted(specs: Mapping[str, Mapping[str, Any]] | None) -> bool:
    """Whether any Surface-EPR spec asks for inset margins (otherwise no bands are requested)."""
    return any(
        spec.get("inset_margins_um") or spec.get("inset_margins_by_entity_um")
        for spec in (specs or {}).values()
    )


def needs_unbanded_build(specs: Mapping[str, Mapping[str, Any]] | None) -> bool:
    """Whether the requests need the faces of named conductors, which only a built model lists."""
    return any(
        spec.get("inset_margins_by_entity_um") for spec in (specs or {}).values()
    )


def inset_requests(
    specs: Mapping[str, Mapping[str, Any]],
    unbanded_sidecar: Mapping[str, Any] | None = None,
) -> tuple[InsetRequest, ...]:
    """The inset requests of ``specs``.

    ``unbanded_sidecar`` is the sidecar of a build of the same input without bands; it is needed exactly
    when :func:`needs_unbanded_build` is true.
    """
    named: list[InsetRequest] = []
    if needs_unbanded_build(specs):
        if unbanded_sidecar is None:
            raise ValueError(
                "inset_margins_by_entity_um needs the sidecar of an unbanded build to find the faces."
            )
        named = _named_conductor_requests(specs, unbanded_sidecar)
    by_type = [
        _type_request(interface_type, spec)
        for interface_type, spec in sorted(specs.items())
        if spec.get("inset_margins_um")
    ]
    return (*named, *by_type)


def _type_request(interface_type: str, spec: Mapping[str, Any]) -> InsetRequest:
    """Every face of ``interface_type`` the spec's face kinds allow, except those of customised conductors."""
    face_kinds = spec.get("face_kinds")
    support = InterfaceSurfaces(
        interface_type,
        face_kinds=None if face_kinds is None else tuple(face_kinds),
        exclude_entity_ids=tuple(sorted(spec.get("inset_margins_by_entity_um") or ())),
    )
    return InsetRequest(
        f"epr_{interface_type}",
        support,
        tuple(float(margin) for margin in spec["inset_margins_um"]),
    )


def _named_conductor_requests(
    specs: Mapping[str, Mapping[str, Any]], sidecar: Mapping[str, Any]
) -> list[InsetRequest]:
    """One concrete request per face of a customised conductor, with the one set of margins the face takes.

    A Route A sheet is one face for both ``MA`` and ``MS``, and one face has one band family, so every spec
    that reaches it must give it the same margins: the customised ones by entity, the others by type.  Two
    concrete requests may not name one face, so the specs' margins become one request.
    """
    grid = Grid(sidecar["grid_um"])
    asks_by_face: dict[ConductorSurface, list[_Ask]] = defaultdict(list)
    for interface_type, spec in sorted(specs.items()):
        for entity in sorted(spec.get("inset_margins_by_entity_um") or ()):
            faces = _conductor_faces(interface_type, spec, entity, sidecar)
            if not faces:
                raise ValueError(
                    f"Surface EPR {interface_type} inset_margins_by_entity_um names {entity!r}, which has "
                    f"no {interface_type} face to band" + _face_kinds_note(spec) + "."
                )
            ask = _ask_for(interface_type, spec, entity)
            for face in faces:
                asks_by_face[face].append(ask)
    _add_type_level_asks(asks_by_face, specs, sidecar)
    return [
        InsetRequest(
            f"epr::{face.entity_id}::{face.side}",
            face,
            _shared_margins(face, asks_by_face[face], grid),
        )
        for face in sorted(asks_by_face, key=lambda f: (f.entity_id, f.side))
    ]


@dataclass(frozen=True)
class _Ask:
    """The margins one Surface-EPR spec gives a conductor, and which key of the spec they came from."""

    interface_type: str
    key: str
    margins_um: tuple[float, ...]


def _ask_for(interface_type: str, spec: Mapping[str, Any], entity: str) -> _Ask:
    """The margins ``spec`` gives the customised ``entity``."""
    margins = spec["inset_margins_by_entity_um"][entity]
    return _Ask(
        interface_type, "inset_margins_by_entity_um", tuple(map(float, margins))
    )


def _add_type_level_asks(
    asks_by_face: dict[ConductorSurface, list[_Ask]],
    specs: Mapping[str, Mapping[str, Any]],
    sidecar: Mapping[str, Any],
) -> None:
    """Add the type-level margins of the specs that reach a customised face without customising it.

    They are asked of SGB by type, not by face, but they land on this face too and must agree with it.
    A face only they reach gets no concrete request.
    """
    for interface_type, spec in sorted(specs.items()):
        margins = spec.get("inset_margins_um")
        if not margins:
            continue
        ask = _Ask(interface_type, "inset_margins_um", tuple(map(float, margins)))
        own = spec.get("inset_margins_by_entity_um") or {}
        for entity in sorted({face.entity_id for face in asks_by_face} - set(own)):
            for face in _conductor_faces(interface_type, spec, entity, sidecar):
                if face in asks_by_face:
                    asks_by_face[face].append(ask)


def _shared_margins(
    face: ConductorSurface, asks: list[_Ask], grid: Grid
) -> tuple[float, ...]:
    """The margins every ask on ``face`` gives; asks that differ on the model grid are an error."""
    first = asks[0]
    expected = _on_grid(first, face, grid)
    for other in asks[1:]:
        if _on_grid(other, face, grid) != expected:
            raise ValueError(
                f"Surface EPR {first.interface_type} ({first.key}) asks inset margins "
                f"{list(first.margins_um)} um and {other.interface_type} ({other.key}) asks "
                f"{list(other.margins_um)} um for the {face.side} of {face.entity_id!r}; "
                "one sheet is one band family and takes one set of margins."
            )
    return first.margins_um


def _on_grid(ask: _Ask, face: ConductorSurface, grid: Grid) -> tuple[int, ...]:
    """The margins of ``ask`` as multiples of the model grid, so ``0.3`` and ``0.1 + 0.2`` compare equal.

    A margin off the lattice is refused here: compared as a raw length it could pass for an index.
    """
    units: list[int] = []
    for margin in ask.margins_um:
        unit = grid.lattice_units(margin)
        if unit is None or unit <= 0:
            raise ValueError(
                f"Surface EPR {ask.interface_type} ({ask.key}) inset margin {margin} um for the "
                f"{face.side} of {face.entity_id!r} is not a positive multiple of the "
                f"{grid.size_um} um model grid."
            )
        units.append(unit)
    return tuple(units)


def _conductor_faces(
    interface_type: str,
    spec: Mapping[str, Any],
    entity: str,
    sidecar: Mapping[str, Any],
) -> set[ConductorSurface]:
    """The faces of ``entity`` that carry a Surface-EPR row of ``interface_type``, within the spec's face kinds."""
    face_kinds = spec.get("face_kinds")
    groups = sidecar["physical_groups"]
    faces: set[ConductorSurface] = set()
    for row in sidecar["surface_epr_rows"].values():
        group = groups[row["group"]]
        kind = group["face_kind"]
        if (
            row["interface_type"] == interface_type
            and entity in group["entities"]
            and kind in CONDUCTOR_FACE_KINDS
            and (face_kinds is None or kind in face_kinds)
        ):
            faces.add(ConductorSurface(entity, cast(Side, kind)))
    return faces


def _face_kinds_note(spec: Mapping[str, Any]) -> str:
    face_kinds = spec.get("face_kinds")
    return "" if face_kinds is None else f" among face_kinds {list(face_kinds)}"
