"""Turn Surface-EPR inset margins into SGB inset requests.

Bands are geometry: SGB cuts them into the model, so the margins of ``set_surface_epr`` must be known when
the geometry is built.  SGB owns every rule about which faces can carry bands (a Route A sheet, the top or
bottom of a finite conductor, the solution interface; never a sidewall; only where a gap edge exists), so
Palace states what it wants and lets SGB find the faces.

* **Default path, one build.**  A spec with ``inset_margins_um`` becomes one type-level request,
  ``InterfaceSurfaces(<type>, face_kinds=<the spec's face kinds>)``.
* **Custom path, two builds.**  A spec may customise conductors with ``inset_margins_by_entity_um``.  The
  faces of those conductors are found in a first, unbanded build of the same input (the sidecar lists every
  Surface-EPR row with its group and face kind) and each gets a concrete request with its own margins.  The
  spec's type-level request then leaves those conductors out (``exclude_entity_ids``), so a face never has
  two requests of one spec.

A Route A sheet carries both an ``MA`` and an ``MS`` row, so the requests of both specs can meet on one face;
SGB then keeps one band family with the union of their margins, and each spec selects its own margins from
it (see :mod:`.physical_groups` and ``preparation.surface_epr``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

from scgsim.geometry.sgb.cutfamilies.requests import (
    FACE_KINDS,
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    Side,
)

# The kinds of face a conductor Entity has that SGB can band: a concrete request names one of these.
CONDUCTOR_FACE_KINDS: tuple[Side, ...] = ("sheet", "top", "bottom")


def check_bandable_face_kinds(interface_type: str, face_kinds: list[str]) -> None:
    """Refuse a face kind SGB cannot band.

    Palace names a face by the ``face_kind`` of the sidecar group it lies on, which is SGB's own vocabulary:
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
    """One concrete request per face of a customised conductor, with the margins every spec gives that face.

    A Route A sheet is one face for both ``MA`` and ``MS``; two concrete requests may not name one face, so
    the margins of the specs that customise it are merged into one request.
    """
    margins_by_face: dict[ConductorSurface, set[float]] = defaultdict(set)
    for interface_type, spec in sorted(specs.items()):
        for entity, margins in sorted(
            (spec.get("inset_margins_by_entity_um") or {}).items()
        ):
            faces = _conductor_faces(interface_type, spec, entity, sidecar)
            if not faces:
                raise ValueError(
                    f"Surface EPR {interface_type} inset_margins_by_entity_um names {entity!r}, which has "
                    f"no {interface_type} face to band" + _face_kinds_note(spec) + "."
                )
            for face in faces:
                margins_by_face[face].update(float(margin) for margin in margins)
    return [
        InsetRequest(
            f"epr::{face.entity_id}::{face.side}",
            face,
            tuple(sorted(margins_by_face[face])),
        )
        for face in sorted(margins_by_face, key=lambda f: (f.entity_id, f.side))
    ]


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
