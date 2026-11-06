"""Turn Surface-EPR inset margins into SGB inset requests.

Bands are geometry: SGB cuts them into the model, so the margins of ``set_surface_epr`` must be known when
the geometry is built.  Which faces carry a Surface-EPR row, and on which support, is a fact of the built
model, so the requests are derived from a first, unbanded build of the same input: its sidecar lists
every row, the group that carries it and the face it lies on.

A support is one face of one Entity (``ConductorSurface``: a Route A ``sheet``, or the ``top`` or ``bottom`` of
a Route B finite body) or one solution interface (``SolutionInterface``: the ``SA`` gap).  Sidewalls are not
banded (SGB has no sidewall bands in version 1) and so only the baseline row covers them.  The margins of a
support are the union of the margins of every spec whose interface type has a row on it, sorted; a Route A
sheet carries both an ``MA`` and an ``MS`` row, so one request serves both specs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    SolutionInterface,
)

BANDED_FACE_KINDS = ("sheet", "top", "bottom", "interface")


def requests_wanted(specs: Mapping[str, Mapping[str, Any]] | None) -> bool:
    """Whether any Surface-EPR spec asks for inset margins (otherwise no unbanded build is needed)."""
    return any(spec.get("inset_margins_um") for spec in (specs or {}).values())


def inset_requests(
    specs: Mapping[str, Mapping[str, Any]], unbanded_sidecar: Mapping[str, Any]
) -> tuple[InsetRequest, ...]:
    """The inset requests of ``specs`` on the supports of the rows in ``unbanded_sidecar``."""
    margins_by_support: dict[ConductorSurface | SolutionInterface, set[float]] = (
        defaultdict(set)
    )
    for interface_type, spec in sorted(specs.items()):
        margins = spec.get("inset_margins_um")
        if not margins:
            continue
        for support in _supports(interface_type, spec, unbanded_sidecar):
            margins_by_support[support].update(float(m) for m in margins)
    ordered = sorted(margins_by_support, key=_support_key)
    requests = tuple(
        InsetRequest(
            _request_id(index, support),
            support,
            tuple(sorted(margins_by_support[support])),
        )
        for index, support in enumerate(ordered)
    )
    return requests


def _supports(
    interface_type: str, spec: Mapping[str, Any], sidecar: Mapping[str, Any]
) -> set[ConductorSurface | SolutionInterface]:
    """The supports of the rows of ``interface_type`` (restricted to the spec's face kinds)."""
    physical = sidecar["physical_groups"]
    face_kinds = spec.get("face_kinds")
    found: set[ConductorSurface | SolutionInterface] = set()
    for row in sidecar["surface_epr_rows"].values():
        if row["interface_type"] != interface_type:
            continue
        group = physical[row["group"]]
        kind = group["face_kind"]
        if kind not in BANDED_FACE_KINDS:
            continue
        if face_kinds is not None and kind not in face_kinds:
            continue
        if interface_type == "SA":
            found.update(_interface_supports(group, sidecar))
        else:
            (entity,) = group["entities"]
            found.add(ConductorSurface(entity, kind))  # type: ignore[arg-type]
    return found


def _interface_supports(
    group: Mapping[str, Any], sidecar: Mapping[str, Any]
) -> set[SolutionInterface]:
    """One support for each level an SA group lies on that SGB can band.

    Bands are measured from gap edges, which exist only on the plane a conductor is attached to; the
    other levels of the same pair (the underside of the substrate) have none, and SGB refuses to band them.
    """
    first, second = group["entities"]
    levels = {sidecar["faces"][face]["z_um"][0] for face in group["ids"]}
    return {
        SolutionInterface(first, second, z_um=z)
        for z in levels & _attachment_levels(sidecar)
    }


def _attachment_levels(sidecar: Mapping[str, Any]) -> set[float]:
    """Heights of the planes conductors are attached to: a sheet's plane, or the bottom of a finite body."""
    levels: set[float] = set()
    for group in sidecar["physical_groups"].values():
        if group["dim"] != 2:
            continue
        attached = group["interface_type"] == "SHEET" or (
            group["interface_type"] in ("MA", "MS") and group["face_kind"] == "bottom"
        )
        if attached:
            levels.update(sidecar["faces"][face]["z_um"][0] for face in group["ids"])
    return levels


def _support_key(
    support: ConductorSurface | SolutionInterface,
) -> tuple[Any, ...]:
    if isinstance(support, ConductorSurface):
        return (0, support.entity_id, support.side, 0.0)
    return (
        1,
        support.first_entity_id,
        support.second_entity_id,
        support.z_um or 0.0,
    )


def _request_id(index: int, support: ConductorSurface | SolutionInterface) -> str:
    """A request id that names the support (it prefixes the lineage of the cut edges)."""
    if isinstance(support, ConductorSurface):
        label = f"{support.entity_id}_{support.side}"
    else:
        label = f"{support.first_entity_id}_{support.second_entity_id}_{support.z_um}"
    return f"epr{index}_" + re.sub(r"[^0-9A-Za-z_.-]+", "_", label)
