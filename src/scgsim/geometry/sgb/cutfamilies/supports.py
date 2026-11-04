"""Support regions of inset requests: the physical face before any partition (design brief section 5).

Owns turning a support selection into a :class:`SupportRegion`.  The face is found in a *physical
pre-pass*: the level arrangement of the catalogue's prisms alone -- no ports, no cut families -- labelled
as usual.  The cells whose label matches the selection are the support face; the edges that separate them
from other cells are its boundary rings.  Because the pre-pass contains only physical prisms, the distance
origin of a band is the physical boundary of its face and nothing else: ports, contact caps, other bands
and tile seams cannot move it, by construction.

Arc edges of the boundary are expressed on the exact circle of their source arc (centre and radius of the
carrier) through the snapped end points, so offsets stay concentric with the source circle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from scgsim.geometry.sgb.cutfamilies.requests import (
    INTERFACE_TYPES,
    ConductorSurface,
    InsetRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.derivation.labelled import (
    LabelledArrangements,
    derive_from_catalogue,
)
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.records.carriers import (
    ArcCarrier,
    Lineage,
    LineEdge,
    RingPrimitive,
)
from scgsim.geometry.sgb.records.grid import Grid, exact_z
from scgsim.geometry.sgb.records.labels import LevelLabel
from scgsim.geometry.sgb.records.prisms import PrismCatalogue
from scgsim.geometry.sgb.records.support import SupportRegion
from scgsim.geometry.sgb.routes.naming import interface_type_of


@dataclass(frozen=True)
class ResolvedSupport:
    """A request with the level its face lies on and the face's boundary."""

    request: InsetRequest
    level: Fraction
    region: SupportRegion


def resolve_supports(
    catalogue: PrismCatalogue, requests: tuple[InsetRequest, ...]
) -> list[ResolvedSupport]:
    """The support face of every inset request, found in the physical pre-pass."""
    if not requests:
        return []
    physical = replace(
        catalogue,
        prisms=tuple(p for p in catalogue.prisms if p.material_kind != "port"),
        families=(),
    )
    labelled = derive_from_catalogue(physical)
    return [_resolve(request, catalogue, labelled) for request in requests]


def _resolve(
    request: InsetRequest, catalogue: PrismCatalogue, labelled: LabelledArrangements
) -> ResolvedSupport:
    margins = _margins(request, catalogue.grid)
    layer, selected = _select_cells(request, catalogue, labelled)
    if request.interface is not None:
        _check_interface(request, layer, selected)
    rings = _boundary_rings(request.request_id, layer, selected, labelled)
    return ResolvedSupport(request, layer.level_z, SupportRegion(rings, margins))


def _margins(request: InsetRequest, grid: Grid) -> tuple[int, ...]:
    if not request.margins_um:
        raise InputError(
            f"inset request {request.request_id!r} needs at least one margin"
        )
    margins: list[int] = []
    for value in request.margins_um:
        units = grid.lattice_units(value)
        if units is None or units <= 0:
            raise InputError(
                f"inset request {request.request_id!r}: margin {value} um is not a positive multiple of "
                f"the {grid.size_um} um model grid"
            )
        margins.append(units)
    if any(b <= a for a, b in zip(margins, margins[1:], strict=False)):
        raise InputError(
            f"inset request {request.request_id!r}: margins must be strictly increasing"
        )
    return tuple(margins)


# --------------------------------------------------------------------------- selecting the cells


def _select_cells(
    request: InsetRequest, catalogue: PrismCatalogue, labelled: LabelledArrangements
) -> tuple[Layer, set[int]]:
    support = request.support
    if isinstance(support, ConductorSurface):
        z = _conductor_level(request, support, catalogue)
        level = labelled.levels.levels.index(z)
        selected = _faces(
            labelled.level_layers[level], lambda lab: _on_conductor(lab, support)
        )
        what = f"{support.side} of {support.entity_id!r}"
    else:
        level = _interface_level(request, support, catalogue, labelled)
        selected = _faces(
            labelled.level_layers[level], lambda lab: _on_interface(lab, support)
        )
        what = (
            f"interface of {support.first_entity_id!r} and {support.second_entity_id!r}"
        )
    if not selected:
        raise InputError(
            f"inset request {request.request_id!r}: the {what} has no face to partition"
        )
    return labelled.level_layers[level], selected


def _faces(layer: Layer, matches) -> set[int]:  # noqa: ANN001 - a predicate over a level label
    return {f for f in range(1, len(layer.arr.faces)) if matches(layer.labels[f])}


def _conductor_level(
    request: InsetRequest, support: ConductorSurface, catalogue: PrismCatalogue
) -> Fraction:
    prisms = [p for p in catalogue.prisms if p.entity_id == support.entity_id]
    if not prisms or prisms[0].material_kind != "conductor":
        raise InputError(
            f"inset request {request.request_id!r}: {support.entity_id!r} is not a conductor entity"
        )
    is_sheet = prisms[0].is_sheet
    if support.side == "sheet" and not is_sheet:
        raise InputError(
            f"inset request {request.request_id!r}: {support.entity_id!r} is finite; ask for its 'top' or 'bottom' face"
        )
    if support.side != "sheet" and is_sheet:
        raise InputError(
            f"inset request {request.request_id!r}: {support.entity_id!r} is a sheet; ask for side='sheet'"
        )
    return prisms[0].z1 if support.side == "top" else prisms[0].z0


def _on_conductor(label: LevelLabel, support: ConductorSurface) -> bool:
    if support.side == "sheet":
        return (
            label.sheet is not None
            and label.sheet.kind == "conductor"
            and label.sheet.name == support.entity_id
        )
    domain = label.below if support.side == "top" else label.above
    return (
        domain is not None
        and domain.kind == "conductor"
        and domain.owner == support.entity_id
    )


def _on_interface(label: LevelLabel, support: SolutionInterface) -> bool:
    if label.sheet is not None or label.below is None or label.above is None:
        return False
    return {label.below.owner, label.above.owner} == {
        support.first_entity_id,
        support.second_entity_id,
    }


def _interface_level(
    request: InsetRequest,
    support: SolutionInterface,
    catalogue: PrismCatalogue,
    labelled: LabelledArrangements,
) -> int:
    solutions = {
        p.entity_id
        for p in catalogue.prisms
        if p.material_kind in ("dielectric", "vacuum")
    }
    for entity in (support.first_entity_id, support.second_entity_id):
        if entity not in solutions:
            raise InputError(
                f"inset request {request.request_id!r}: {entity!r} is not a solution region"
            )
    levels = [
        k
        for k, layer in enumerate(labelled.level_layers)
        if _faces(layer, lambda lab: _on_interface(lab, support))
    ]
    if support.z_um is not None:
        wanted = exact_z(support.z_um)
        levels = [k for k in levels if labelled.levels.levels[k] == wanted]
    if len(levels) != 1:
        found = [str(labelled.levels.levels[k]) for k in levels]
        raise InputError(
            f"inset request {request.request_id!r}: the interface lies on {len(levels)} levels {found}; "
            "pass z_um to choose one"
            if levels
            else f"inset request {request.request_id!r}: the two solution regions share no interface"
        )
    return levels[0]


def _check_interface(request: InsetRequest, layer: Layer, selected: set[int]) -> None:
    if request.interface not in INTERFACE_TYPES:
        raise InputError(
            f"inset request {request.request_id!r}: unknown interface type {request.interface!r}"
        )
    found = {
        kind
        for f in selected
        if (
            kind := interface_type_of(
                layer.labels[f].below, layer.labels[f].above, layer.labels[f].sheet
            )
        )
        is not None
    }
    if request.interface not in found:
        raise InputError(
            f"inset request {request.request_id!r}: expected a {request.interface} face, "
            f"the selected face is {', '.join(sorted(found)) or 'unnamed'}"
        )


# --------------------------------------------------------------------------- boundary rings


def _boundary_rings(
    request_id: str, layer: Layer, selected: set[int], labelled: LabelledArrangements
) -> tuple[tuple[RingPrimitive, ...], ...]:
    arr = layer.arr
    next_half_edge: dict[int, int] = {}
    for cycle in arr.cycles:
        count = len(cycle.half_edges)
        for i, h in enumerate(cycle.half_edges):
            next_half_edge[h] = cycle.half_edges[(i + 1) % count]
    boundary = [
        h
        for h in range(2 * len(arr.edges))
        if arr.he_face[h] in selected and arr.he_face[h ^ 1] not in selected
    ]
    seen: set[int] = set()
    rings: list[list[int]] = []
    for start in boundary:
        if start in seen:
            continue
        ring, h = [], start
        while h not in seen:
            seen.add(h)
            ring.append(h)
            g = next_half_edge[h]
            while (
                arr.he_face[g ^ 1] in selected
            ):  # an interior edge: turn into the neighbouring selected cell
                g = next_half_edge[g ^ 1]
            h = g
        rings.append(_from_smallest_vertex(arr, ring))
    rings.sort(key=lambda ring: (arr.origin(ring[0]), ring[0]))
    return tuple(
        tuple(
            _primitive(f"{request_id}/ring{ri}", k, layer, h, labelled)
            for k, h in enumerate(ring)
        )
        for ri, ring in enumerate(rings)
    )


def _from_smallest_vertex(arr, ring: list[int]) -> list[int]:  # noqa: ANN001 - the arrangement of a layer
    first = min(range(len(ring)), key=lambda i: (arr.origin(ring[i]), ring[i]))
    return ring[first:] + ring[:first]


def _primitive(
    ring_id: str,
    index: int,
    layer: Layer,
    half_edge: int,
    labelled: LabelledArrangements,
) -> RingPrimitive:
    """A boundary half-edge as a straight edge or an arc on the exact circle of its source arc.

    ``ring_id`` is ``<support>/ring<r>``; the primitive's lineage member is ``<ring_id>/edge<index>``.
    """
    arr = layer.arr
    a, b = arr.origin(half_edge), arr.target(half_edge)
    curve = layer.edge_curve[half_edge >> 1]
    if curve is None:
        return LineEdge(a, b, index)
    carrier_id = sorted(arr.edges[half_edge >> 1].carriers)[0]
    arc = labelled.arcs[carrier_id]
    root = labelled.arcs.get(arc.root_id, arc)
    member = f"{ring_id}/edge{index}"
    return ArcCarrier(
        member,
        root.centre,
        root.radius,
        curve.ccw if half_edge % 2 == 0 else not curve.ccw,
        (Fraction(a[0]), Fraction(a[1])),
        (Fraction(b[0]), Fraction(b[1])),
        Lineage(
            "family_member", family_id=ring_id.rsplit("/ring", 1)[0], member=member
        ),
        root.root_id,
    )
