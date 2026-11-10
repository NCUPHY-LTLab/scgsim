"""Support regions of inset requests: the physical face before any partition (design brief section 5).

Owns turning a support selection into a :class:`SupportRegion`.  The face is found in a *physical
pre-pass*: the level arrangement of the catalogue's prisms alone -- no ports, no cut families -- labelled
as usual.  The cells whose label matches the selection are the support face; the edges that separate them
from other cells are its boundary rings.  Because the pre-pass contains only physical prisms, ports, contact caps, other bands and tile seams cannot
move a band, by construction (a port lying on the substrate surface counts as the exposed surface it covers).

Distance origins.  Bands are measured only from *gap edges*: the parts of a conductor footprint's boundary
whose coplanar neighbour, across the edge, is an exposed interface between two non-conductor solution regions
with at least one dielectric (an SA or SS surface).  The Surface-EPR correction factor of a band comes from the
metal-gap-metal cross-section, so an edge with no gap before it -- a conductor edge on the die or substrate
outer edge, an edge shared with another conductor, an edge facing only vacuum -- has no meaningful band and is
not an origin.  A solution interface is measured from the gap edges on its own boundary.  A support with no gap
edge fails closed.

A conductor face is measured from the gap edges of its footprint on the plane the conductor is attached to: its
sheet plane, or the face of a finite body that lies on a dielectric (its metal-substrate face).  That is the
bottom on an upright die and the top on a flipped one (the upper die of a flip chip, a film under a substrate);
a body with a dielectric on neither face, or on both, keeps its bottom.  On that plane the face is the support
itself, and the physical pre-pass has already cut its boundary wherever the cell across it changes, so each
boundary half-edge is an origin or not as a whole.  The other face of a finite conductor lies on another plane:
each of its boundary edges is cut where a gap edge of the attachment plane that lies along the same source edge
begins or ends (``cutfamilies.gap_intervals``), and only the pieces a gap edge covers are origins.  A source edge
that faces a gap along part of its length is thus an origin along that part only.

The die edge.  A boundary piece that is no origin lies on the die edge when, on the attachment plane, the cell
across it lies beyond the die: it holds no conductor and no dielectric (it is outside the model, or only vacuum
lies on either side of the plane there), and such cells connect it to the unbounded face of the level.  That is
the substrate's outer edge, a dicing line, and a gap edge that runs onto it belongs to a conductor the dicing
cut: its distance origin continues straight past the die edge (``SupportRegion.die_cuts``).  The rim of a
vacuum bore, trench or cavity that the die encloses is no dicing line, so a gap edge that ends there keeps the
half circle of its bands.  A solution interface is read the same way on its own level.  The face of a finite
conductor on its other plane takes the die edges of the attachment plane along the same source edges, as it
takes the gap edges.  A support whose continuation the die outline comes back to is refused
(``cutfamilies.die_cuts``).

Arc edges of the boundary are expressed on the exact circle of their source arc (centre and radius of the
carrier) through the snapped end points, so offsets stay concentric with the source circle.

The pre-pass leaves the ports out so they never move a band; the seams of the bands are clipped against the
ports afterwards (``cutfamilies.inset``), so they stop at a port's boundary.

A type-level request (:class:`InterfaceSurfaces`) is turned into concrete faces here too, because the same labels that find
a face find the faces of a type: :func:`faces_of_type` lists, in the pre-pass, every conductor face with a
Surface-EPR row of the type and every solution interface of it, each as a :class:`FaceId`.  Whether such a face
has a gap edge is only known when it is resolved (:class:`NoGapEdgeError`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction

from scgsim.geometry.sgb.cutfamilies.die_cuts import (
    cells_beyond_the_die,
    check_die_cuts,
)
from scgsim.geometry.sgb.cutfamilies.gap_intervals import (
    ArcSpan,
    LineSpan,
    Piece,
    Span,
    split_at_gaps,
)
from scgsim.geometry.sgb.cutfamilies.requests import (
    BAND_TYPES,
    FACE_KINDS,
    INTERFACE_TYPES,
    ConcreteSupport,
    ConductorSurface,
    FaceKind,
    InsetRequest,
    InterfaceSurfaces,
    Side,
    SolutionInterface,
)
from scgsim.geometry.sgb.derivation.labelled import (
    LabelledArrangements,
    derive_from_catalogue,
)
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.errors import InputError, NoGapEdgeError
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
from scgsim.geometry.sgb.routes.naming import (
    SHEET,
    interface_type_of,
    sheet_face_sides,
)


@dataclass(frozen=True)
class ResolvedSupport:
    """A request with the level its face lies on and the face's boundary."""

    request: InsetRequest
    level: Fraction
    region: SupportRegion


@dataclass(frozen=True, order=True)
class FaceId:
    """One bandable face, whoever asked for it: its kind, the entities it belongs to and the level it lies on.

    ``kind`` is one of :data:`~scgsim.geometry.sgb.cutfamilies.requests.FACE_KINDS`.  A conductor face has its one
    entity; an interface has its two solution regions, the dielectric first (the order of its group name).
    Two requests land on the same face exactly when their :class:`FaceId` s are equal.
    """

    kind: FaceKind
    entities: tuple[str, ...]
    level: Fraction

    def support(self) -> ConductorSurface | SolutionInterface:
        """The concrete support that selects this face."""
        if self.kind == "interface":
            first, second = self.entities
            return SolutionInterface(first, second, float(self.level))
        return ConductorSurface(self.entities[0], self.kind)

    def label(self) -> str:
        """The face as a request-id suffix: ``<entity>::<side>`` or ``<first>|<second>::z=<z>`` (entity ids never hold ``::``)."""
        if self.kind == "interface":
            return f"{'|'.join(self.entities)}::{level_label(self.level)}"
        return f"{self.entities[0]}::{self.kind}"


def level_label(level: Fraction) -> str:
    """The level of an interface as it appears in request ids: ``z=<z>`` with ``<z>`` a float in micrometres."""
    return f"z={float(level)}"


def physical_labelling(catalogue: PrismCatalogue) -> LabelledArrangements:
    """The physical pre-pass: the catalogue's prisms without ports or cut families, labelled."""
    physical = replace(
        catalogue,
        prisms=tuple(p for p in catalogue.prisms if p.material_kind != "port"),
        families=(),
    )
    return derive_from_catalogue(physical)


def resolve_support(
    request: InsetRequest, catalogue: PrismCatalogue, labelled: LabelledArrangements
) -> ResolvedSupport:
    """The support face of one concrete inset request, found in the physical pre-pass.

    Raises :class:`NoGapEdgeError` (an :class:`InputError`) when the face has no gap edge to measure from.
    """
    support = concrete_support(request)
    margins = lattice_margins(request, catalogue.grid)
    layer, selected = _select_cells(request, support, catalogue, labelled)
    if request.interface is not None:
        _check_interface(request, layer, selected)
    # The die edge is read on the plane the gap edges are: the attachment plane of a conductor face.
    plane = (
        layer
        if isinstance(support, SolutionInterface)
        else labelled.level_layers[
            _attach_level(support.entity_id, catalogue, labelled)
        ]
    )
    beyond = cells_beyond_the_die(plane)
    pieces_of = _origin_pieces(support, labelled, layer, plane, beyond)
    rings, origin_mask, die_edge_mask = _boundary_rings(
        request.request_id, layer, selected, labelled, pieces_of
    )
    if not any(any(ring) for ring in origin_mask):
        raise NoGapEdgeError(
            f"inset request {request.request_id!r}: {_describe(support)} has no gap edge to "
            "measure the bands from (a conductor edge counts only where exposed dielectric lies next to it)"
        )
    region = SupportRegion(rings, margins, origin_mask, die_edge_mask)
    conductor = support.entity_id if isinstance(support, ConductorSurface) else None
    check_die_cuts(
        request.request_id, region, plane, beyond, conductor, catalogue.grid.size_um
    )
    return ResolvedSupport(request, layer.level_z, region)


def concrete_support(request: InsetRequest) -> ConcreteSupport:
    """The face a request names; a type-level request names none (``inset_plan`` expands it into faces first)."""
    if isinstance(request.support, InterfaceSurfaces):
        raise InputError(
            f"inset request {request.request_id!r}: a type-level support names no single face"
        )
    return request.support


def _describe(support: ConcreteSupport) -> str:
    if isinstance(support, ConductorSurface):
        return f"the {support.side} of {support.entity_id!r}"
    return (
        f"the interface of {support.first_entity_id!r} and {support.second_entity_id!r}"
    )


# --------------------------------------------------------------------------- gap edges


def _origin_pieces(
    support: ConcreteSupport,
    labelled: LabelledArrangements,
    layer: Layer,
    plane: Layer,
    beyond: frozenset[int],
) -> Callable[[int], list[Piece]]:
    """The pieces of a boundary half-edge of the support face: each a distance origin (a gap edge) or not, and on the die edge or not.

    ``plane`` is the level the gap edges and the die edge are read on, and ``beyond`` its cells beyond the die.
    """
    arr = layer.arr

    def whole(h: int, *, origin: bool) -> list[Piece]:
        on_die_edge = not origin and arr.he_face[h ^ 1] in beyond
        return [Piece(arr.origin(h), arr.target(h), origin, on_die_edge)]

    if isinstance(support, SolutionInterface):
        return lambda h: whole(
            h,
            origin=_is_exposed_dielectric(layer.labels[arr.he_face[h]])
            and _is_conductor_cell(layer.labels[arr.he_face[h ^ 1]]),
        )
    if plane is not layer:
        gaps = _conductor_edges_by_source(
            plane, labelled, lambda face: _is_exposed_dielectric(plane.labels[face])
        )
        die_edges = _conductor_edges_by_source(
            plane, labelled, lambda face: face in beyond
        )
        return lambda h: _pieces_along_gaps(layer, h, labelled, gaps, die_edges)
    # The support is on its own plane, whose arrangement is cut wherever the cell across a boundary edge
    # changes: a boundary edge is a gap edge as a whole, or not at all, and the same for the die edge.
    return lambda h: whole(
        h, origin=_is_exposed_dielectric(layer.labels[arr.he_face[h ^ 1]])
    )


def _attach_level(
    entity_id: str, catalogue: PrismCatalogue, labelled: LabelledArrangements
) -> int:
    """Level of the plane a conductor is attached to: its sheet plane, or the face of a finite body on a dielectric.

    The gap edges lie where the conductor meets the substrate, so the attachment plane is the face with a
    metal-substrate (``MS``) cell: the bottom on an upright die, the top on a flipped die whose film hangs under
    its substrate.  A body with a dielectric on neither face, or on both, keeps its bottom.
    """
    prism = next(p for p in catalogue.prisms if p.entity_id == entity_id)
    bottom = labelled.levels.levels.index(prism.z0)
    if prism.is_sheet:
        return bottom
    top = labelled.levels.levels.index(prism.z1)
    if not _lies_on_dielectric(
        labelled.level_layers[bottom], ConductorSurface(entity_id, "bottom")
    ) and _lies_on_dielectric(
        labelled.level_layers[top], ConductorSurface(entity_id, "top")
    ):
        return top
    return bottom


def _lies_on_dielectric(layer: Layer, surface: ConductorSurface) -> bool:
    """Whether some cell of a conductor's top or bottom face is a metal-substrate (``MS``) interface."""
    return any(
        "MS" in _surface_epr_types(layer.labels[f])
        for f in _conductor_cells(layer, surface)
    )


def _conductor_edges_by_source(
    layer: Layer,
    labelled: LabelledArrangements,
    across: Callable[[int], bool],
) -> dict[str, list[Span]]:
    """The edges of a level with a conductor on one side and a cell ``across`` accepts on the other, by source edge.

    ``across`` takes the index of the cell.  Each edge is travelled with its conductor on the left.  With an
    exposed dielectric across they are the gap edges; with a cell beyond the die, the conductor's edges on the
    die edge.
    """
    arr = layer.arr
    by_source: dict[str, list[Span]] = {}
    for edge in range(len(arr.edges)):
        for h in (2 * edge, 2 * edge + 1):
            if _is_conductor_cell(layer.labels[arr.he_face[h]]) and across(
                arr.he_face[h ^ 1]
            ):
                span = _span_of(layer, h, labelled)
                for source in layer.edge_sources[edge]:
                    by_source.setdefault(source, []).append(span)
    return by_source


def _pieces_along_gaps(
    layer: Layer,
    half_edge: int,
    labelled: LabelledArrangements,
    gaps: dict[str, list[Span]],
    die_edges: dict[str, list[Span]],
) -> list[Piece]:
    """A boundary half-edge cut at the gap edges and die edges of another level along the same source edges."""
    sources = layer.edge_sources[half_edge >> 1]
    return split_at_gaps(
        _span_of(layer, half_edge, labelled),
        [gap for source in sources for gap in gaps.get(source, ())],
        [edge for source in sources for edge in die_edges.get(source, ())],
    )


def _span_of(layer: Layer, half_edge: int, labelled: LabelledArrangements) -> Span:
    """A half-edge as a stretch of its carrier: a straight edge, or an arc about the centre of its source circle."""
    arr = layer.arr
    start, end = arr.origin(half_edge), arr.target(half_edge)
    curve = layer.edge_curve[half_edge >> 1]
    if curve is None:
        return LineSpan(start, end)
    root = _root_arc(layer, half_edge, labelled)
    return ArcSpan(root.centre, _is_ccw(curve.ccw, half_edge), start, end)


def _root_arc(
    layer: Layer, half_edge: int, labelled: LabelledArrangements
) -> ArcCarrier:
    """The source arc whose exact circle an arc half-edge lies on."""
    carrier_id = sorted(layer.arr.edges[half_edge >> 1].carriers)[0]
    arc = labelled.arcs[carrier_id]
    return labelled.arcs.get(arc.root_id, arc)


def _is_ccw(edge_ccw: bool, half_edge: int) -> bool:
    """Sense of travel of a half-edge along an arc edge that runs counter-clockwise (or not) from its start to its end."""
    return edge_ccw if half_edge % 2 == 0 else not edge_ccw


def _is_conductor_cell(label: LevelLabel | None) -> bool:
    """Whether a level cell (``None``: outside the model) lies under a conductor footprint.

    A sheet conductor, or a finite conductor above or below the plane.
    """
    if label is None:
        return False
    if label.sheet is not None and label.sheet.kind == "conductor":
        return True
    return any(
        domain is not None and domain.kind == "conductor"
        for domain in (label.below, label.above)
    )


def _is_exposed_dielectric(label: LevelLabel | None) -> bool:
    """Whether a level cell is an interface of two solution regions, at least one a dielectric (SA or SS)."""
    if label is None or label.sheet is not None:
        return False
    if label.below is None or label.above is None:
        return False
    kinds = {label.below.kind, label.above.kind}
    return "conductor" not in kinds and "dielectric" in kinds


def lattice_margins(request: InsetRequest, grid: Grid) -> tuple[int, ...]:
    """The margins of a request as positive, strictly increasing multiples of the model grid (grid units)."""
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
    request: InsetRequest,
    support: ConcreteSupport,
    catalogue: PrismCatalogue,
    labelled: LabelledArrangements,
) -> tuple[Layer, set[int]]:
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
    request_id: str,
    layer: Layer,
    selected: set[int],
    labelled: LabelledArrangements,
    pieces_of: Callable[[int], list[Piece]],
) -> tuple[
    tuple[tuple[RingPrimitive, ...], ...],
    tuple[tuple[bool, ...], ...],
    tuple[tuple[bool, ...], ...],
]:
    """The boundary rings of the selected cells, and per primitive whether it is a distance origin and whether it lies on the die edge.

    A boundary half-edge becomes one primitive, or several where it is a gap edge along part of its length.
    """
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
    pieces = [[(h, piece) for h in ring for piece in pieces_of(h)] for ring in rings]
    primitives = tuple(
        tuple(
            _primitive(f"{request_id}/ring{ri}", k, layer, h, piece, labelled)
            for k, (h, piece) in enumerate(ring_pieces)
        )
        for ri, ring_pieces in enumerate(pieces)
    )
    origin_mask = tuple(
        tuple(piece.faces_gap for _, piece in ring_pieces) for ring_pieces in pieces
    )
    die_edge_mask = tuple(
        tuple(piece.on_die_edge for _, piece in ring_pieces) for ring_pieces in pieces
    )
    return primitives, origin_mask, die_edge_mask


def _from_smallest_vertex(arr, ring: list[int]) -> list[int]:  # noqa: ANN001 - the arrangement of a layer
    first = min(range(len(ring)), key=lambda i: (arr.origin(ring[i]), ring[i]))
    return ring[first:] + ring[:first]


def _primitive(  # noqa: PLR0913 - a primitive is a ring position, a half-edge and the piece of it
    ring_id: str,
    index: int,
    layer: Layer,
    half_edge: int,
    piece: Piece,
    labelled: LabelledArrangements,
) -> RingPrimitive:
    """A piece of a boundary half-edge as a straight edge or an arc on the exact circle of its source arc.

    ``ring_id`` is ``<support>/ring<r>``; the primitive's lineage member is ``<ring_id>/edge<index>``.
    """
    curve = layer.edge_curve[half_edge >> 1]
    a, b = piece.start, piece.end
    if curve is None:
        return LineEdge(a, b, index)
    root = _root_arc(layer, half_edge, labelled)
    member = f"{ring_id}/edge{index}"
    return ArcCarrier(
        member,
        root.centre,
        root.radius,
        _is_ccw(curve.ccw, half_edge),
        (Fraction(a[0]), Fraction(a[1])),
        (Fraction(b[0]), Fraction(b[1])),
        Lineage(
            "family_member", family_id=ring_id.rsplit("/ring", 1)[0], member=member
        ),
        root.root_id,
    )


# --------------------------------------------------------------------------- faces by identity and by type


def locate_face(
    request: InsetRequest, catalogue: PrismCatalogue, labelled: LabelledArrangements
) -> FaceId:
    """The face a concrete request names, with the same checks on its entities and levels as resolving it."""
    support = concrete_support(request)
    if isinstance(support, ConductorSurface):
        level = _conductor_level(request, support, catalogue)
        return FaceId(support.side, (support.entity_id,), level)
    k = _interface_level(request, support, catalogue, labelled)
    first, second = sorted(
        (support.first_entity_id, support.second_entity_id),
        key=lambda entity: (_is_vacuum(catalogue, entity), entity),
    )
    return FaceId("interface", (first, second), labelled.levels.levels[k])


def _is_vacuum(catalogue: PrismCatalogue, entity_id: str) -> bool:
    return any(
        p.entity_id == entity_id and p.material_kind == "vacuum"
        for p in catalogue.prisms
    )


def faces_of_type(
    request_id: str,
    support: InterfaceSurfaces,
    catalogue: PrismCatalogue,
    labelled: LabelledArrangements,
) -> list[FaceId]:
    """Every face the sidecar would give a Surface-EPR row of the type, in the order of :class:`FaceId`.

    The faces come from the labels of the physical pre-pass and from the interface-type rules the sidecar's rows
    come from (``routes.naming``), so a request by type and the model it banded cannot disagree about what a
    face is.  Sidewalls are never listed (no sidewall bands in version 1).  Whether a face has a gap edge is not
    asked here.
    """
    kinds = _checked_kinds(request_id, support)
    excluded = _checked_exclusions(request_id, support, catalogue)
    if support.interface_type in ("MA", "MS"):
        faces = _conductor_faces_of_type(
            support.interface_type, kinds, excluded, catalogue, labelled
        )
    elif "interface" in kinds:
        faces = _solution_faces_of_type(excluded, labelled)
    else:
        faces = []
    return sorted(faces)


def _checked_kinds(request_id: str, support: InterfaceSurfaces) -> tuple[str, ...]:
    if support.interface_type not in BAND_TYPES:
        raise InputError(
            f"inset request {request_id!r}: bands are asked for by Surface-EPR type {', '.join(BAND_TYPES)}, "
            f"not {support.interface_type!r}"
        )
    if support.face_kinds is None:
        return FACE_KINDS
    unknown = sorted(set(support.face_kinds) - set(FACE_KINDS))
    if unknown or not support.face_kinds:
        raise InputError(
            f"inset request {request_id!r}: face_kinds must name some of {', '.join(FACE_KINDS)} "
            f"(sidewalls are not banded), got {list(support.face_kinds)}"
        )
    return support.face_kinds


def _checked_exclusions(
    request_id: str, support: InterfaceSurfaces, catalogue: PrismCatalogue
) -> frozenset[str]:
    entities = {p.entity_id for p in catalogue.prisms if p.material_kind != "port"}
    unknown = sorted(set(support.exclude_entity_ids) - entities)
    if unknown:
        raise InputError(
            f"inset request {request_id!r}: exclude_entity_ids names no entity of the model: {', '.join(unknown)}"
        )
    return frozenset(support.exclude_entity_ids)


def _conductor_faces_of_type(
    row_type: str,
    kinds: tuple[str, ...],
    excluded: frozenset[str],
    catalogue: PrismCatalogue,
    labelled: LabelledArrangements,
) -> list[FaceId]:
    """The sheets and the top and bottom faces of the conductors that have a cell with a row of ``row_type``."""
    faces: list[FaceId] = []
    conductors = {
        p.entity_id for p in catalogue.prisms if p.material_kind == "conductor"
    }
    for entity in sorted(conductors - excluded):
        prism = next(p for p in catalogue.prisms if p.entity_id == entity)
        sides: tuple[Side, ...] = ("sheet",) if prism.is_sheet else ("top", "bottom")
        for side in sides:
            level = prism.z1 if side == "top" else prism.z0
            layer = labelled.level_layers[labelled.levels.levels.index(level)]
            cells = _conductor_cells(layer, ConductorSurface(entity, side))
            if side in kinds and any(
                row_type in _surface_epr_types(layer.labels[f]) for f in cells
            ):
                faces.append(FaceId(side, (entity,), level))
    return faces


def _conductor_cells(layer: Layer, surface: ConductorSurface) -> set[int]:
    return _faces(layer, lambda label: _on_conductor(label, surface))


def _surface_epr_types(label: LevelLabel) -> set[str]:
    """The Surface-EPR row types of a level cell: its interface type, or for a Route A sheet the types of its neighbours."""
    kind = interface_type_of(label.below, label.above, label.sheet)
    if kind == SHEET:
        return set(sheet_face_sides(label.below, label.above))
    return {kind} if kind in BAND_TYPES else set()


def _solution_faces_of_type(
    excluded: frozenset[str], labelled: LabelledArrangements
) -> list[FaceId]:
    """Each substrate-air pair of solution regions on each level where they meet."""
    faces: set[FaceId] = set()
    for level, layer in zip(labelled.levels.levels, labelled.level_layers, strict=True):
        for f in range(1, len(layer.arr.faces)):
            label = layer.labels[f]
            if (
                label.sheet is not None
                or label.below is None
                or label.above is None
                or interface_type_of(label.below, label.above, None) != "SA"
            ):
                continue
            first, second = sorted(
                (label.below, label.above), key=lambda d: (d.kind == "vacuum", d.owner)
            )
            if not {first.owner, second.owner} & excluded:
                faces.add(FaceId("interface", (first.owner, second.owner), level))
    return list(faces)
