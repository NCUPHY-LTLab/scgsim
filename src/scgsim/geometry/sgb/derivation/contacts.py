"""Different-Net contact and non-manifold checks (design brief section 14).

A different-Net contact is an error in every route and on every plane.  Overlap
is already rejected while labelling (``labelling.prism_set``); this module catches
the *touching* cases, including a contact that exists only on the grid -- a
coarse grid, or snapping, that closes a source gap between different Nets.

The test is combinatorial: a vertex (or a face) of a dissolved arrangement whose
incident faces carry conductors of more than one Net is a contact.  Each
arrangement vertex is a point where its incident faces' footprints meet, so this
finds shared edges and shared corners alike.  Ports and dielectrics never count.

The error names both Nets, the entities of each that reach the contact point
(footprint within two grid units at that Z) with their source layers, the location
in micrometres, the plane and who must act.

A non-manifold pinch is the same-Net counterpart: finite metal (or the vacuum
between metal) that meets itself only along a vertical edge.  Sheets of one Net
may meet at a corner; only slab arrangements are examined.  The vacuum also
pinches where a conductor, or a region that is not one such as a finite
dielectric, reaches the model's planar boundary at a single point (a polygon
circle or a diamond with nothing drawn around it, when the vacuum ends at the
layout's extent); that error says so, names what reaches the boundary and what
to change.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import cmp_to_key

from scgsim.documentation import see_also
from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.errors import ContactError, NonManifoldError
from scgsim.geometry.sgb.labelling.prism_set import FIX_NETS, named_entities
from scgsim.geometry.sgb.plane import Point, point_in_ring, segment_point_distance_sq
from scgsim.geometry.sgb.records.labels import Domain, LevelLabel, SheetOwner
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord

NEAR_GRID_UNITS_SQ = 4  # a prism "reaches" a point within two grid units (squared)
# Who must act when a conductor reaches the vacuum's edge at one point: something drawn around the conductor
# moves the model's planar boundary away from it, and a vacuum region with planar padding does the same.
_FIX_AT_THE_EDGE = "Fix the layout: draw the die outline or a ground plane around it, or set a vacuum region."
# A region that is not a conductor, such as a dielectric island, can instead meet the boundary along an edge or
# keep clear of it; a ground plane is no fix for it.
_FIX_REGION_AT_THE_EDGE = (
    "Fix the layout: draw the die outline around it, or change that region's outline so it meets "
    "the boundary along an edge or not at all."
)


def check_contacts(catalogue: PrismCatalogue, layers: Sequence[Layer]) -> None:
    """Raise :class:`ContactError` for the first different-Net contact found (deterministic order)."""
    for layer in layers:
        arr = layer.arr
        nets_of_face = [
            _conductor_nets(layer.labels[fi]) for fi in range(len(arr.faces))
        ]
        for fi in range(1, len(arr.faces)):
            if (
                len(nets_of_face[fi]) > 1
            ):  # stacked face to face (below one Net, above another)
                _raise_contact(catalogue, layer, nets_of_face[fi], arr.face_anchor(fi))
        around: dict[int, set[str]] = {}
        for ei, (a, b) in enumerate(arr.edge_ends):
            nets = (
                nets_of_face[arr.he_face[2 * ei]]
                | nets_of_face[arr.he_face[2 * ei + 1]]
            )
            for vertex in (a, b):
                around.setdefault(vertex, set()).update(nets)
        for vi in sorted(around):
            if len(around[vi]) > 1:
                _raise_contact(catalogue, layer, around[vi], arr.vertices[vi])


def check_manifold(catalogue: PrismCatalogue, slabs: Sequence[Layer]) -> None:
    """Raise :class:`NonManifoldError` where a volume would use a vertical edge four times.

    In a slab arrangement the wedges around a vertex are the faces to the left of
    its outgoing half-edges.  If more than one wedge carries the same (non-empty)
    domain, that volume touches itself at the vertex: two cells of one Net meeting
    at a corner, a hole touching its outer boundary, or the vacuum pinched
    between them.  This runs before shell assembly, which would report the same
    defect as an edge "used 4 times".  The refusal names what the arrangement
    itself puts around the vertex, with no distance tolerance (``_raise_pinch``).
    """
    for layer in slabs:
        arr = layer.arr
        wedges: dict[int, dict[Domain, list[int]]] = {}
        for ei, (a, b) in enumerate(arr.edge_ends):
            for half_edge, vertex in (
                (2 * ei, a),
                (2 * ei + 1, b),
            ):  # half-edge leaves `vertex`
                domain = layer.labels[arr.he_face[half_edge]]
                if domain is not None:
                    wedges.setdefault(vertex, {}).setdefault(domain, []).append(
                        arr.he_face[half_edge]
                    )
        for vi in sorted(wedges):
            by_domain = sorted(
                wedges[vi].items(), key=lambda kv: (kv[0].kind != "conductor", kv[0])
            )
            for domain, faces in by_domain:
                if len(faces) > 1:
                    _raise_pinch(catalogue, layer, domain, vi)


def _conductor_nets(label: object) -> frozenset[str]:
    if label is None:
        return frozenset()
    if isinstance(label, Domain):
        return frozenset([label.net]) if label.kind == "conductor" else frozenset()
    assert isinstance(label, LevelLabel)
    nets = {
        d.net
        for d in (label.below, label.above)
        if d is not None and d.kind == "conductor"
    }
    if isinstance(label.sheet, SheetOwner) and label.sheet.kind == "conductor":
        nets.add(label.sheet.net)
    return frozenset(nets)


def _raise_contact(
    catalogue: PrismCatalogue,
    layer: Layer,
    nets: set[str] | frozenset[str],
    point: Point,
) -> None:
    grid = catalogue.grid
    z_lo, z_hi = _layer_z_range(layer)
    parts = []
    for net in sorted(nets):
        reaching = [p for p in catalogue.prisms if _reaches(p, net, point, z_lo, z_hi)]
        parts.append(f"{net} (entities {named_entities(reaching) or '?'})")
    raise ContactError(
        f"different-Net contact between {' and '.join(parts)} at "
        f"({grid.to_um(point[0]):.4f}, {grid.to_um(point[1]):.4f}) um "
        f"on {layer.plane} (grid g = {grid.size_um} um; "
        f"a gap smaller than the grid closes when the sources snap). {FIX_NETS}"
    )


def _raise_pinch(
    catalogue: PrismCatalogue, layer: Layer, domain: Domain, vertex: int
) -> None:
    """Raise the pinch of ``domain`` at ``vertex``, naming what separates its wedges there.

    The pinch point is a vertex of the slab arrangement, an exact grid point, and the wedges round it carry
    the arrangement's own labels, so what the message names needs no distance tolerance.  A wedge with no
    domain is outside the model: the vertex then lies on the model's planar boundary.
    """
    grid = catalogue.grid
    z_lo, z_hi = _layer_z_range(layer)
    point = layer.arr.vertices[vertex]
    where = (
        f"({grid.to_um(point[0]):.4f}, {grid.to_um(point[1]):.4f}) um, "
        f"z {float(z_lo)}..{float(z_hi)} um ({layer.plane})"
    )
    wedges = _wedges_around(layer, vertex)
    if domain.kind == "conductor":
        same_net = [d for d in wedges if d is not None and d.net == domain.net]
        what = (
            f"Net {domain.net} (entities {_named(catalogue, same_net, (z_lo, z_hi))})"
        )
    else:
        # Should every run between its wedges pass the outside, name what touches the vertex instead.
        separating = _separating(wedges, domain) or [
            d for d in dict.fromkeys(wedges) if d is not None and d != domain
        ]
        conductors = [d for d in separating if d.kind == "conductor"]
        names = _named(catalogue, conductors, (z_lo, z_hi))
        nets = ", ".join(sorted({d.net for d in conductors}))
        # Only a pinch between conductors is named by them; otherwise by the regions that separate it.
        regions = _region_phrase(
            catalogue, [d for d in separating if d.kind != "conductor"], (z_lo, z_hi)
        )
        if domain.kind == "vacuum" and None in wedges:
            if conductors:
                _raise_edge_pinch(domain, conductors, names, nets, where)
            _raise_region_edge_pinch(domain, regions, where)
        what = (
            f"{domain.name} (pinched between conductors {names} of Net {nets})"
            if conductors
            else f"{domain.name} (pinched between {regions[0]})"
        )
    # A pinch means nothing to most layout authors, so the message ends with the section that explains it.
    raise NonManifoldError(
        f"non-manifold pinch: {what} touches itself along the vertical edge at "
        f"{where}: two cells of one domain meet only at a vertex. "
        "Fix the layout: give them a shared edge or separate them."
        + see_also("docs/known-geometry-problems", "pinch-inside-the-layout")
    )


def _raise_edge_pinch(
    vacuum: Domain,
    conductors: Sequence[Domain],
    names: str,
    nets: str,
    where: str,
) -> None:
    """The vacuum ends where the ``conductors`` reach the model's planar boundary at one point."""
    one = len(set(conductors)) == 1
    raise NonManifoldError(
        f"non-manifold pinch: {'conductor' if one else 'conductors'} {names} of Net {nets} "
        f"{'reaches' if one else 'reach'} the edge of {vacuum.name} only at {where}, a point on "
        f"the model's planar boundary, so {vacuum.name} touches itself along that vertical edge. "
        + _FIX_AT_THE_EDGE
        + see_also("docs/known-geometry-problems", "boundary-point-pinch")
    )


def _raise_region_edge_pinch(
    vacuum: Domain, regions: tuple[str, bool], where: str
) -> None:
    """The vacuum ends where a region that is not a conductor, such as a dielectric island, meets the boundary at one point."""
    phrase, one = regions
    raise NonManifoldError(
        f"non-manifold pinch: {phrase} {'reaches' if one else 'reach'} the edge of {vacuum.name} "
        f"only at {where}, a point on the model's planar boundary, so {vacuum.name} touches itself "
        "along that vertical edge. "
        + _FIX_REGION_AT_THE_EDGE
        + see_also("docs/known-geometry-problems", "boundary-point-pinch")
    )


def _wedges_around(layer: Layer, vertex: int) -> list[Domain | None]:
    """The labels of the wedges round ``vertex``, counter-clockwise; ``None`` is outside the model.

    Each wedge is the face to the left of an outgoing half-edge, so ordering the outgoing half-edges by
    direction orders the wedges.  The directions are integer vectors, compared exactly.
    """
    arr = layer.arr
    outgoing = [
        half_edge
        for ei, (a, b) in enumerate(arr.edge_ends)
        for half_edge, origin in ((2 * ei, a), (2 * ei + 1, b))
        if origin == vertex
    ]

    def direction(half_edge: int) -> tuple[int, int]:
        (x0, y0), (x1, y1) = arr.origin(half_edge), arr.target(half_edge)
        return x1 - x0, y1 - y0

    ordered = sorted(
        outgoing, key=cmp_to_key(lambda h, k: _ccw(direction(h), direction(k)))
    )
    return [layer.labels[arr.he_face[half_edge]] for half_edge in ordered]


def _ccw(first: tuple[int, int], second: tuple[int, int]) -> int:
    """Compare two directions by their angle from +x, counter-clockwise, exactly."""
    upper = (
        first[1] > 0 or (first[1] == 0 and first[0] > 0),
        second[1] > 0 or (second[1] == 0 and second[0] > 0),
    )
    if upper[0] != upper[1]:
        return -1 if upper[0] else 1
    turn = first[0] * second[1] - first[1] * second[0]
    return -1 if turn > 0 else (1 if turn < 0 else 0)


def _separating(wedges: list[Domain | None], pinched: Domain) -> list[Domain]:
    """The domains between two wedges of ``pinched`` round the vertex: what pinches it.

    Going round from one wedge of ``pinched`` to the next, the wedges passed separate them.  A run that
    passes the outside of the model is the boundary side, where the model simply ends; the domains on it
    lie along the boundary and pinch nothing.
    """
    start = wedges.index(pinched)
    found: list[Domain] = []
    run: list[Domain | None] = []
    for label in [*wedges[start + 1 :], *wedges[:start], pinched]:
        if label != pinched:
            run.append(label)
            continue
        if None not in run:
            found += [d for d in run if d is not None and d not in found]
        run = []
    return found


def _named(
    catalogue: PrismCatalogue,
    domains: Sequence[Domain],
    z_range: tuple[Fraction, Fraction],
) -> str:
    """The domains' Entities with their source layers, from the prisms that fill the slab.

    Which domains a pinch names comes from the arrangement, so no distance enters here; the prisms only
    supply the layers.  A prism must fill the slab, not only touch it from below or above: one that ends
    where the slab begins (a pad under a dielectric island) is not where the domain is in this slab.
    """
    owners = {d.name for d in domains}
    z_lo, z_hi = z_range
    prisms = [
        p
        for p in catalogue.prisms
        if p.entity_id in owners and p.z0 < z_hi and p.z1 > z_lo
    ]
    return named_entities(prisms) or ", ".join(sorted(owners)) or "?"


def _region_phrase(
    catalogue: PrismCatalogue,
    regions: Sequence[Domain],
    z_range: tuple[Fraction, Fraction],
) -> tuple[str, bool]:
    """``dielectric region island`` or ``dielectric regions a, b``, and whether it names one region."""
    one = len(set(regions)) == 1
    kinds = ", ".join(sorted({d.kind for d in regions})) or "?"
    names = _named(catalogue, regions, z_range)
    return f"{kinds} {'region' if one else 'regions'} {names}", one


def _layer_z_range(layer: Layer) -> tuple[Fraction, Fraction]:
    if isinstance(layer.z, tuple):
        return layer.z
    return layer.z, layer.z


def _reaches(
    prism: PrismRecord, net: str, point: Point, z_lo: Fraction, z_hi: Fraction
) -> bool:
    """Whether the footprint of a conductor prism of ``net`` touches the point at that Z.

    For the different-Net contact message.  The Z test is inclusive, because a contact found on a level plane
    is between prisms that end and begin there, and a footprint within two grid units counts, because the
    contact point is a vertex of the snap-rounded arrangement, which may lie off the source ring.
    """
    if (
        prism.material_kind != "conductor"
        or prism.net_id != net
        or prism.z0 > z_hi
        or prism.z1 < z_lo
    ):
        return False
    for ring in prism.footprint.rings():
        for i in range(len(ring)):
            distance_sq = segment_point_distance_sq(
                ring[i], ring[(i + 1) % len(ring)], point
            )
            if distance_sq <= NEAR_GRID_UNITS_SQ:
                return True
    inside = sum(
        point_in_ring(ring, (point[0], point[1], 1)) for ring in prism.footprint.rings()
    )
    return inside % 2 == 1
