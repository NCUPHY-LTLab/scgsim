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
may meet at a corner; only slab arrangements are examined.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.errors import ContactError, NonManifoldError
from scgsim.geometry.sgb.labelling.prism_set import FIX_NETS, named_entities
from scgsim.geometry.sgb.plane import Point, point_in_ring, segment_point_distance_sq
from scgsim.geometry.sgb.records.labels import Domain, LevelLabel, SheetOwner
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord

NEAR_GRID_UNITS_SQ = 4  # a prism "reaches" a point within two grid units (squared)


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
    defect as an edge "used 4 times".
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
                    _raise_pinch(catalogue, layer, domain, arr.vertices[vi])


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
    catalogue: PrismCatalogue, layer: Layer, domain: Domain, point: Point
) -> None:
    grid = catalogue.grid
    z_lo, z_hi = _layer_z_range(layer)
    if domain.kind == "conductor":
        reaching = [
            p for p in catalogue.prisms if _reaches(p, domain.net, point, z_lo, z_hi)
        ]
        what = f"Net {domain.net} (entities {named_entities(reaching) or '?'})"
    else:
        around = [
            p
            for p in catalogue.prisms
            if p.material_kind == "conductor"
            and _reaches(p, p.net_id or "", point, z_lo, z_hi)
        ]
        names = named_entities(around) or "?"
        nets = ", ".join(sorted({p.net_id or "" for p in around}))
        what = f"{domain.name} (pinched between conductors {names} of Net {nets})"
    raise NonManifoldError(
        f"non-manifold pinch: {what} touches itself along the vertical edge at "
        f"({grid.to_um(point[0]):.4f}, {grid.to_um(point[1]):.4f}) um, "
        f"z {float(z_lo)}..{float(z_hi)} um ({layer.plane}): two cells of one domain "
        "meet only at a vertex. Fix the layout: give them a shared edge or separate them."
    )


def _layer_z_range(layer: Layer) -> tuple[Fraction, Fraction]:
    if isinstance(layer.z, tuple):
        return layer.z
    return layer.z, layer.z


def _reaches(
    prism: PrismRecord, net: str, point: Point, z_lo: Fraction, z_hi: Fraction
) -> bool:
    """Whether the footprint of a conductor prism of ``net`` touches the point at that Z."""
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
