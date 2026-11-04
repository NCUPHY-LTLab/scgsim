"""Different-Net contact check (brief section 14): an error in every route, on every plane.

Owns detecting two conductors of different Nets that touch: share an edge or a vertex of a slab arrangement
(finite metal in Route B), a sheet plane (Route A sheets, or a sheet next to finite metal), or are stacked
face-to-face across a level. Overlap is already rejected while labelling (``labels``); this module catches the
*touching* cases, including a contact that exists only on the grid -- a coarse grid, or snapping, that closes a
source gap between different Nets.

The test is combinatorial: a vertex (or a face) of a dissolved arrangement whose incident faces carry conductors
of more than one Net is a contact. Each arrangement vertex is a point where its incident faces' footprints meet,
so this finds shared edges and shared corners alike. Ports and dielectrics never count.

The error names both Nets, the entities of each that reach the contact point (footprint within two grid units
at that Z), the location in micrometres and the plane.
"""

from __future__ import annotations

from sgbproto.errors import ContactError, NonManifoldError
from sgbproto.geometry import point_in_ring, segment_point_distance_sq
from sgbproto.records import Domain, LevelLabel, ModelSpec, PrismRecord, SheetOwner


def check_contacts(spec: ModelSpec, layers) -> None:
    """Raise :class:`ContactError` for the first different-Net contact found (deterministic order)."""
    for layer in layers:
        arr = layer.arr
        nets_of_face = [_conductor_nets(layer.labels[fi]) for fi in range(len(arr.faces))]
        for fi in range(1, len(arr.faces)):
            if len(nets_of_face[fi]) > 1:  # stacked face to face (below one Net, above another)
                x, y, _ = _anchor(layer, fi)
                _raise(spec, layer, nets_of_face[fi], (x, y))
        around: dict[int, set[str]] = {}
        for ei, (a, b) in enumerate(arr.edge_ends):
            nets = nets_of_face[arr.he_face[2 * ei]] | nets_of_face[arr.he_face[2 * ei + 1]]
            for v in (a, b):
                around.setdefault(v, set()).update(nets)
        for vi in sorted(around):
            if len(around[vi]) > 1:
                _raise(spec, layer, around[vi], arr.vertices[vi])


def check_manifold(spec: ModelSpec, slabs) -> None:
    """Raise :class:`NonManifoldError` where a volume would use a vertical edge four times.

    In a slab arrangement the wedges around a vertex are the faces to the left of its outgoing half-edges. If more
    than one wedge carries the same (non-empty) domain, that volume touches itself at the vertex: two cells of one
    Net meeting at a corner, a hole touching its outer boundary, or the vacuum pinched between them. Same-Net
    sheets (Route A) are not affected: only slab arrangements are examined, and sheets live on levels.
    This runs before shell assembly (which would report the same defect as an edge "used 4 times").
    """
    for layer in slabs:
        arr = layer.arr
        wedges: dict[int, dict] = {}
        for ei, (a, b) in enumerate(arr.edge_ends):
            for h, vertex in ((2 * ei, a), (2 * ei + 1, b)):  # half-edge h leaves `vertex`
                domain = layer.labels[arr.he_face[h]]
                if domain is not None:
                    wedges.setdefault(vertex, {}).setdefault(domain, []).append(arr.he_face[h])
        for vi in sorted(wedges):
            for domain, faces in sorted(wedges[vi].items(), key=lambda kv: (kv[0].kind != "conductor", kv[0])):
                if len(faces) > 1:
                    _raise_pinch(spec, layer, domain, arr.vertices[vi])


def _raise_pinch(spec: ModelSpec, layer, domain: Domain, point) -> None:
    g = spec.grid.g
    z_lo, z_hi = layer.z
    if domain.kind == "conductor":
        names = sorted(p.entity_id for p in spec.prisms if _reaches(p, domain.net, point, z_lo, z_hi))
        what = f"Net {domain.net} (entities {', '.join(names) or '?'})"
    else:
        around = sorted(p.entity_id for p in spec.prisms if p.material_kind == "conductor" and _reaches(p, p.net_id, point, z_lo, z_hi))
        what = f"{domain.name} (pinched between conductors {', '.join(around) or '?'})"
    raise NonManifoldError(
        f"non-manifold pinch: {what} touches itself along the vertical edge at ({point[0] * g:.4f}, {point[1] * g:.4f}) um, "
        f"z {float(z_lo)}..{float(z_hi)} um ({layer.plane}): two cells of one domain meet only at a vertex; "
        "give them a shared edge or separate them"
    )


def _conductor_nets(label) -> frozenset[str]:
    if label is None:
        return frozenset()
    if isinstance(label, Domain):
        return frozenset([label.net]) if label.kind == "conductor" else frozenset()
    assert isinstance(label, LevelLabel)
    nets = {d.net for d in (label.below, label.above) if d is not None and d.kind == "conductor"}
    if isinstance(label.sheet, SheetOwner) and label.sheet.kind == "conductor":
        nets.add(label.sheet.name)
    return frozenset(nets)


def _anchor(layer, fi: int):
    arr = layer.arr
    p = arr.origin(arr.cycles[arr.faces[fi].outer].half_edges[0])
    return p[0], p[1], None


def _raise(spec: ModelSpec, layer, nets, point) -> None:
    g = spec.grid.g
    z = layer.z if layer.kind == "level" else layer.z[0]
    z_hi = layer.z if layer.kind == "level" else layer.z[1]
    parts = []
    for net in sorted(nets):
        names = sorted(p.entity_id for p in spec.prisms if _reaches(p, net, point, z, z_hi))
        parts.append(f"{net} (entities {', '.join(names) or '?'})")
    raise ContactError(
        f"different-Net contact between {' and '.join(parts)} at ({point[0] * g:.4f}, {point[1] * g:.4f}) um "
        f"on {layer.plane} (grid g = {g} um; a gap smaller than the grid closes when the sources snap)"
    )


def _reaches(prism: PrismRecord, net: str, point, z_lo, z_hi) -> bool:
    if prism.material_kind != "conductor" or prism.net_id != net or prism.z0 > z_hi or prism.z1 < z_lo:
        return False
    p = (round(point[0]), round(point[1]))
    for ring in prism.footprint.rings():
        for i in range(len(ring)):
            if segment_point_distance_sq(ring[i], ring[(i + 1) % len(ring)], p) <= 4:
                return True
    return sum(point_in_ring(ring, (p[0], p[1], 1)) for ring in prism.footprint.rings()) % 2 == 1  # inside
