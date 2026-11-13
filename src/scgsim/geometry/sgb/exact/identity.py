"""Content-addressed identity (design brief section 7, stage S8 part 1).

Owns the hashing rules: every vertex, edge, face, cell and volume id is the
SHA-256 of a canonical JSON payload, so the same input yields the same ids
regardless of iteration order.

* vertex = H(plane, grid coordinate)
* edge   = H(plane, sorted endpoint vertex ids, sorted source carrier ids)
* cell   = H(slab, domain, sorted boundary edge ids)
* face   = H(plane, label, sorted boundary edge ids); a wall's id uses its horizontal
  boundary chains and the columns' end points only, never the vertical edges, which
  change whenever a neighbouring wall starts or ends on the column
* volume = H(domain, sorted cell ids)

Semantic names (``MA::<entity>__<face kind>`` ...) belong to later stages.  Native Gmsh tags are
never part of identity.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

from scgsim.geometry.sgb.exact.plane import Point

ID_HEX_DIGITS = 16


def canonical_json(payload: object) -> str:
    """Key-sorted, whitespace-free JSON of a payload (dataclasses and sets included)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_encode)


def _encode(value: object) -> object:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return {name: getattr(value, name) for name in fields}
    raise TypeError(f"internal: cannot hash {type(value)}")


def digest(kind: str, payload: object) -> str:
    """``<kind>:<16 hex>`` identifier of a canonical payload."""
    text = canonical_json([kind, payload])
    return f"{kind}:{hashlib.sha256(text.encode()).hexdigest()[:ID_HEX_DIGITS]}"


def vertex_id(plane: str, p: Point) -> str:
    """Id of a vertex of a plane."""
    return digest("v", [plane, p[0], p[1]])


def edge_id(plane: str, p: Point, q: Point, sources: tuple[str, ...]) -> str:
    """Id of an edge: its end vertices and the source carriers that produced it."""
    ends = sorted([vertex_id(plane, p), vertex_id(plane, q)])
    return digest("e", [plane, ends, sorted(sources)])


def cell_id(slab: str, domain: object, boundary_edge_ids: list[str]) -> str:
    """Id of a slab cell."""
    return digest("c", [slab, domain, sorted(boundary_edge_ids)])


def face_id(kind: str, plane: str, label: object, boundary_edge_ids: list[str]) -> str:
    """Id of a horizontal face of a level."""
    return digest("f", [kind, plane, label, sorted(boundary_edge_ids)])


def vertical_edge_id(p: Point, z_lo: Fraction, z_hi: Fraction) -> str:
    """Id of a vertical edge on the column through ``p`` between two levels."""
    return digest("ve", [p[0], p[1], str(z_lo), str(z_hi)])


def volume_id(domain: object, cell_ids: list[str]) -> str:
    """Id of a volume: its domain and the cells it is made of."""
    return digest("vol", [domain, sorted(cell_ids)])


def plane_key_level(z: Fraction) -> str:
    """Plane key of a level arrangement."""
    return f"level[{z}]"


def plane_key_slab(z0: Fraction, z1: Fraction) -> str:
    """Plane key of a slab arrangement."""
    return f"slab[{z0},{z1}]"
