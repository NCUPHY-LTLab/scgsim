"""Content-addressed identity (brief section 7, stage S8 part 1).

Owns the hashing rules: every vertex, edge, face, cell and volume id is the
SHA-256 of a canonical JSON payload, so the same input yields the same ids
regardless of worker count, tile layout or iteration order.

* vertex  = H(plane, grid coordinate)
* edge    = H(plane, endpoint vertex ids, source carrier ids)
* cell    = H(slab, domain, sorted boundary edge ids)
* face    = H(plane, label, sorted boundary edge ids)
* volume  = H(domain, sorted cell ids)

Semantic names (``MA::<net>`` ...) live in ``routes``; lineage records are
assembled in ``assembly``. Native Gmsh tags are never part of identity.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

from sgbproto.geometry import Point


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_encode)


def _encode(value: object) -> object:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: getattr(value, k) for k in value.__dataclass_fields__}
    raise TypeError(f"cannot hash {type(value)}")


def digest(kind: str, payload: object) -> str:
    """``<kind>:<16 hex>`` identifier of a canonical payload."""
    text = canonical_json([kind, payload])
    return f"{kind}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def vertex_id(plane: str, p: Point) -> str:
    return digest("v", [plane, p[0], p[1]])


def edge_id(plane: str, p: Point, q: Point, sources: tuple[str, ...]) -> str:
    return digest("e", [plane, vertex_id(plane, p), vertex_id(plane, q), sorted(sources)])


def vertical_edge_id(p: Point, z_lo: Fraction, z_hi: Fraction) -> str:
    return digest("ve", [p[0], p[1], str(z_lo), str(z_hi)])


def cell_id(slab: str, domain: object, boundary_edge_ids: list[str]) -> str:
    return digest("c", [slab, domain, sorted(boundary_edge_ids)])


def face_id(kind: str, plane: str, label: object, boundary_edge_ids: list[str]) -> str:
    return digest("f", [kind, plane, label, sorted(boundary_edge_ids)])


def volume_id(domain: object, cell_ids: list[str]) -> str:
    return digest("vol", [domain, sorted(cell_ids)])


def plane_key_level(z: Fraction) -> str:
    return f"level[{z}]"


def plane_key_slab(z0: Fraction, z1: Fraction) -> str:
    return f"slab[{z0},{z1}]"
