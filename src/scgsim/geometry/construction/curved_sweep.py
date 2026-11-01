"""Sweep the shared planar curve complex along Z into native walls.

Owns the 1D to 2D native sweep: copy every retained planar edge to the first Z
level, extrude each slab, and record the lifted edges, shared vertical edges and
swept walls. Does not create caps, surfaces or volumes.
"""

from __future__ import annotations

from scgsim.geometry.construction.curved_context import SweptComplex
from scgsim.geometry.construction.curved_kernel import _edge_vertices


def sweep_curve_complex(session, arrangement, z_values):
    """Sweep the entire shared curve complex together, strictly 1D -> 2D.

    A native prism preserves trimmed rational/spline geometry and shares the
    vertical edge of each source vertex across adjacent walls.
    """
    gmsh = session.gmsh
    occ = session.occ
    retained_edges = arrangement.retained_edges
    lifted = _copy_edges_to_base_level(session, retained_edges, z_values[0])
    occ.synchronize()
    lifted_vertices = {}
    for edge in retained_edges:
        for source, copied in zip(
            _edge_vertices(gmsh, edge),
            _edge_vertices(gmsh, lifted[0, edge]),
            strict=False,
        ):
            lifted_vertices[0, source] = copied
    vertical = {}
    swept_walls = {}
    for slab, (lo, hi) in enumerate(zip(z_values, z_values[1:], strict=False)):
        _sweep_slab(
            session,
            retained_edges,
            slab,
            hi - lo,
            (lifted, lifted_vertices, vertical, swept_walls),
        )
    return SweptComplex(lifted=lifted, vertical=vertical, swept_walls=swept_walls)


def _copy_edges_to_base_level(session, retained_edges, base_z):
    """Copy each planar edge to the lowest Z level and rebind it after fragmenting."""
    occ = session.occ
    lifted = {}
    for edge in retained_edges:
        tag = occ.copy([(1, edge)])[0][1]
        occ.translate([(1, tag)], 0, 0, base_z)
        lifted[0, edge] = tag
    _, copied_history = occ.fragment(
        [(1, lifted[0, edge]) for edge in retained_edges], []
    )
    for edge, history in zip(retained_edges, copied_history, strict=False):
        tags = [tag for dim, tag in history if dim == 1]
        if len(tags) != 1:
            raise ValueError("Z-copy changed an already arranged planar edge")
        lifted[0, edge] = tags[0]
    return lifted


def _sweep_slab(session, retained_edges, slab, height, tables):
    """Extrude one slab; `tables` are the lifted, vertex, vertical and wall dicts."""
    gmsh = session.gmsh
    occ = session.occ
    lifted, lifted_vertices, vertical, swept_walls = tables
    source_by_tag = {lifted[slab, edge]: edge for edge in retained_edges}
    source_by_vertex = {
        tag: vertex for (level, vertex), tag in lifted_vertices.items() if level == slab
    }
    output = occ.extrude([(1, tag) for tag in source_by_tag], 0, 0, height)
    occ.synchronize()
    for dim, face in output:
        if dim != 2:
            continue
        boundary = [
            abs(tag)
            for d, tag in gmsh.model.getBoundary([(2, face)], oriented=True)
            if d == 1
        ]
        bottom_tags = [tag for tag in boundary if tag in source_by_tag]
        if len(bottom_tags) != 1:
            raise ValueError(
                "native edge sweep did not retain one canonical source edge per wall"
            )
        edge = source_by_tag[bottom_tags[0]]
        swept_walls[slab, edge] = face
        for tag in boundary:
            if tag == bottom_tags[0]:
                continue
            vertices = _edge_vertices(gmsh, tag)
            lower_vertices = [v for v in vertices if v in source_by_vertex]
            if not lower_vertices:
                lifted[slab + 1, edge] = tag
            elif len(lower_vertices) == 1:
                lower = lower_vertices[0]
                vertex = source_by_vertex[lower]
                key = (slab, vertex)
                if key in vertical and vertical[key] != tag:
                    raise ValueError(
                        "native collective sweep duplicated a shared vertical edge"
                    )
                vertical[key] = tag
                lifted_vertices[slab + 1, vertex] = next(
                    v for v in vertices if v != lower
                )
            else:
                raise ValueError(
                    "native edge sweep returned an unexpected horizontal boundary"
                )
        if (slab + 1, edge) not in lifted:
            raise ValueError("native edge sweep lost the canonical upper edge")
    if any((slab, edge) not in swept_walls for edge in retained_edges):
        raise ValueError("native edge sweep did not cover the arranged curve complex")
