"""Fragment all source footprints into shared planar cells with edge provenance.

Owns the first native stage of curved construction: build one OCC face per Entity
polygon and port sheet, fragment them together with their source edges, and record
for every resulting edge which source curve it came from. Does not decide Z
ownership (see `curved_ownership`) or name surfaces.
"""

from __future__ import annotations

from collections import defaultdict

from scgsim.geometry.construction.curved_context import PlanarArrangement
from scgsim.geometry.construction.curved_kernel import face_loops, identity
from scgsim.geometry.models.input import SourceCurveSpec


def build_planar_arrangement(session, rings, chains):
    """Build footprint faces, fragment them, and describe the retained edges."""
    gmsh = session.gmsh
    occ = session.occ
    footprints, source_edges = _footprint_faces(session, rings, chains)
    source_edges = list(dict.fromkeys(source_edges))
    occ.synchronize()
    source_footprints = [
        {
            "source_owner": owner,
            "area_um2": occ.getMass(2, face),
            "native_bounds_um": occ.getBoundingBox(2, face),
        }
        for face, owner in footprints
    ]
    # Unfragmented copies let us ask OCC for each source curve's own parameter.
    references = {tag: occ.copy([(1, tag)])[0][1] for tag in source_edges}
    cells, membership, native_edges = _fragment_footprints(
        session, footprints, source_edges, references
    )
    cell_loops = {tag: face_loops(gmsh, occ, tag) for tag in cells}
    edge_cells = defaultdict(set)
    for cell, loops in cell_loops.items():
        for loop in loops:
            for edge in loop:
                edge_cells[abs(edge)].add(cell)
    _recover_edge_history(session, native_edges, edge_cells, references)
    # Remove only scratch source/reference entities after the Boolean history has been detached.
    retained_edges = set(edge_cells)
    occ.remove(
        [(2, t) for _, t in occ.getEntities(2) if t not in cells], recursive=False
    )
    occ.remove(
        [(1, t) for _, t in occ.getEntities(1) if t not in retained_edges],
        recursive=False,
    )
    occ.synchronize()
    curve_descriptors = {
        edge: {
            "native_type": gmsh.model.getType(1, edge),
            "sources": native_edges[edge],
            "native_bounds": [
                float(u[0]) for u in gmsh.model.getParametrizationBounds(1, edge)
            ],
        }
        for edge in retained_edges
    }
    # A cell id hashes its boundary curves and covering owners, so it is stable
    # across native tag renumbering.
    cell_ids = {
        cell: identity(
            "CELL__",
            {
                "curves": sorted(
                    identity("", curve_descriptors[abs(e)])
                    for loop in loops
                    for e in loop
                ),
                "source_owners": sorted(membership[cell]),
            },
        )
        for cell, loops in cell_loops.items()
    }
    return PlanarArrangement(
        cells=cells,
        membership=membership,
        cell_loops=cell_loops,
        edge_cells=edge_cells,
        retained_edges=retained_edges,
        curve_descriptors=curve_descriptors,
        cell_ids=cell_ids,
        native_edges=native_edges,
        source_footprints=source_footprints,
    )


def _footprint_faces(session, rings, chains):
    """Create one face per Entity polygon, then one per port sheet, in that order."""
    kernel = session.kernel
    footprints = []
    source_edges = []
    for entity_id, records in rings.items():
        for pid, _outer, holes in records:
            sources = []
            for role, index in [
                ("outer", None),
                *[("hole", i) for i in range(len(holes))],
            ]:
                key = (entity_id, pid, role, index)
                sources.append(
                    (
                        chains[key],
                        {
                            "entity_id": entity_id,
                            "polygon_id": pid,
                            "role": role,
                            "hole_index": index,
                        },
                    )
                )
            face, edges = kernel.face(sources)
            footprints.append((face, entity_id))
            source_edges.extend(edges)
    for port in session.prepared.port_sheet_regions:
        sources = []
        for role, index, ring in [
            ("outer", None, port.exterior),
            *[("hole", i, h) for i, h in enumerate(port.holes)],
        ]:
            curves = tuple(
                SourceCurveSpec("line_segment", (tuple(a), tuple(b)))
                for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
            )
            sources.append(
                (
                    curves,
                    {
                        "port_sheet_id": port.port_sheet_id,
                        "role": role,
                        "hole_index": index,
                    },
                )
            )
        face, edges = kernel.face(sources)
        footprints.append((face, "PORT::" + port.port_sheet_id))
        source_edges.extend(edges)
    return footprints, source_edges


def _fragment_footprints(session, footprints, source_edges, references):
    """Fragment footprint faces with their source edges; read back cells and history."""
    gmsh = session.gmsh
    occ = session.occ
    output, mapping = occ.fragment(
        [(2, face) for face, _ in footprints], [(1, tag) for tag in source_edges]
    )
    occ.synchronize()
    cells = {tag for dim, tag in output if dim == 2}
    membership = defaultdict(set)
    for (_, owner), mapped in zip(footprints, mapping[: len(footprints)], strict=False):
        for dim, tag in mapped:
            if dim == 2:
                membership[tag].add(owner)
    native_edges = defaultdict(list)
    for source, mapped in zip(source_edges, mapping[len(footprints) :], strict=False):
        reference = references[source]
        for dim, tag in mapped:
            if dim != 1:
                continue
            bounds = gmsh.model.getParametrizationBounds(1, tag)
            endpoint_values = [
                gmsh.model.getValue(1, tag, [float(u[0])]).tolist() for u in bounds
            ]
            source_interval = [
                float(gmsh.model.getParametrization(1, reference, xyz)[0])
                for xyz in endpoint_values
            ]
            native_edges[tag].append(
                {
                    **session.kernel.source_edges[source],
                    "source_parameter_interval": source_interval,
                    "native_parameter_bounds": [float(u[0]) for u in bounds],
                }
            )
    return cells, membership, native_edges


def _recover_edge_history(session, native_edges, edge_cells, references):
    """Give every edge without Boolean history a source curve via common geometry."""
    gmsh = session.gmsh
    occ = session.occ
    # OCC may omit lower-dimensional history for an unchanged exterior
    # wire while replacing its native tags during a face Boolean. Recover
    # that history through native common geometry, never endpoint matching
    # or a caller-chosen geometric tolerance.
    for edge in edge_cells:
        if native_edges[edge]:
            continue
        for source, reference in references.items():
            existing = set(occ.getEntities())
            common, _ = occ.intersect(
                [(1, edge)], [(1, reference)], removeObject=False, removeTool=False
            )
            occ.synchronize()
            for dim, common_edge in common:
                if dim != 1:
                    continue
                bounds = gmsh.model.getParametrizationBounds(1, edge)
                common_bounds = gmsh.model.getParametrizationBounds(1, common_edge)
                values = [
                    gmsh.model.getValue(1, common_edge, [float(u[0])])
                    for u in common_bounds
                ]
                native_edges[edge].append(
                    {
                        **session.kernel.source_edges[source],
                        "source_parameter_interval": [
                            float(gmsh.model.getParametrization(1, reference, xyz)[0])
                            for xyz in values
                        ],
                        "arrangement_parameter_interval": [
                            float(gmsh.model.getParametrization(1, edge, xyz)[0])
                            for xyz in values
                        ],
                        "native_parameter_bounds": [float(u[0]) for u in bounds],
                        "native_history_method": "occ_common_geometry",
                    }
                )
            occ.remove(
                [entity for entity in common if entity not in existing],
                recursive=False,
            )
        if not native_edges[edge]:
            raise ValueError(
                "native arrangement boundary has no source curve provenance"
            )
