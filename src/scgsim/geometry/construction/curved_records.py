"""Turn registered native surfaces into the plan's curve, loop and surface records.

Owns the record stage of curved construction: detach each used native edge into a
`CurvePlanRecord` with content-addressed ids, build loops, surface and interface
records, hidden metal-metal contacts, and resolve lumped-port terminals. Reads
native topology but creates no native entities and no volumes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace

from scgsim.geometry.construction.curved_context import EdgeLookup
from scgsim.geometry.construction.curved_kernel import (
    edge_vertices,
    face_loops,
    identity,
)
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    CurveRefRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.surface_contract import with_surface_contract_metadata
from scgsim.geometry.primitives.entities import component_metadata


@dataclass
class CurveTable:
    """Detached curve and point records for every edge used by a surface."""

    used_edges: set
    # Native edge tag -> curve id.
    curve_ids: dict
    curve_records: list
    # Point id -> xyz.
    points: dict
    # Point id -> ids of the curves ending there.
    point_curves: dict
    lookup: EdgeLookup


@dataclass
class SurfaceTable:
    """Loop, surface and interface records plus per-curve usage indexes."""

    loops: list
    surfaces: list
    interfaces: list
    curve_surfaces: dict
    curve_owners: dict
    curve_volumes: dict
    curve_interfaces: dict


def _detached_edge(gmsh, edge, lookup):
    """Describe one native edge by geometry alone, independent of native tags."""
    endpoints = [
        tuple(float(v) for v in gmsh.model.getValue(0, tag, []))
        for tag in edge_vertices(gmsh, edge)
    ]
    if len(endpoints) != 2:
        raise ValueError(
            "native arrangement edge lacks two detached topology endpoints"
        )
    if edge in lookup.lifted_inverse:
        level, source = lookup.lifted_inverse[edge]
        geometry = {
            **lookup.curve_descriptors[source],
            "z_um": lookup.z_values[level],
        }
        kinds = {record["geometry"]["kind"] for record in lookup.native_edges[source]}
        kind = next(iter(kinds)) if len(kinds) == 1 else "bspline"
    elif edge in lookup.vertical_inverse:
        geometry = {"kind": "line_segment", "endpoints_um": endpoints}
        kind = "line_segment"
    else:
        raise ValueError(
            "native sidewall construction replaced a planned shared boundary curve"
        )
    cid = identity("C__", geometry)
    pids = [identity("P__", xyz) for xyz in endpoints]
    bounds = gmsh.model.getParametrizationBounds(1, edge)
    return CurvePlanRecord(
        cid,
        kind,
        *pids,
        geometry=geometry,
        parameter_interval=(float(bounds[0][0]), float(bounds[1][0])),
    ), dict(zip(pids, endpoints, strict=False))


def build_curve_table(context, book):
    """Detach every native edge used by a registered surface, in tag order."""
    gmsh = context.session.gmsh
    # Native history and geometry determine curve identity; endpoints alone cannot identify a curve.
    used_edges = {
        abs(t)
        for tag in book.native_surfaces.values()
        for _, t in gmsh.model.getBoundary([(2, tag)], oriented=True)
    }
    lookup = EdgeLookup(
        lifted_inverse={
            tag: (level, edge) for (level, edge), tag in context.sweep.lifted.items()
        },
        vertical_inverse={tag: key for key, tag in context.sweep.vertical.items()},
        curve_descriptors=context.arrangement.curve_descriptors,
        native_edges=context.arrangement.native_edges,
        z_values=context.ownership.z_values,
    )
    points = {}
    curve_ids = {}
    curve_records = []
    point_curves = defaultdict(set)
    for edge in sorted(used_edges):
        record, edge_points = _detached_edge(gmsh, edge, lookup)
        curve_ids[edge] = record.curve_id
        points.update(edge_points)
        for pid in edge_points:
            point_curves[pid].add(record.curve_id)
        curve_records.append(record)
    return CurveTable(
        used_edges, curve_ids, curve_records, points, point_curves, lookup
    )


def build_surface_table(context, book, curves):
    """Build loop, surface and interface records for every registered surface."""
    gmsh = context.session.gmsh
    entities = context.session.entities
    route = context.session.route
    table = SurfaceTable(
        [],
        [],
        [],
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    for sid, tag in book.native_surfaces.items():
        data = book.surface_data[sid]
        loop_ids = _add_surface_loops(context, table, curves, (sid, tag, data))
        geometry = {
            "native_curved_face": True,
            "native_surface_type": gmsh.model.getType(2, tag),
            "arrangement_surface_id": sid,
            "representation": data["metadata"]["representation"],
            "shell_part": data["face_kind"],
            "source_polygon_ids": tuple(
                pid
                for owner in data["metadata"]["owner_semantic_ids"]
                if owner in entities
                for pid in entities[owner].polygon_ids
            ),
        }
        table.surfaces.append(
            SurfacePlanRecord(
                sid,
                data["owner"],
                data["role"],
                geometry,
                loop_ids[0],
                tuple(loop_ids[1:]),
                interface_id=data["interface"],
                normal_hint=data["normal"],
                valid_routes=(route,),
                metadata=data["metadata"],
            )
        )
        if data["interface"]:
            owners_tuple = data["metadata"]["owner_semantic_ids"]
            pair = (owners_tuple[0], owners_tuple[-1])
            table.interfaces.append(
                InterfacePlanRecord(
                    data["interface"],
                    data["metadata"]["interface_kinds"][0],
                    pair,
                    "native_curved_planar_arrangement_z_sweep",
                    metadata={"curved_arrangement_surface_id": sid},
                )
            )
    return table


def _add_surface_loops(context, table, curves, surface):
    """Append the loops of one surface and index which curves it uses."""
    sid, tag, data = surface
    gmsh = context.session.gmsh
    loop_ids = []
    native_loops = face_loops(gmsh, context.session.occ, tag)
    for index, tags in enumerate(native_loops):
        lid = identity("LOOP__", [sid, index])
        loop_ids.append(lid)
        refs = tuple(
            CurveRefRecord(
                curves.curve_ids[abs(int(t))], 1 if t > 0 else -1, "boundary"
            )
            for t in tags
        )
        table.loops.append(
            SurfaceLoopRecord(lid, refs, "outer" if index == 0 else "hole", sid)
        )
        for ref in refs:
            table.curve_surfaces[ref.curve_id].add(sid)
            table.curve_owners[ref.curve_id].update(
                data["metadata"]["owner_semantic_ids"]
            )
            table.curve_volumes[ref.curve_id].update(
                data["metadata"]["boundary_volume_ids"]
            )
            if data["interface"]:
                table.curve_interfaces[ref.curve_id].add(data["interface"])
    return loop_ids


def build_mm_contacts(context, book, curves):
    """Hidden metal-metal contact records, with shared conductor components."""
    session = context.session
    entities = session.entities
    contact_root, contact_members = _contact_components(book.native_contacts)
    mm_contacts = []
    for row in book.native_contacts:
        tag = row["tag"]
        a, b = row["pair"]
        members = tuple(sorted(contact_members[contact_root(a)]))
        component_id = identity("COMP__", members)
        equipotentials = {
            entities[eid].metadata.get("equipotential_id") for eid in members
        }
        equipotentials.discard(None)
        if len(equipotentials) > 1:
            raise ValueError("native curved contact has conflicting equipotential IDs")
        cid = identity("MM__", [row["pair"], row["key"]])
        geometry, contact_loops, contact_curves, contact_points = _contact_geometry(
            context, curves.lookup, cid, row
        )
        polygon = None
        if row["normal"] is not None and not geometry["native_curved_face"]:
            polygon = tuple(
                contact_points[
                    contact_curves[ref.curve_id].start_point_id
                    if ref.orientation > 0
                    else contact_curves[ref.curve_id].end_point_id
                ][:2]
                for ref in contact_loops[0].curve_refs
            )
        mm_contacts.append(
            MMContactRecord(
                cid,
                a,
                b,
                a
                + (
                    "__top"
                    if row["operation"] == "native_planar_contact"
                    else "__" + row["operation"]
                ),
                b
                + (
                    "__bottom"
                    if row["operation"] == "native_planar_contact"
                    else "__" + row["operation"]
                ),
                entities[a].polygon_ids,
                entities[b].polygon_ids,
                polygon,
                session.occ.getMass(2, tag),
                row["normal"],
                component_id,
                entities[a].net_id,
                next(iter(equipotentials), None),
                layer_provenance={
                    eid: entities[eid].metadata.get("source_layer_name")
                    for eid in row["pair"]
                },
                material_provenance={
                    eid: entities[eid].material_id for eid in row["pair"]
                },
                source_provenance={
                    "route": session.route,
                    "recognition_rule": row["operation"],
                    "hidden_solver_contact": True,
                    "source_entity_ids": row["pair"],
                },
                geometry_ref=geometry,
            )
        )
    return tuple(mm_contacts)


def _contact_components(native_contacts):
    """Union-find over contact pairs; entities that touch share one component."""
    contact_parents = {eid: eid for row in native_contacts for eid in row["pair"]}

    def contact_root(eid):
        while contact_parents[eid] != eid:
            eid = contact_parents[eid]
        return eid

    for row in native_contacts:
        a, b = row["pair"]
        contact_parents[contact_root(b)] = contact_root(a)
    contact_members = defaultdict(set)
    for eid in contact_parents:
        contact_members[contact_root(eid)].add(eid)
    return contact_root, contact_members


def _contact_geometry(context, lookup, cid, row):
    """Detach the boundary of one contact face into curves, points and loops."""
    gmsh = context.session.gmsh
    tag = row["tag"]
    contact_curves = {}
    contact_points = {}
    contact_loops = []
    for index, edges in enumerate(face_loops(gmsh, context.session.occ, tag)):
        refs = []
        for edge in edges:
            record, edge_points = _detached_edge(gmsh, abs(edge), lookup)
            contact_curves[record.curve_id] = record
            contact_points.update(edge_points)
            refs.append(
                CurveRefRecord(record.curve_id, 1 if edge > 0 else -1, "boundary")
            )
        contact_loops.append(
            SurfaceLoopRecord(
                identity("LOOP__", [cid, index]),
                tuple(refs),
                "outer" if index == 0 else "hole",
            )
        )
    geometry = {
        "native_curved_face": any(
            c.curve_kind != "line_segment" for c in contact_curves.values()
        ),
        "native_surface_type": gmsh.model.getType(2, tag),
        "surface_method": (
            "planar_arrangement_cap"
            if row["normal"] is not None
            else "native_collective_exact_edge_sweep"
        ),
        "sweep_direction": None if row["normal"] is not None else (0.0, 0.0, 1.0),
        "outer_loop_ref": contact_loops[0].loop_id,
        "hole_loop_refs": tuple(loop.loop_id for loop in contact_loops[1:]),
        "points": [
            asdict(PointPlanRecord(pid, xyz)) for pid, xyz in contact_points.items()
        ],
        "curves": [asdict(c) for c in contact_curves.values()],
        "surface_loops": [asdict(loop) for loop in contact_loops],
    }
    return geometry, contact_loops, contact_curves, contact_points


def bind_port_conductor_components(surfaces, mm_contacts):
    """Replace each lumped-port host's provisional conductor component id.

    Ports are registered before contacts exist, so their component ids start as
    `COMP__<host>`; the contact graph decides the real shared component.
    """

    component_by_entity = {
        eid: component["conductor_component_id"]
        for component in component_metadata(mm_contacts)
        for eid in component["members"]
    }
    for index, surface_record in enumerate(surfaces):
        if surface_record.surface_role != "lumped_port":
            continue
        attribute = surface_record.metadata["physical_attribute"]
        surfaces[index] = replace(
            surface_record,
            metadata={
                **surface_record.metadata,
                "physical_attribute": {
                    **attribute,
                    "owner_provenance": tuple(
                        {
                            **owner,
                            "conductor_component_id": component_by_entity.get(
                                owner["semantic_id"],
                                owner["conductor_component_id"],
                            ),
                        }
                        for owner in attribute["owner_provenance"]
                    ),
                },
            },
        )


def apply_surface_contract(session, table, mm_contacts):
    """Add surface-contract metadata to every non-port surface."""

    ordinary = {
        surface.surface_id: surface
        for surface in with_surface_contract_metadata(
            session.prepared,
            route=session.route,
            interfaces=tuple(table.interfaces),
            mm_contacts=mm_contacts,
            surfaces=tuple(
                surface
                for surface in table.surfaces
                if surface.surface_role != "lumped_port"
            ),
        )
    }
    table.surfaces = [
        ordinary.get(surface.surface_id, surface) for surface in table.surfaces
    ]


def bind_port_terminals(session, table):
    """Exact terminal identities come from shared arrangement curves, not from polygon chord tests."""
    route = session.route
    surfaces = table.surfaces
    for port_surface in tuple(
        surface for surface in surfaces if surface.surface_role == "lumped_port"
    ):
        port_curves = {
            ref.curve_id
            for loop in table.loops
            if loop.surface_id == port_surface.surface_id
            for ref in loop.curve_refs
        }
        overlaps = []
        for overlap in port_surface.metadata["source_provenance"]["overlaps"]:
            owner = overlap["host_semantic_id"]
            terminal_ids = tuple(
                sorted(
                    cid
                    for cid in port_curves
                    if any(
                        surface.owner_semantic_id == owner
                        and surface.surface_role != "lumped_port"
                        and (
                            route == "A"
                            or surface.geometry_ref.get("shell_part") == "sidewall"
                        )
                        and surface.surface_id in table.curve_surfaces[cid]
                        for surface in surfaces
                    )
                )
            )
            overlaps.append({**overlap, "native_terminal_curve_ids": terminal_ids})
            if route == "B":
                _mark_port_sidewalls(table, port_surface, overlap, terminal_ids)
        index = next(
            i
            for i, surface in enumerate(surfaces)
            if surface.surface_id == port_surface.surface_id
        )
        surfaces[index] = replace(
            port_surface,
            metadata={
                **port_surface.metadata,
                "source_provenance": {
                    **port_surface.metadata["source_provenance"],
                    "overlaps": overlaps,
                },
            },
        )


def _mark_port_sidewalls(table, port_surface, overlap, terminal_ids):
    """Route B: record which host sidewalls the port sheet partitions."""
    surfaces = table.surfaces
    owner = overlap["host_semantic_id"]
    for index, owner_surface in enumerate(surfaces):
        if (
            owner_surface.owner_semantic_id != owner
            or owner_surface.geometry_ref.get("shell_part") != "sidewall"
        ):
            continue
        if any(
            owner_surface.surface_id in table.curve_surfaces[cid]
            for cid in terminal_ids
        ):
            binding = {
                "port_surface_id": port_surface.surface_id,
                "overlap_id": overlap["overlap_id"],
                "host_semantic_id": owner,
            }
            surfaces[index] = replace(
                owner_surface,
                metadata={
                    **owner_surface.metadata,
                    "route_b_port_sheet_sidewall_partition": True,
                    "route_b_port_sheet_bindings": (
                        *owner_surface.metadata.get("route_b_port_sheet_bindings", ()),
                        binding,
                    ),
                },
            )


def finalize_curves_and_points(curves, table):
    """Attach usage indexes to curve and point records."""
    curve_records = tuple(
        replace(
            curve,
            used_by_surface_ids=tuple(sorted(table.curve_surfaces[curve.curve_id])),
            owner_semantic_ids=tuple(sorted(table.curve_owners[curve.curve_id])),
            interface_ids=tuple(sorted(table.curve_interfaces[curve.curve_id])),
            boundary_volume_ids=tuple(sorted(table.curve_volumes[curve.curve_id])),
        )
        for curve in curves.curve_records
    )
    point_records = tuple(
        PointPlanRecord(
            pid, xyz, used_by_curve_ids=tuple(sorted(curves.point_curves[pid]))
        )
        for pid, xyz in curves.points.items()
    )
    return curve_records, point_records
