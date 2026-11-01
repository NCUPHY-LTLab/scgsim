"""Register native caps, swept walls and metal contacts as plan surfaces.

Owns the surface stage of curved construction: which cap or wall becomes a plan
surface, which volumes it bounds and with which orientation, its interface kind
(`MS`, `MA`, `SA`, `SS`, `AA`), lumped-port metadata, and the hidden metal-metal
contacts found on shared cells and walls. Fills a `SurfaceBook`; it does not
build curve, loop or volume records.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from scgsim.geometry.construction.curved_context import SurfaceBook
from scgsim.geometry.construction.curved_kernel import _identity
from scgsim.geometry.models.topology import SurfaceRefRecord


@dataclass(frozen=True)
class _ContactRequest:
    tag: int
    pair: tuple
    key: tuple
    # Cap normal, or None for a lateral (wall) contact.
    normal: tuple | None
    operation: str


@dataclass(frozen=True)
class _SurfaceRequest:
    # Content-addressed key naming the surface (cell+Z, or wall edge+Z interval).
    key: dict
    tag: int
    # (slab, cell) keys of the owned cells touching this surface.
    adjacent: list
    metal: tuple = ()
    port: Any = None
    normal: tuple | None = None
    face_kind: str = "interface"


def register_surfaces(context):
    """Register contacts, horizontal caps, then swept walls, in that order."""
    book = SurfaceBook()
    _register_contacts(context, book)
    _register_horizontal_surfaces(context, book)
    _register_sidewall_surfaces(context, book)
    context.session.occ.synchronize()
    return book


def _register_contacts(context, book):
    """Finite shared cells own cap contacts; native walls own lateral contacts.

    Neither a Net label nor a polygon witness creates a geometric contact.
    """
    _register_cell_contacts(context, book)
    _register_wall_contacts(context, book)


def _register_cell_contacts(context, book):
    arrangement = context.arrangement
    ownership = context.ownership
    entities = context.session.entities
    ranges = ownership.ranges
    sheet_ids = ownership.sheet_ids
    z_values = ownership.z_values
    for cell in arrangement.cells:
        metals = sorted(
            eid
            for eid in arrangement.membership[cell]
            if eid in entities and entities[eid].material_kind == "conductor"
        )
        for a, b in combinations(metals, 2):
            lo = max(ranges[a][0], ranges[b][0])
            hi = min(ranges[a][1], ranges[b][1])
            if lo > hi:
                continue
            if (
                a in sheet_ids
                and ranges[a][0] != lo
                or b in sheet_ids
                and ranges[b][0] != lo
            ):
                continue
            level = z_values.index(lo)
            _add_contact(
                context,
                book,
                _ContactRequest(
                    _cap(context, cell, level),
                    (a, b),
                    ("cell", arrangement.cell_ids[cell], lo),
                    (0.0, 0.0, 1.0),
                    "native_planar_contact"
                    if lo == hi
                    else "native_overlap_normalization",
                ),
            )


def _register_wall_contacts(context, book):
    arrangement = context.arrangement
    ownership = context.ownership
    for slab in range(len(ownership.z_values) - 1):
        for edge, neighbors in arrangement.edge_cells.items():
            if len(neighbors) != 2:
                continue
            left, right = tuple(neighbors)
            pairs = {
                (a, b)
                for a in ownership.occupied.get((slab, left), ())
                for b in ownership.occupied.get((slab, right), ())
                if a != b
            }
            if pairs:
                tag = _wall(context, edge, slab)
                for pair in pairs:
                    _add_contact(
                        context,
                        book,
                        _ContactRequest(
                            tag,
                            pair,
                            (
                                "edge",
                                _identity("", arrangement.curve_descriptors[edge]),
                                slab,
                            ),
                            None,
                            "native_shared_wall_contact",
                        ),
                    )


def _add_contact(context, book, request):
    entities = context.session.entities
    ranges = context.ownership.ranges
    pair = tuple(sorted(request.pair))
    if request.normal is not None and ranges[pair[0]][0] > ranges[pair[1]][0]:
        pair = pair[::-1]
    if pair[0] == pair[1] or (pair, request.key) in book.contact_keys:
        return
    if len({entities[eid].net_id for eid in pair}) > 1:
        raise ValueError("native curved contact joins conductors on different Nets")
    book.contact_keys.add((pair, request.key))
    book.native_contacts.append(
        {
            "tag": request.tag,
            "pair": pair,
            "key": request.key,
            "normal": request.normal,
            "operation": request.operation,
        }
    )


def _cap(context, cell, level):
    """Planar cap of an arrangement cell at a Z level, from the lifted edges."""
    occ = context.session.occ
    lifted = context.sweep.lifted
    wires = []
    for index, loop in enumerate(context.arrangement.cell_loops[cell]):
        basis = loop if index == 0 else tuple(-edge for edge in reversed(loop))
        wires.append(
            occ.addCurveLoop(
                [int(math.copysign(lifted[level, abs(edge)], edge)) for edge in basis]
            )
        )
    return occ.addPlaneSurface(wires)


def _wall(context, edge, slab):
    return context.sweep.swept_walls[slab, edge]


def _register_horizontal_surfaces(context, book):
    """Caps between Z levels. Children are arrangement cells, so port and metal
    footprints have already been partitioned."""
    session = context.session
    arrangement = context.arrangement
    ownership = context.ownership
    owners = ownership.owners
    ranges = ownership.ranges
    route = session.route
    ports = {p.port_sheet_id: p for p in session.prepared.port_sheet_regions}
    for level, z in enumerate(ownership.z_values):
        for cell in arrangement.cells:
            below = (level - 1, cell)
            above = (level, cell)
            adjacent = [key for key in (below, above) if key in owners]
            metals = set(ownership.occupied.get(below, ())) | set(
                ownership.occupied.get(above, ())
            )
            metals.update(
                eid
                for eid in arrangement.membership[cell]
                if eid in ownership.sheet_ids and ranges[eid][0] == z
            )
            active_ports = [
                ports[eid[6:]]
                for eid in arrangement.membership[cell]
                if eid.startswith("PORT::") and ownership.port_planes[eid[6:]] == z
            ]
            if not adjacent or (
                len(adjacent) == 2
                and ownership.volume_for[below] == ownership.volume_for[above]
                and not metals
                and not active_ports
            ):
                continue
            tag = _cap(context, cell, level)
            face_kind = (
                "interface"
                if route == "A" and any(eid in ownership.sheet_ids for eid in metals)
                else "bottom"
                if metals and z == ranges[sorted(metals)[0]][0]
                else "top"
                if metals
                else "interface"
            )
            sid = _register_surface(
                context,
                book,
                _SurfaceRequest(
                    {"cell": arrangement.cell_ids[cell], "z_um": z},
                    tag,
                    adjacent,
                    tuple(sorted(metals)),
                    active_ports[0] if active_ports and not metals else None,
                    (0.0, 0.0, 1.0),
                    face_kind,
                ),
            )
            # A Route B port sheet is embedded in its vacuum volume, so it is not
            # part of that volume's boundary.
            if not (route == "B" and active_ports and not metals):
                for key in adjacent:
                    book.volume_refs[ownership.volume_for[key]].append(
                        SurfaceRefRecord(
                            sid,
                            "forward" if key == below else "reversed",
                            "boundary",
                        )
                    )


def _register_sidewall_surfaces(context, book):
    arrangement = context.arrangement
    ownership = context.ownership
    z_values = ownership.z_values
    for slab in range(len(z_values) - 1):
        for edge, neighbors in arrangement.edge_cells.items():
            adjacent = [(slab, c) for c in neighbors if (slab, c) in ownership.owners]
            if not adjacent or (
                len(adjacent) == 2
                and ownership.volume_for[adjacent[0]]
                == ownership.volume_for[adjacent[1]]
            ):
                continue
            metals = tuple(
                sorted(
                    {
                        eid
                        for c in neighbors
                        for eid in ownership.occupied.get((slab, c), ())
                    }
                )
            )
            tag = _wall(context, edge, slab)
            sid = _register_surface(
                context,
                book,
                _SurfaceRequest(
                    {
                        "edge": arrangement.curve_descriptors[edge],
                        "z_interval": [z_values[slab], z_values[slab + 1]],
                    },
                    tag,
                    adjacent,
                    metals,
                    face_kind="sidewall",
                ),
            )
            for key in adjacent:
                cell = key[1]
                sign = next(
                    e
                    for loop in arrangement.cell_loops[cell]
                    for e in loop
                    if abs(e) == edge
                )
                book.volume_refs[ownership.volume_for[key]].append(
                    SurfaceRefRecord(
                        sid, "forward" if sign > 0 else "reversed", "boundary"
                    )
                )


def _register_surface(context, book, request):
    """Name one cap or wall and record its owner, interface kind and metadata."""
    entities = context.session.entities
    route = context.session.route
    owners = context.ownership.owners
    port = request.port
    sid = _identity("SURF__", request.key)
    book.native_surfaces[sid] = request.tag
    boundary_keys = [k for k in request.adjacent if k in owners]
    boundary_owners = tuple(owners[k] for k in boundary_keys)
    semantic_owners = tuple(dict.fromkeys((*request.metal, *boundary_owners)))
    kinds = _interface_kinds(entities, owners, request.metal, boundary_keys)
    prefix = "_".join(kinds)
    iid = (
        (kinds[0] + "__" + "__".join(semantic_owners) + "__" + sid)
        if semantic_owners
        else None
    )
    metadata = {
        "owner_semantic_ids": semantic_owners,
        "boundary_volume_ids": boundary_owners,
        "interface_kinds": tuple(kinds),
        "curved_arrangement_surface": True,
        "physical_name": prefix
        + "__"
        + "__".join(semantic_owners)
        + "__"
        + request.face_kind.upper(),
        "exposed_surface_role": request.face_kind,
        "representation": (
            "surface_sheet"
            if request.metal and route == "A"
            else "cutout_boundary_shell"
            if request.metal
            else "solution_surface"
        ),
    }
    if port:
        semantic_owners = _apply_port_metadata(
            context, sid, request, metadata, boundary_keys
        )
        iid = None
    book.surface_data[sid] = {
        "owner": semantic_owners[0] if semantic_owners else port.port_sheet_id,
        "interface": iid,
        "metadata": metadata,
        "normal": request.normal,
        "role": "lumped_port"
        if port
        else "cutout_boundary_shell"
        if request.metal and route == "B"
        else "interface"
        if len(semantic_owners) > 1
        else "domain_boundary",
        "face_kind": request.face_kind,
    }
    return sid


def _interface_kinds(entities, owners, metal, boundary_keys):
    """Interface kinds from the materials on each side.

    MS is metal-substrate, MA metal-air, SA substrate-air, SS substrate-substrate
    and AA air-air. Here "substrate" means a dielectric solution region and "air"
    a vacuum one.
    """
    if metal:
        return list(
            dict.fromkeys(
                "MS" if entities[owners[k]].material_kind == "dielectric" else "MA"
                for k in boundary_keys
            )
        )
    if len(boundary_keys) == 2:
        materials = [entities[owners[k]].material_kind for k in boundary_keys]
        return [
            "SS"
            if materials == ["dielectric", "dielectric"]
            else "AA"
            if materials == ["vacuum", "vacuum"]
            else "SA"
        ]
    return [
        "SA"
        if boundary_keys
        and entities[owners[boundary_keys[0]]].material_kind == "dielectric"
        else "AA"
    ]


def _apply_port_metadata(context, sid, request, metadata, boundary_keys):
    """Turn a surface's metadata into a lumped-port sheet; return its owners."""
    entities = context.session.entities
    route = context.session.route
    owners = context.ownership.owners
    port = request.port
    boundary_owners = tuple(owners[k] for k in boundary_keys)
    hosts = tuple(dict.fromkeys(overlap.host_semantic_id for overlap in port.overlaps))
    domain_owners = tuple(dict.fromkeys(boundary_owners))
    if route == "B":
        domain_owners = domain_owners[:1]
    embedded = next(
        owner for owner in domain_owners if entities[owner].material_kind == "vacuum"
    )
    source = {
        **dict(port.metadata),
        "source_layer": port.source_layer,
        "source_polygon_id": port.source_polygon_id,
        "overlaps": [asdict(overlap) for overlap in port.overlaps],
        "host_polygon_ids": tuple(overlap.host_polygon_id for overlap in port.overlaps),
        "route": route,
    }
    if route == "B":
        metadata["embedded_volume_plan_ids"] = tuple(
            dict.fromkeys(
                context.ownership.volume_for[key]
                for key in boundary_keys
                if entities[owners[key]].material_kind == "vacuum"
            )
        )
        if not metadata["embedded_volume_plan_ids"]:
            raise ValueError(f"{sid} active port has no final vacuum volume host")
    metadata.update(
        {
            "physical_name": "LUMPED_PORT__" + port.metadata["source_name"],
            "route": route,
            "representation": "lumped_port_sheet",
            "interface_type": "lumped_port",
            "face_kind": "sheet",
            "net_id": None,
            "conductor_component_id": None,
            "equipotential_id": None,
            "owner_semantic_ids": hosts,
            "boundary_volume_ids": domain_owners,
            "source_provenance": source,
            "route_a_boundary_port": route == "A",
            "embedded_surface": True,
            "embedded_volume_id": embedded,
            "physical_attribute": {
                "port_index": port.metadata["port_index"],
                "port_name": port.metadata["source_name"],
                "source_layer": port.source_layer,
                "target_layer": port.metadata["target_layer"],
                "direction": port.metadata["direction"],
                "embedded_volume_id": embedded,
                "owner_semantic_ids": hosts,
                "owner_provenance": tuple(
                    {
                        "semantic_id": host,
                        "net_id": entities[host].net_id,
                        "equipotential_id": entities[host].metadata.get(
                            "equipotential_id"
                        ),
                        "conductor_component_id": "COMP__" + host,
                    }
                    for host in hosts
                ),
            },
        }
    )
    return hosts
