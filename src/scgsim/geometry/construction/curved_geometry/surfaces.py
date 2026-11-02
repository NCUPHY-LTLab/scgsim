"""Sweep shared native edges and name contact, interface and port surfaces."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import cast

from scgsim.geometry.construction.curved_geometry.arrangement import PlanarArrangement
from scgsim.geometry.construction.curved_geometry.kernel import (
    edge_vertices,
    geometry_identity,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.topology import (
    SurfaceRefRecord,
)


@dataclass(frozen=True)
class _SurfacePlacement:
    """One retained native face and its adjacent semantic cells."""

    key: object
    tag: int
    adjacent: list[tuple[int, int]]
    metal: tuple[str, ...] = ()
    port: PortSheetRegionRecord | None = None
    normal: tuple[float, float, float] | None = None
    face_kind: str = "interface"


class SharedSurfaces:
    """Retain shared sweep registries while naming contacts and solver surfaces."""

    def __init__(self, planar: PlanarArrangement):
        self.planar = planar
        self.gmsh = planar.kernel.gmsh
        self.occ = planar.kernel.occ

    def build(self):
        """Sweep the shared curves before creating contacts and material faces."""
        self.sweep_edges()
        self.create_contacts()
        self.horizontal_surfaces()
        self.vertical_surfaces()
        self.occ.synchronize()
        return self

    def sweep_edges(self):
        """Sweep the entire shared complex strictly from edges to walls."""
        # Sweep the entire shared curve complex together, strictly 1D -> 2D.
        # A native prism preserves trimmed rational/spline geometry and shares
        # the vertical edge of each source vertex across adjacent walls.
        self.lifted = {}
        self.lifted_vertices = {}
        self.vertical = {}
        self.swept_walls = {}
        for edge in self.planar.retained_edges:
            tag = self.occ.copy([(1, edge)])[0][1]
            self.occ.translate([(1, tag)], 0, 0, self.planar.z_values[0])
            self.lifted[0, edge] = tag
        _, copied_history = self.occ.fragment(
            [(1, self.lifted[0, edge]) for edge in self.planar.retained_edges], []
        )
        for edge, history in zip(
            self.planar.retained_edges, copied_history, strict=False
        ):
            tags = [tag for dim, tag in history if dim == 1]
            if len(tags) != 1:
                raise ValueError("Z-copy changed an already arranged planar edge")
            self.lifted[0, edge] = tags[0]
        self.occ.synchronize()
        for edge in self.planar.retained_edges:
            for source, copied in zip(
                edge_vertices(self.gmsh, edge),
                edge_vertices(self.gmsh, self.lifted[0, edge]),
                strict=False,
            ):
                self.lifted_vertices[0, source] = copied
        for slab, (lo, hi) in enumerate(
            zip(self.planar.z_values, self.planar.z_values[1:], strict=False)
        ):
            self._sweep_slab(slab, lo, hi)

    def _sweep_slab(self, slab, lo, hi):
        """Sweep one shared slab and retain its upper and vertical edge identities."""
        source_by_tag = {
            self.lifted[slab, edge]: edge for edge in self.planar.retained_edges
        }
        source_by_vertex = {
            tag: vertex
            for (level, vertex), tag in self.lifted_vertices.items()
            if level == slab
        }
        output = self.occ.extrude([(1, tag) for tag in source_by_tag], 0, 0, hi - lo)
        self.occ.synchronize()
        for dim, face in output:
            if dim != 2:
                continue
            boundary = [
                abs(tag)
                for d, tag in self.gmsh.model.getBoundary([(2, face)], oriented=True)
                if d == 1
            ]
            bottom_tags = [tag for tag in boundary if tag in source_by_tag]
            if len(bottom_tags) != 1:
                raise ValueError(
                    "native edge sweep did not retain one canonical source edge per wall"
                )
            edge = source_by_tag[bottom_tags[0]]
            self.swept_walls[slab, edge] = face
            for tag in boundary:
                if tag == bottom_tags[0]:
                    continue
                vertices = edge_vertices(self.gmsh, tag)
                lower_vertices = [v for v in vertices if v in source_by_vertex]
                if not lower_vertices:
                    self.lifted[slab + 1, edge] = tag
                elif len(lower_vertices) == 1:
                    lower = lower_vertices[0]
                    vertex = source_by_vertex[lower]
                    key = (slab, vertex)
                    if key in self.vertical and self.vertical[key] != tag:
                        raise ValueError(
                            "native collective sweep duplicated a shared vertical edge"
                        )
                    self.vertical[key] = tag
                    self.lifted_vertices[slab + 1, vertex] = next(
                        v for v in vertices if v != lower
                    )
                else:
                    raise ValueError(
                        "native edge sweep returned an unexpected horizontal boundary"
                    )
            if (slab + 1, edge) not in self.lifted:
                raise ValueError("native edge sweep lost the canonical upper edge")
        if any(
            (slab, edge) not in self.swept_walls for edge in self.planar.retained_edges
        ):
            raise ValueError(
                "native edge sweep did not cover the arranged curve complex"
            )

    def contact(self, tag, pair, key, normal, operation):
        """Normalize an encountered native contact after checking its Nets."""
        pair = tuple(sorted(pair))
        if (
            normal is not None
            and self.planar.ranges[pair[0]][0] > self.planar.ranges[pair[1]][0]
        ):
            pair = pair[::-1]
        if pair[0] == pair[1] or (pair, key) in self.contact_keys:
            return
        if len({self.planar.entities[eid].net_id for eid in pair}) > 1:
            raise ValueError("native curved contact joins conductors on different Nets")
        self.contact_keys.add((pair, key))
        self.native_contacts.append(
            {
                "tag": tag,
                "pair": pair,
                "key": key,
                "normal": normal,
                "operation": operation,
            }
        )

    def cap(self, cell, level):
        """Build one cap in its arranged outer-wire basis."""
        wires = []
        for index, loop in enumerate(self.planar.cell_loops[cell]):
            basis = loop if index == 0 else tuple(-edge for edge in reversed(loop))
            wires.append(
                self.occ.addCurveLoop(
                    [
                        int(math.copysign(self.lifted[level, abs(edge)], edge))
                        for edge in basis
                    ]
                )
            )
        return self.occ.addPlaneSurface(wires)

    def create_contacts(self):
        """Recognize contacts only from shared native cells and walls."""
        self.native_surfaces = {}
        self.surface_data = {}
        self.volume_refs = defaultdict(list)
        self.native_contacts = []
        self.contact_keys = set()

        # Finite shared arrangement cells own contact normalization. Native
        # walls own lateral contacts; neither a Net label nor a polygon witness
        # creates a geometric contact.
        for cell in self.planar.cells:
            metals = sorted(
                eid
                for eid in self.planar.membership[cell]
                if eid in self.planar.entities
                and self.planar.entities[eid].material_kind == "conductor"
            )
            for a, b in combinations(metals, 2):
                lo = max(self.planar.ranges[a][0], self.planar.ranges[b][0])
                hi = min(self.planar.ranges[a][1], self.planar.ranges[b][1])
                if lo > hi:
                    continue
                if (
                    a in self.planar.sheet_ids
                    and self.planar.ranges[a][0] != lo
                    or b in self.planar.sheet_ids
                    and self.planar.ranges[b][0] != lo
                ):
                    continue
                level = self.planar.z_values.index(lo)
                self.contact(
                    self.cap(cell, level),
                    (a, b),
                    ("cell", self.planar.cell_ids[cell], lo),
                    (0.0, 0.0, 1.0),
                    "native_planar_contact"
                    if lo == hi
                    else "native_overlap_normalization",
                )
        for slab in range(len(self.planar.z_values) - 1):
            for edge, neighbors in self.planar.edge_cells.items():
                if len(neighbors) != 2:
                    continue
                left, right = tuple(neighbors)
                pairs = {
                    (a, b)
                    for a in self.planar.occupied.get((slab, left), ())
                    for b in self.planar.occupied.get((slab, right), ())
                    if a != b
                }
                if pairs:
                    tag = self.swept_walls[slab, edge]
                    for pair in pairs:
                        self.contact(
                            tag,
                            pair,
                            (
                                "edge",
                                geometry_identity(
                                    "", self.planar.curve_descriptors[edge]
                                ),
                                slab,
                            ),
                            None,
                            "native_shared_wall_contact",
                        )

    def surface(self, placement: _SurfacePlacement):
        """Name a retained solver surface and its semantic ownership."""
        key = placement.key
        tag = placement.tag
        adjacent = placement.adjacent
        metal = placement.metal
        port = placement.port
        normal = placement.normal
        face_kind = placement.face_kind
        sid = geometry_identity("SURF__", key)
        self.native_surfaces[sid] = tag
        boundary_keys = [k for k in adjacent if k in self.planar.owners]
        boundary_owners = tuple(self.planar.owners[k] for k in boundary_keys)
        semantic_owners = tuple(dict.fromkeys((*metal, *boundary_owners)))
        kinds = []
        if metal:
            kinds = list(
                dict.fromkeys(
                    "MS"
                    if self.planar.entities[self.planar.owners[k]].material_kind
                    == "dielectric"
                    else "MA"
                    for k in boundary_keys
                )
            )
        elif len(boundary_keys) == 2:
            materials = [
                self.planar.entities[self.planar.owners[k]].material_kind
                for k in boundary_keys
            ]
            kinds = [
                "SS"
                if materials == ["dielectric", "dielectric"]
                else "AA"
                if materials == ["vacuum", "vacuum"]
                else "SA"
            ]
        else:
            kinds = [
                "SA"
                if boundary_keys
                and self.planar.entities[
                    self.planar.owners[boundary_keys[0]]
                ].material_kind
                == "dielectric"
                else "AA"
            ]
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
            + face_kind.upper(),
            "exposed_surface_role": face_kind,
            "representation": (
                "surface_sheet"
                if metal and self.planar.route == "A"
                else "cutout_boundary_shell"
                if metal
                else "solution_surface"
            ),
        }
        if port:
            semantic_owners = self._bind_port(sid, port, boundary_keys, metadata)
            iid = None
        self.surface_data[sid] = {
            "owner": semantic_owners[0]
            if semantic_owners
            else cast(PortSheetRegionRecord, port).port_sheet_id,
            "interface": iid,
            "metadata": metadata,
            "normal": normal,
            "role": "lumped_port"
            if port
            else "cutout_boundary_shell"
            if metal and self.planar.route == "B"
            else "interface"
            if len(semantic_owners) > 1
            else "domain_boundary",
            "face_kind": face_kind,
        }
        return sid

    def _bind_port(self, sid, port, boundary_keys, metadata):
        """Attach source port ownership to an already named arrangement surface."""
        boundary_owners = metadata["boundary_volume_ids"]
        hosts = tuple(
            dict.fromkeys(overlap.host_semantic_id for overlap in port.overlaps)
        )
        domain_owners = tuple(dict.fromkeys(boundary_owners))
        if self.planar.route == "B":
            domain_owners = domain_owners[:1]
        embedded = next(
            owner
            for owner in domain_owners
            if self.planar.entities[owner].material_kind == "vacuum"
        )
        source = {
            **dict(port.metadata),
            "source_layer": port.source_layer,
            "source_polygon_id": port.source_polygon_id,
            "overlaps": [asdict(overlap) for overlap in port.overlaps],
            "host_polygon_ids": tuple(
                overlap.host_polygon_id for overlap in port.overlaps
            ),
            "route": self.planar.route,
        }
        if self.planar.route == "B":
            metadata["embedded_volume_plan_ids"] = tuple(
                dict.fromkeys(
                    self.planar.volume_for[key]
                    for key in boundary_keys
                    if self.planar.entities[self.planar.owners[key]].material_kind
                    == "vacuum"
                )
            )
            if not metadata["embedded_volume_plan_ids"]:
                raise ValueError(f"{sid} active port has no final vacuum volume host")
        metadata.update(
            {
                "physical_name": "LUMPED_PORT__" + port.metadata["source_name"],
                "route": self.planar.route,
                "representation": "lumped_port_sheet",
                "interface_type": "lumped_port",
                "face_kind": "sheet",
                "net_id": None,
                "conductor_component_id": None,
                "equipotential_id": None,
                "owner_semantic_ids": hosts,
                "boundary_volume_ids": domain_owners,
                "source_provenance": source,
                "route_a_boundary_port": self.planar.route == "A",
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
                            "net_id": self.planar.entities[host].net_id,
                            "equipotential_id": self.planar.entities[host].metadata.get(
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

    def horizontal_surfaces(self):
        """Emit arranged horizontal children, including active port partitions."""
        # Horizontal children are arrangement cells, hence port and metal footprints have already been partitioned.
        ports = {p.port_sheet_id: p for p in self.planar.prepared.port_sheet_regions}
        for level, z in enumerate(self.planar.z_values):
            for cell in self.planar.cells:
                below = (level - 1, cell)
                above = (level, cell)
                adjacent = [key for key in (below, above) if key in self.planar.owners]
                metals = set(self.planar.occupied.get(below, ())) | set(
                    self.planar.occupied.get(above, ())
                )
                metals.update(
                    eid
                    for eid in self.planar.membership[cell]
                    if eid in self.planar.sheet_ids and self.planar.ranges[eid][0] == z
                )
                active_ports = [
                    ports[eid[6:]]
                    for eid in self.planar.membership[cell]
                    if eid.startswith("PORT::")
                    and self.planar.port_planes[eid[6:]] == z
                ]
                if not adjacent or (
                    len(adjacent) == 2
                    and self.planar.volume_for[below] == self.planar.volume_for[above]
                    and not metals
                    and not active_ports
                ):
                    continue
                tag = self.cap(cell, level)
                face_kind = (
                    "interface"
                    if self.planar.route == "A"
                    and any(eid in self.planar.sheet_ids for eid in metals)
                    else "bottom"
                    if metals and z == self.planar.ranges[sorted(metals)[0]][0]
                    else "top"
                    if metals
                    else "interface"
                )
                sid = self.surface(
                    _SurfacePlacement(
                        {"cell": self.planar.cell_ids[cell], "z_um": z},
                        tag,
                        adjacent,
                        tuple(sorted(metals)),
                        active_ports[0] if active_ports and not metals else None,
                        normal=(0.0, 0.0, 1.0),
                        face_kind=face_kind,
                    )
                )
                if not (self.planar.route == "B" and active_ports and not metals):
                    for key in adjacent:
                        self.volume_refs[self.planar.volume_for[key]].append(
                            SurfaceRefRecord(
                                sid,
                                "forward" if key == below else "reversed",
                                "boundary",
                            )
                        )

    def vertical_surfaces(self):
        """Emit exposed shared walls and oriented volume references."""
        for slab in range(len(self.planar.z_values) - 1):
            for edge, neighbors in self.planar.edge_cells.items():
                adjacent = [
                    (slab, c) for c in neighbors if (slab, c) in self.planar.owners
                ]
                if not adjacent or (
                    len(adjacent) == 2
                    and self.planar.volume_for[adjacent[0]]
                    == self.planar.volume_for[adjacent[1]]
                ):
                    continue
                metals = tuple(
                    sorted(
                        {
                            eid
                            for c in neighbors
                            for eid in self.planar.occupied.get((slab, c), ())
                        }
                    )
                )
                tag = self.swept_walls[slab, edge]
                sid = self.surface(
                    _SurfacePlacement(
                        {
                            "edge": self.planar.curve_descriptors[edge],
                            "z_interval": [
                                self.planar.z_values[slab],
                                self.planar.z_values[slab + 1],
                            ],
                        },
                        tag,
                        adjacent,
                        metals,
                        face_kind="sidewall",
                    )
                )
                for key in adjacent:
                    cell = key[1]
                    sign = next(
                        e
                        for loop in self.planar.cell_loops[cell]
                        for e in loop
                        if abs(e) == edge
                    )
                    self.volume_refs[self.planar.volume_for[key]].append(
                        SurfaceRefRecord(
                            sid, "forward" if sign > 0 else "reversed", "boundary"
                        )
                    )
