"""Arrange planar footprints and resolve material ownership across Z slabs."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy
    from numpy.typing import NDArray

from scgsim.geometry.construction.curved_geometry.boundaries import (
    BoundaryChains,
    SourceRings,
    source_z_range,
)
from scgsim.geometry.construction.curved_geometry.kernel import (
    CurveKernel,
    face_loops,
    geometry_identity,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SourceCurveSpec
from scgsim.geometry.planning.interfaces import validate_volumetric_overlap_ownership
from scgsim.geometry.primitives.constants import TOPOLOGY_EPS_UM
from scgsim.geometry.primitives.entity_validation import (
    resolve_contact_pad_attachment,
)


class PlanarArrangement:
    """Share one arranged curve complex across material and slab ownership steps."""

    def __init__(
        self,
        prepared: GeometryBuildInput,
        route: RouteLiteral,
        rings: SourceRings,
        chains: BoundaryChains,
    ):
        self.prepared = prepared
        self.route: RouteLiteral = route
        self.rings = rings
        self.chains = chains
        self.entities = {e.semantic_id: e for e in prepared.entities}

    def build(self, kernel: CurveKernel):
        """Arrange source footprints before resolving their shared solution cells."""
        self.kernel = kernel
        self.create_footprints()
        self.fragment_footprints()
        self.recover_edge_history()
        self.name_cells()
        self.validate_conductor_ownership()
        self.assign_slabs()
        self.connect_solution_cells()
        return self

    def create_footprints(self):
        """Create conductor, domain and port footprints in source order."""
        self.footprints = []
        self.source_edges = []
        for entity_id, records in self.rings.items():
            for pid, _outer, holes in records:
                sources = []
                for role, index in [
                    ("outer", None),
                    *[("hole", i) for i in range(len(holes))],
                ]:
                    key = (entity_id, pid, role, index)
                    sources.append(
                        (
                            self.chains[key],
                            {
                                "entity_id": entity_id,
                                "polygon_id": pid,
                                "role": role,
                                "hole_index": index,
                            },
                        )
                    )
                face, edges = self.kernel.face(sources)
                self.footprints.append((face, entity_id))
                self.source_edges.extend(edges)
        for port in self.prepared.port_sheet_regions:
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
            face, edges = self.kernel.face(sources)
            self.footprints.append((face, "PORT::" + port.port_sheet_id))
            self.source_edges.extend(edges)
        self.source_edges = list(dict.fromkeys(self.source_edges))
        self.kernel.occ.synchronize()

    def fragment_footprints(self):
        """Detach source history from the single shared planar Boolean."""
        self.source_footprints = [
            {
                "source_owner": owner,
                "area_um2": self.kernel.occ.getMass(2, face),
                "native_bounds_um": self.kernel.occ.getBoundingBox(2, face),
            }
            for face, owner in self.footprints
        ]
        self.references = {
            tag: self.kernel.occ.copy([(1, tag)])[0][1] for tag in self.source_edges
        }
        output, mapping = self.kernel.occ.fragment(
            [(2, face) for face, _ in self.footprints],
            [(1, tag) for tag in self.source_edges],
        )
        self.kernel.occ.synchronize()
        self.cells = {tag for dim, tag in output if dim == 2}
        self.membership = defaultdict(set)
        for (_, owner), mapped in zip(
            self.footprints, mapping[: len(self.footprints)], strict=False
        ):
            for dim, tag in mapped:
                if dim == 2:
                    self.membership[tag].add(owner)
        self.native_edges = defaultdict(list)
        for source, mapped in zip(
            self.source_edges, mapping[len(self.footprints) :], strict=False
        ):
            reference = self.references[source]
            for dim, tag in mapped:
                if dim != 1:
                    continue
                bounds = self.kernel.gmsh.model.getParametrizationBounds(1, tag)
                endpoint_values = [
                    # Gmsh returns NumPy coordinates with the required NumPy runtime.
                    cast(
                        "NDArray[numpy.float64]",
                        self.kernel.gmsh.model.getValue(1, tag, [float(u[0])]),
                    ).tolist()
                    for u in bounds
                ]
                source_interval = [
                    float(
                        self.kernel.gmsh.model.getParametrization(1, reference, xyz)[0]
                    )
                    for xyz in endpoint_values
                ]
                self.native_edges[tag].append(
                    {
                        **self.kernel.source_edges[source],
                        "source_parameter_interval": source_interval,
                        "native_parameter_bounds": [float(u[0]) for u in bounds],
                    }
                )
        self.cell_loops = {
            tag: face_loops(self.kernel.gmsh, self.kernel.occ, tag)
            for tag in self.cells
        }
        self.edge_cells = defaultdict(set)
        for cell, loops in self.cell_loops.items():
            for loop in loops:
                for edge in loop:
                    self.edge_cells[abs(edge)].add(cell)

    def recover_edge_history(self):
        """Recover omitted OCC history through native common geometry."""
        # OCC may omit lower-dimensional history for an unchanged exterior
        # wire while replacing its native tags during a face Boolean. Recover
        # that history through native common geometry, never endpoint matching
        # or a caller-chosen geometric tolerance.
        for edge in self.edge_cells:
            if self.native_edges[edge]:
                continue
            for source, reference in self.references.items():
                existing = set(self.kernel.occ.getEntities())
                common, _ = self.kernel.occ.intersect(
                    [(1, edge)], [(1, reference)], removeObject=False, removeTool=False
                )
                self.kernel.occ.synchronize()
                for dim, common_edge in common:
                    if dim != 1:
                        continue
                    bounds = self.kernel.gmsh.model.getParametrizationBounds(1, edge)
                    common_bounds = self.kernel.gmsh.model.getParametrizationBounds(
                        1, common_edge
                    )
                    values = [
                        self.kernel.gmsh.model.getValue(1, common_edge, [float(u[0])])
                        for u in common_bounds
                    ]
                    self.native_edges[edge].append(
                        {
                            **self.kernel.source_edges[source],
                            "source_parameter_interval": [
                                float(
                                    self.kernel.gmsh.model.getParametrization(
                                        1, reference, xyz
                                    )[0]
                                )
                                for xyz in values
                            ],
                            "arrangement_parameter_interval": [
                                float(
                                    self.kernel.gmsh.model.getParametrization(
                                        1, edge, xyz
                                    )[0]
                                )
                                for xyz in values
                            ],
                            "native_parameter_bounds": [float(u[0]) for u in bounds],
                            "native_history_method": "occ_common_geometry",
                        }
                    )
                self.kernel.occ.remove(
                    [entity for entity in common if entity not in existing],
                    recursive=False,
                )
            if not self.native_edges[edge]:
                raise ValueError(
                    "native arrangement boundary has no source curve provenance"
                )

    def name_cells(self):
        """Discard scratch shapes and name cells from retained source geometry."""
        # Remove only scratch source/reference entities after the Boolean history has been detached.
        self.retained_edges = set(self.edge_cells)
        self.kernel.occ.remove(
            [(2, t) for _, t in self.kernel.occ.getEntities(2) if t not in self.cells],
            recursive=False,
        )
        self.kernel.occ.remove(
            [
                (1, t)
                for _, t in self.kernel.occ.getEntities(1)
                if t not in self.retained_edges
            ],
            recursive=False,
        )
        self.kernel.occ.synchronize()
        self.curve_descriptors = {
            edge: {
                "native_type": self.kernel.gmsh.model.getType(1, edge),
                "sources": self.native_edges[edge],
                "native_bounds": [
                    float(u[0])
                    for u in self.kernel.gmsh.model.getParametrizationBounds(1, edge)
                ],
            }
            for edge in self.retained_edges
        }
        self.cell_ids = {
            cell: geometry_identity(
                "CELL__",
                {
                    "curves": sorted(
                        geometry_identity("", self.curve_descriptors[abs(e)])
                        for loop in loops
                        for e in loop
                    ),
                    "source_owners": sorted(self.membership[cell]),
                },
            )
            for cell, loops in self.cell_loops.items()
        }
        # Route A face metal is a sheet; all other finite conductors remain excluded bodies.
        self.sheet_ids = {
            eid
            for eid, e in self.entities.items()
            if self.route == "A"
            and e.material_kind == "conductor"
            and e.part_role == "face_metal"
        }
        self.ranges = {eid: source_z_range(e) for eid, e in self.entities.items()}

    def finite_overlap(self, first, second):
        """Use shared native cells as the conductor overlap witness."""
        return any(
            first.semantic_id in self.membership[cell]
            and second.semantic_id in self.membership[cell]
            for cell in self.cells
        )

    def validate_conductor_ownership(self):
        """Apply the existing attachment and volumetric overlap policy."""
        conductors = tuple(
            sorted(
                (e for e in self.entities.values() if e.material_kind == "conductor"),
                key=lambda e: e.semantic_id,
            )
        )
        for conductor in conductors:
            if conductor.part_role == "contact_pad":
                resolve_contact_pad_attachment(
                    conductor, conductors, finite_overlap=self.finite_overlap
                )
        for lower, upper in combinations(conductors, 2):
            lower_min, lower_max = self.ranges[lower.semantic_id]
            upper_min, upper_max = self.ranges[upper.semantic_id]
            if min(lower_max, upper_max) - max(lower_min, upper_min) <= TOPOLOGY_EPS_UM:
                continue
            if self.finite_overlap(lower, upper):
                validate_volumetric_overlap_ownership(
                    lower, upper, conductors, finite_overlap=self.finite_overlap
                )

    def assign_slabs(self):
        """Assign metals and solution owners on each exact source Z interval."""
        self.port_planes = {}
        for port in self.prepared.port_sheet_regions:
            host_ids = tuple(
                dict.fromkeys(overlap.host_semantic_id for overlap in port.overlaps)
            )
            intervals = [self.ranges[eid] for eid in host_ids]
            if len(host_ids) != 2 or len(set(intervals)) != 1:
                raise ValueError(
                    "curved lumped-port hosts must share one authored Z range"
                )
            lo, hi = intervals[0]
            self.port_planes[port.port_sheet_id] = (
                lo if self.route == "A" else (lo + hi) / 2
            )
        self.z_values = sorted(
            {
                z
                for eid, interval in self.ranges.items()
                for z in (interval[:1] if eid in self.sheet_ids else interval)
            }
            | set(self.port_planes.values())
        )
        self.occupied = {}
        self.owners = {}
        for slab, (lo, hi) in enumerate(
            zip(self.z_values, self.z_values[1:], strict=False)
        ):
            middle = (lo + hi) / 2
            for cell in self.cells:
                candidates = [
                    eid
                    for eid in self.membership[cell]
                    if eid in self.entities
                    and eid not in self.sheet_ids
                    and self.ranges[eid][0] < middle < self.ranges[eid][1]
                ]
                metals = [
                    eid
                    for eid in candidates
                    if self.entities[eid].material_kind == "conductor"
                ]
                if len({self.entities[eid].net_id for eid in metals}) > 1:
                    raise ValueError(
                        "native curved arrangement found overlapping conductors on different Nets"
                    )
                if metals:
                    self.occupied[slab, cell] = sorted(metals)
                    continue
                domains = [
                    eid
                    for eid in candidates
                    if self.entities[eid].role == "solution_region"
                ]
                if domains:
                    owner = max(
                        domains,
                        key=lambda eid: (
                            self.entities[eid].material_kind == "dielectric",
                            self.entities[eid].priority,
                            eid,
                        ),
                    )
                    self.owners[slab, cell] = owner

    def connect_solution_cells(self):
        """Connect same-owner cells without crossing a sheet barrier."""
        # Connected planar/slab cells form solution shells; no 3D Boolean repair.
        parents = {key: key for key in self.owners}

        def find(key):
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        def union(a, b):
            if (
                a in self.owners
                and b in self.owners
                and self.owners[a] == self.owners[b]
            ):
                parents[find(b)] = find(a)

        for slab, cell in self.owners:
            for edge in (abs(e) for loop in self.cell_loops[cell] for e in loop):
                for other in self.edge_cells[edge]:
                    union((slab, cell), (slab, other))
            if not any(
                eid in self.membership[cell]
                and self.ranges[eid][0] == self.z_values[slab]
                for eid in self.sheet_ids
            ):
                union((slab, cell), (slab - 1, cell))
        self.components = defaultdict(list)
        for key in self.owners:
            self.components[find(key)].append(key)
        self.volume_ids = {
            key: geometry_identity(
                "VOL__",
                {
                    "owner": self.owners[key],
                    "cells": sorted(
                        (slab, self.cell_ids[cell]) for slab, cell in members
                    ),
                },
            )
            for key, members in self.components.items()
        }
        self.volume_for = {key: self.volume_ids[find(key)] for key in self.owners}
