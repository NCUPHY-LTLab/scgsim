"""Detach canonical topology, orient volume shells and export native XAO."""

from __future__ import annotations

import base64
import hashlib
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from scgsim.geometry.compiler.validation import (
    validate_curve_plan_coverage,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)
from scgsim.geometry.construction.curved_geometry.kernel import (
    CurveDetacher,
    face_loops,
    geometry_identity,
)
from scgsim.geometry.construction.curved_geometry.surfaces import SharedSurfaces
from scgsim.geometry.construction.native import (
    add_surface_first_volume,
    rebind_surface_loops,
    rebind_volume_surface_refs,
    surface_outer_wire_edges,
)
from scgsim.geometry.models.construction import (
    ConstructionPlanRecord,
    RouteABConstructionPlanRecord,
)
from scgsim.geometry.models.topology import (
    CurveRefRecord,
    InnerPecVoidShellRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    RouteABVolumePlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.surfaces import with_surface_contract_metadata
from scgsim.geometry.planning.tags import plan_route_tags
from scgsim.geometry.primitives.entities import component_metadata


class CanonicalRecords:
    """Detach shared topology and apply final global face bindings to the plan."""

    def __init__(self, geometry: SharedSurfaces, operations):
        self.geometry = geometry
        self.planar = geometry.planar
        self.gmsh = geometry.planar.kernel.gmsh
        self.occ = geometry.planar.kernel.occ
        self.operations = operations

    def build(self):
        """Detach records, repair shells, export exact shapes, then validate."""
        self.detach_curves()
        self.detach_surfaces()
        self.detach_contacts()
        self.bind_surface_contracts()
        self.bind_port_terminals()
        self.finish_incidence()
        self.build_volumes()
        self.export_boundaries()
        return self.assemble()

    def detach_curves(self):
        """Derive curve identity from exact native history, not endpoints alone."""
        # Native history and geometry determine curve identity; endpoints alone cannot identify a curve.
        self.used_edges = {
            abs(t)
            for tag in self.geometry.native_surfaces.values()
            for _, t in self.gmsh.model.getBoundary([(2, tag)], oriented=True)
        }
        self.points = {}
        self.curve_ids = {}
        self.curve_records = []
        self.point_curves = defaultdict(set)
        self.detacher = CurveDetacher(self.geometry)
        for edge in sorted(self.used_edges):
            record, edge_points = self.detacher.detach(edge)
            self.curve_ids[edge] = record.curve_id
            self.points.update(edge_points)
            for pid in edge_points:
                self.point_curves[pid].add(record.curve_id)
            self.curve_records.append(record)

    def detach_surfaces(self):
        """Detach ordered wires and attach their curve incidence to surfaces."""
        self.loops = []
        self.surfaces: list[SurfacePlanRecord] = []
        self.interfaces = []
        self.curve_surfaces = defaultdict(set)
        self.curve_owners = defaultdict(set)
        self.curve_volumes = defaultdict(set)
        self.curve_interfaces = defaultdict(set)
        for sid, tag in self.geometry.native_surfaces.items():
            data = self.geometry.surface_data[sid]
            native_loops = face_loops(self.gmsh, self.occ, tag)
            loop_ids = []
            for index, tags in enumerate(native_loops):
                lid = geometry_identity("LOOP__", [sid, index])
                loop_ids.append(lid)
                refs = tuple(
                    CurveRefRecord(
                        self.curve_ids[abs(int(t))], 1 if t > 0 else -1, "boundary"
                    )
                    for t in tags
                )
                self.loops.append(
                    SurfaceLoopRecord(lid, refs, "outer" if index == 0 else "hole", sid)
                )
                for ref in refs:
                    self.curve_surfaces[ref.curve_id].add(sid)
                    self.curve_owners[ref.curve_id].update(
                        data["metadata"]["owner_semantic_ids"]
                    )
                    self.curve_volumes[ref.curve_id].update(
                        data["metadata"]["boundary_volume_ids"]
                    )
                    if data["interface"]:
                        self.curve_interfaces[ref.curve_id].add(data["interface"])
            geometry = {
                "native_curved_face": True,
                "native_surface_type": self.gmsh.model.getType(2, tag),
                "arrangement_surface_id": sid,
                "representation": data["metadata"]["representation"],
                "shell_part": data["face_kind"],
                "source_polygon_ids": tuple(
                    pid
                    for owner in data["metadata"]["owner_semantic_ids"]
                    if owner in self.planar.entities
                    for pid in self.planar.entities[owner].polygon_ids
                ),
            }
            self.surfaces.append(
                SurfacePlanRecord(
                    sid,
                    data["owner"],
                    data["role"],
                    geometry,
                    loop_ids[0],
                    tuple(loop_ids[1:]),
                    interface_id=data["interface"],
                    normal_hint=data["normal"],
                    valid_routes=(self.planar.route,),
                    metadata=data["metadata"],
                )
            )
            if data["interface"]:
                owners_tuple = data["metadata"]["owner_semantic_ids"]
                pair = (owners_tuple[0], owners_tuple[-1])
                self.interfaces.append(
                    InterfacePlanRecord(
                        data["interface"],
                        data["metadata"]["interface_kinds"][0],
                        pair,
                        "native_curved_planar_arrangement_z_sweep",
                        metadata={"curved_arrangement_surface_id": sid},
                    )
                )

    def detach_contacts(self):
        """Detach contact topology and shared conductor component provenance."""
        contact_parents = {
            eid: eid for row in self.geometry.native_contacts for eid in row["pair"]
        }

        def contact_root(eid):
            while contact_parents[eid] != eid:
                eid = contact_parents[eid]
            return eid

        for row in self.geometry.native_contacts:
            a, b = row["pair"]
            contact_parents[contact_root(b)] = contact_root(a)
        contact_members = defaultdict(set)
        for eid in contact_parents:
            contact_members[contact_root(eid)].add(eid)
        contacts: list[MMContactRecord] = []
        for row in self.geometry.native_contacts:
            tag = row["tag"]
            a, b = row["pair"]
            members = tuple(sorted(contact_members[contact_root(a)]))
            component_id = geometry_identity("COMP__", members)
            equipotentials = {
                self.planar.entities[eid].metadata.get("equipotential_id")
                for eid in members
            }
            equipotentials.discard(None)
            if len(equipotentials) > 1:
                raise ValueError(
                    "native curved contact has conflicting equipotential IDs"
                )
            cid = geometry_identity("MM__", [row["pair"], row["key"]])
            contact_curves = {}
            contact_points = {}
            contact_loops = []
            for index, edges in enumerate(
                face_loops(
                    self.gmsh,
                    self.occ,
                    tag,
                )
            ):
                refs = []
                for edge in edges:
                    record, edge_points = self.detacher.detach(abs(edge))
                    contact_curves[record.curve_id] = record
                    contact_points.update(edge_points)
                    refs.append(
                        CurveRefRecord(
                            record.curve_id, 1 if edge > 0 else -1, "boundary"
                        )
                    )
                contact_loops.append(
                    SurfaceLoopRecord(
                        geometry_identity("LOOP__", [cid, index]),
                        tuple(refs),
                        "outer" if index == 0 else "hole",
                    )
                )
            geometry = {
                "native_curved_face": any(
                    c.curve_kind != "line_segment" for c in contact_curves.values()
                ),
                "native_surface_type": self.gmsh.model.getType(2, tag),
                "surface_method": (
                    "planar_arrangement_cap"
                    if row["normal"] is not None
                    else "native_collective_exact_edge_sweep"
                ),
                "sweep_direction": None
                if row["normal"] is not None
                else (0.0, 0.0, 1.0),
                "outer_loop_ref": contact_loops[0].loop_id,
                "hole_loop_refs": tuple(loop.loop_id for loop in contact_loops[1:]),
                "points": [
                    asdict(PointPlanRecord(pid, xyz))
                    for pid, xyz in contact_points.items()
                ],
                "curves": [asdict(c) for c in contact_curves.values()],
                "surface_loops": [asdict(loop) for loop in contact_loops],
            }
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
            contacts.append(
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
                    self.planar.entities[a].polygon_ids,
                    self.planar.entities[b].polygon_ids,
                    polygon,
                    self.occ.getMass(2, tag),
                    row["normal"],
                    component_id,
                    self.planar.entities[a].net_id,
                    next(iter(equipotentials), None),
                    layer_provenance={
                        eid: self.planar.entities[eid].metadata.get("source_layer_name")
                        for eid in row["pair"]
                    },
                    material_provenance={
                        eid: self.planar.entities[eid].material_id
                        for eid in row["pair"]
                    },
                    source_provenance={
                        "route": self.planar.route,
                        "recognition_rule": row["operation"],
                        "hidden_solver_contact": True,
                        "source_entity_ids": row["pair"],
                    },
                    geometry_ref=geometry,
                )
            )
        self.mm_contacts = tuple(contacts)

    def bind_surface_contracts(self):
        """Bind port component identities before the ordinary surface contract."""
        component_by_entity = {
            eid: component["conductor_component_id"]
            for component in component_metadata(self.mm_contacts)
            for eid in component["members"]
        }
        for index, surface_record in enumerate(self.surfaces):
            if surface_record.surface_role != "lumped_port":
                continue
            attribute = surface_record.metadata["physical_attribute"]
            self.surfaces[index] = replace(
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
        ordinary = {
            surface.surface_id: surface
            for surface in with_surface_contract_metadata(
                self.planar.prepared,
                route=self.planar.route,
                interfaces=tuple(self.interfaces),
                mm_contacts=self.mm_contacts,
                surfaces=tuple(
                    surface
                    for surface in self.surfaces
                    if surface.surface_role != "lumped_port"
                ),
            )
        }
        self.surfaces = [
            ordinary.get(surface.surface_id, surface) for surface in self.surfaces
        ]

    def bind_port_terminals(self):
        """Bind exact terminal identities from canonical shared curves."""
        # Exact terminal identities come from shared arrangement curves, not from polygon chord tests.
        for port_surface in tuple(
            surface
            for surface in self.surfaces
            if surface.surface_role == "lumped_port"
        ):
            port_curves = {
                ref.curve_id
                for loop in self.loops
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
                                self.planar.route == "A"
                                or surface.geometry_ref.get("shell_part") == "sidewall"
                            )
                            and surface.surface_id in self.curve_surfaces[cid]
                            for surface in self.surfaces
                        )
                    )
                )
                overlaps.append({**overlap, "native_terminal_curve_ids": terminal_ids})
                if self.planar.route == "B":
                    for index, owner_surface in enumerate(self.surfaces):
                        if (
                            owner_surface.owner_semantic_id != owner
                            or owner_surface.geometry_ref.get("shell_part")
                            != "sidewall"
                        ):
                            continue
                        if any(
                            owner_surface.surface_id in self.curve_surfaces[cid]
                            for cid in terminal_ids
                        ):
                            binding = {
                                "port_surface_id": port_surface.surface_id,
                                "overlap_id": overlap["overlap_id"],
                                "host_semantic_id": owner,
                            }
                            self.surfaces[index] = replace(
                                owner_surface,
                                metadata={
                                    **owner_surface.metadata,
                                    "route_b_port_sheet_sidewall_partition": True,
                                    "route_b_port_sheet_bindings": (
                                        *owner_surface.metadata.get(
                                            "route_b_port_sheet_bindings", ()
                                        ),
                                        binding,
                                    ),
                                },
                            )
            index = next(
                i
                for i, surface in enumerate(self.surfaces)
                if surface.surface_id == port_surface.surface_id
            )
            self.surfaces[index] = replace(
                port_surface,
                metadata={
                    **port_surface.metadata,
                    "source_provenance": {
                        **port_surface.metadata["source_provenance"],
                        "overlaps": overlaps,
                    },
                },
            )

    def finish_incidence(self):
        """Attach final surface and curve incidences to canonical records."""
        self.curves = tuple(
            replace(
                curve,
                used_by_surface_ids=tuple(sorted(self.curve_surfaces[curve.curve_id])),
                owner_semantic_ids=tuple(sorted(self.curve_owners[curve.curve_id])),
                interface_ids=tuple(sorted(self.curve_interfaces[curve.curve_id])),
                boundary_volume_ids=tuple(sorted(self.curve_volumes[curve.curve_id])),
            )
            for curve in self.curve_records
        )
        self.point_records = tuple(
            PointPlanRecord(
                pid, xyz, used_by_curve_ids=tuple(sorted(self.point_curves[pid]))
            )
            for pid, xyz in self.points.items()
        )

    def build_volumes(self):
        """Repair disconnected shells and rebase only after the global face map."""
        # Decompose disconnected boundary shells through their canonical edge
        # incidences. A floating PEC body is an inner shell, not another outer
        # face in the same native surface loop.
        self.surface_curves = defaultdict(set)
        for loop in self.loops:
            self.surface_curves[loop.surface_id].update(
                ref.curve_id for ref in loop.curve_refs
            )
        volumes: list[RouteABVolumePlanRecord] = []
        # Source identities survive native shell correction and face replacement.
        self.surface_basis_parity = {sid: 1 for sid in self.geometry.native_surfaces}
        for root, members in self.planar.components.items():
            vid = self.planar.volume_ids[root]
            refs = tuple(self.geometry.volume_refs[vid])
            pending = {ref.surface_id: ref for ref in refs}
            shells = []
            while pending:
                first = next(iter(pending))
                stack = [pending.pop(first)]
                connected = []
                while stack:
                    ref = stack.pop()
                    connected.append(ref)
                    neighbors = [
                        sid
                        for sid in pending
                        if self.surface_curves[sid]
                        & self.surface_curves[ref.surface_id]
                    ]
                    stack.extend(pending.pop(sid) for sid in neighbors)
                shells.append(tuple(connected))
            enclosed = []
            self._orient_shells(shells, enclosed)
            exterior_index = max(range(len(shells)), key=enclosed.__getitem__)
            inner = []
            for index, shell in enumerate(shells):
                if index == exterior_index:
                    continue
                metal_owners = tuple(
                    sorted(
                        {
                            eid
                            for ref in shell
                            for eid in self.geometry.surface_data[ref.surface_id][
                                "metadata"
                            ]["owner_semantic_ids"]
                            if self.planar.entities[eid].material_kind == "conductor"
                        }
                    )
                )
                # A void boundary faces into the enclosed PEC, opposite to
                # the outward direction of its temporary finite solid.
                shell = tuple(
                    replace(
                        ref,
                        orientation="reversed"
                        if ref.orientation == "forward"
                        else "forward",
                    )
                    for ref in shell
                )
                shells[index] = shell
                inner.append(
                    InnerPecVoidShellRecord(
                        geometry_identity(
                            "VOID__", [vid, sorted(ref.surface_id for ref in shell)]
                        ),
                        shell,
                        geometry_identity("COMP__", metal_owners),
                        metal_owners,
                    )
                )
            volumes.append(
                RouteABVolumePlanRecord(
                    vid,
                    self.planar.owners[root],
                    self.planar.entities[self.planar.owners[root]].material_id,
                    tuple(ref for shell in shells for ref in shell),
                    valid_routes=(self.planar.route,),
                    exterior_surface_refs=shells[exterior_index],
                    inner_pec_void_shells=tuple(inner),
                    metadata={
                        "physical_name": self.planar.entities[
                            self.planar.owners[root]
                        ].metadata.get(
                            "auto_vacuum_group_id", self.planar.owners[root]
                        ),
                        "material_kind": self.planar.entities[
                            self.planar.owners[root]
                        ].material_kind,
                        "curved_arrangement_cells": [
                            (s, self.planar.cell_ids[c]) for s, c in members
                        ],
                        "native_boundary_shell_enclosed_volumes_um3": enclosed,
                    },
                )
            )
        # Rebase all retained observations only after the final global face map is known.
        self.volumes = rebind_volume_surface_refs(volumes, self.surface_basis_parity)

    def _orient_shells(self, shells, enclosed):
        """Observe native shell orientations before applying the global face map."""
        for index, shell in enumerate(shells):
            repaired = add_surface_first_volume(
                self.gmsh,
                [[self.geometry.native_surfaces[ref.surface_id] for ref in shell]],
                outer_wire_edges=surface_outer_wire_edges(
                    self.surfaces,
                    self.loops,
                    {cid: edge for edge, cid in self.curve_ids.items()},
                    self.geometry.native_surfaces,
                ),
            )
            self.loops = rebind_surface_loops(
                self.surfaces,
                self.loops,
                {cid: edge for edge, cid in self.curve_ids.items()},
                self.geometry.native_surfaces,
                repaired,
            )
            native_volume = repaired.volume_tag
            for sid, old_tag in tuple(self.geometry.native_surfaces.items()):
                if old_tag in repaired.face_bindings:
                    new_tag, parity = repaired.face_bindings[old_tag]
                    self.geometry.native_surfaces[sid] = new_tag
                    self.surface_basis_parity[sid] *= parity
            signed_mass = self.occ.getMass(3, native_volume)
            enclosed.append(abs(signed_mass))
            self.occ.synchronize()
            outward_sign = -1 if signed_mass < 0 else 1
            directions = {
                abs(tag): "forward" if tag * outward_sign > 0 else "reversed"
                for dim, tag in self.gmsh.model.getBoundary(
                    [(3, native_volume)], oriented=True
                )
                if dim == 2
            }
            # Store the temporary observation in the original source-face basis.
            shells[index] = tuple(
                replace(
                    ref,
                    orientation=(
                        directions[self.geometry.native_surfaces[ref.surface_id]]
                        if self.surface_basis_parity[ref.surface_id] > 0
                        else "reversed"
                        if directions[self.geometry.native_surfaces[ref.surface_id]]
                        == "forward"
                        else "forward"
                    ),
                )
                for ref in shell
            )
            self.occ.remove([(3, native_volume)], recursive=False)

    def export_boundaries(self):
        """Remove scratch geometry before exporting exact shared boundary XAO."""
        # Preserve exact shared boundary shapes in one detached XAO, before any volume exists.
        active_faces = set(self.geometry.native_surfaces.values())
        self.occ.remove(
            [(2, t) for _, t in self.occ.getEntities(2) if t not in active_faces],
            recursive=False,
        )
        self.occ.remove(
            [(1, t) for _, t in self.occ.getEntities(1) if t not in self.used_edges],
            recursive=False,
        )
        self.occ.synchronize()
        active_points = {
            abs(t)
            for edge in self.used_edges
            for _, t in self.gmsh.model.getBoundary([(1, edge)])
        }
        self.occ.remove(
            [(0, t) for _, t in self.occ.getEntities(0) if t not in active_points],
            recursive=False,
        )
        self.occ.synchronize()
        for edge, cid in self.curve_ids.items():
            physical = self.gmsh.model.addPhysicalGroup(1, [edge])
            self.gmsh.model.setPhysicalName(1, physical, "SGB_CURVE::" + cid)
        for sid, tag in self.geometry.native_surfaces.items():
            physical = self.gmsh.model.addPhysicalGroup(2, [tag])
            self.gmsh.model.setPhysicalName(2, physical, "SGB_SURFACE::" + sid)
        with TemporaryDirectory(prefix="scgsim-curved-boundaries-") as temporary:
            path = Path(temporary) / "boundaries.xao"
            self.gmsh.write(str(path))
            payload = path.read_bytes()
        self.metadata = {
            "backend_strategy": "surface_plan_first_bottom_up_occ",
            "curved_boundary_xao": base64.b64encode(payload).decode("ascii"),
            "curved_boundary_xao_sha256": hashlib.sha256(payload).hexdigest(),
            "curved_arrangement": {
                "native_gmsh_version": self.gmsh.__version__,
                "source_boundary_bindings": [
                    asdict(r) for r in self.planar.prepared.boundary_curves
                ],
                "source_footprints": self.planar.source_footprints,
                "reconstruction_operations": self.operations,
                "planar_cells": [
                    {
                        "cell_id": self.planar.cell_ids[c],
                        "source_membership": sorted(self.planar.membership[c]),
                    }
                    for c in sorted(self.planar.cells)
                ],
                "native_curve_intervals": [
                    {
                        "curve_id": self.curve_ids[e],
                        "geometry": self.planar.curve_descriptors[source],
                    }
                    for e, (level, source) in self.detacher.lifted_inverse.items()
                    if e in self.curve_ids
                ],
                "z_levels_um": self.planar.z_values,
                "shared_curves": len(self.curves),
                "shared_surfaces": len(self.surfaces),
                "solution_volumes": len(self.volumes),
                "sidewall_method": "native_collective_exact_edge_sweep_reusing_canonical_edges",
            },
        }

    def assemble(self):
        """Assemble the established plan type and run validators in order."""
        # Public route dispatch selects A/B before native construction. Keep the
        # existing constructor selection and its failure behavior for other routes.
        plan_type = cast(
            type[RouteABConstructionPlanRecord],
            (
                RouteABConstructionPlanRecord
                if self.planar.route in {"A", "B"}
                else ConstructionPlanRecord
            ),
        )
        plan = plan_type(
            self.planar.route,
            interfaces=tuple(self.interfaces),
            points=self.point_records,
            curves=self.curves,
            surface_loops=tuple(self.loops),
            surfaces=tuple(self.surfaces),
            volumes=self.volumes,
            tags=plan_route_tags(
                route=self.planar.route,
                surfaces=tuple(self.surfaces),
                volumes=self.volumes,
            ),
            metadata=self.metadata,
            port_sheet_regions=self.planar.prepared.port_sheet_regions,
            mm_contacts=self.mm_contacts,
        )

        validate_curve_plan_coverage(
            points=plan.points,
            curves=plan.curves,
            surface_loops=plan.surface_loops,
            surfaces=plan.surfaces,
        )
        validate_volume_surface_closure(
            volumes=plan.volumes,
            surfaces=plan.surfaces,
            surface_loops=plan.surface_loops,
        )
        validate_tag_plan_coverage(
            surfaces=plan.surfaces, volumes=plan.volumes, tags=plan.tags
        )
        return plan
