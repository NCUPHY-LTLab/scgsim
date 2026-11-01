"""Assemble solution volumes from registered surfaces and repaired native shells.

Owns the volume stage of curved construction: split each volume's boundary into
connected shells, probe every shell with a temporary native solid to fix surface
orientation and enclosed volume, and classify the shells that are the exterior or
floating PEC (perfect electric conductor) voids. Does not create caps or walls.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from scgsim.geometry.construction.curved_kernel import identity
from scgsim.geometry.construction.native import (
    add_surface_first_volume,
    rebind_surface_loops,
    rebind_volume_surface_refs,
    surface_outer_wire_edges,
)
from scgsim.geometry.models.topology import (
    InnerPecVoidShellRecord,
    RouteABVolumePlanRecord,
)


@dataclass
class _ShellProbe:
    """State shared by every shell probe of one build."""

    context: object
    book: object
    curves: object
    table: object
    # Per surface: +1 if its native face kept the source-face basis, -1 if flipped.
    surface_basis_parity: dict


def build_volumes(context, book, curves, table):
    """Return the volume plans; native shell repair also rebinds `table.loops`."""
    entities = context.session.entities
    route = context.session.route
    ownership = context.ownership
    # Decompose disconnected boundary shells through their canonical edge
    # incidences. A floating PEC body is an inner shell, not another outer
    # face in the same native surface loop.
    surface_curves = _surface_curve_ids(table.loops)
    volumes = []
    # Source identities survive native shell correction and face replacement.
    probe = _ShellProbe(
        context, book, curves, table, {sid: 1 for sid in book.native_surfaces}
    )
    for root, members in ownership.components.items():
        vid = ownership.volume_ids[root]
        shells = _connected_shells(book.volume_refs[vid], surface_curves)
        enclosed = _orient_shells(probe, shells)
        exterior_index = max(range(len(shells)), key=enclosed.__getitem__)
        inner = _inner_void_shells(entities, book, vid, shells, exterior_index)
        volumes.append(
            RouteABVolumePlanRecord(
                vid,
                ownership.owners[root],
                entities[ownership.owners[root]].material_id,
                tuple(ref for shell in shells for ref in shell),
                valid_routes=(route,),
                exterior_surface_refs=shells[exterior_index],
                inner_pec_void_shells=tuple(inner),
                metadata={
                    "physical_name": entities[ownership.owners[root]].metadata.get(
                        "auto_vacuum_group_id", ownership.owners[root]
                    ),
                    "material_kind": entities[ownership.owners[root]].material_kind,
                    "curved_arrangement_cells": [
                        (s, context.arrangement.cell_ids[c]) for s, c in members
                    ],
                    "native_boundary_shell_enclosed_volumes_um3": enclosed,
                },
            )
        )
    # Rebase all retained observations only after the final global face map is known.
    return rebind_volume_surface_refs(volumes, probe.surface_basis_parity)


def _surface_curve_ids(loops):
    surface_curves = defaultdict(set)
    for loop in loops:
        surface_curves[loop.surface_id].update(ref.curve_id for ref in loop.curve_refs)
    return surface_curves


def _connected_shells(refs, surface_curves):
    """Group a volume's surface references into shells that share curves."""
    pending = {ref.surface_id: ref for ref in tuple(refs)}
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
                if surface_curves[sid] & surface_curves[ref.surface_id]
            ]
            stack.extend(pending.pop(sid) for sid in neighbors)
        shells.append(tuple(connected))
    return shells


def _orient_shells(probe, shells):
    """Probe each shell with a temporary solid; fix orientations in `shells`.

    Rewrites `shells` in place and returns the enclosed volume of every shell.
    Native shell repair rebinds `probe.table.loops` and the native face tags.
    """
    gmsh = probe.context.session.gmsh
    occ = probe.context.session.occ
    table = probe.table
    curve_ids = probe.curves.curve_ids
    surface_basis_parity = probe.surface_basis_parity
    native_surfaces = probe.book.native_surfaces
    enclosed = []
    for index, shell in enumerate(shells):
        repaired = add_surface_first_volume(
            gmsh,
            [[native_surfaces[ref.surface_id] for ref in shell]],
            outer_wire_edges=surface_outer_wire_edges(
                table.surfaces,
                table.loops,
                {cid: edge for edge, cid in curve_ids.items()},
                native_surfaces,
            ),
        )
        table.loops = rebind_surface_loops(
            table.surfaces,
            table.loops,
            {cid: edge for edge, cid in curve_ids.items()},
            native_surfaces,
            repaired,
        )
        native_volume = repaired.volume_tag
        for sid, old_tag in tuple(native_surfaces.items()):
            if old_tag in repaired.face_bindings:
                new_tag, parity = repaired.face_bindings[old_tag]
                native_surfaces[sid] = new_tag
                surface_basis_parity[sid] *= parity
        signed_mass = occ.getMass(3, native_volume)
        enclosed.append(abs(signed_mass))
        occ.synchronize()
        outward_sign = -1 if signed_mass < 0 else 1
        directions = {
            abs(tag): "forward" if tag * outward_sign > 0 else "reversed"
            for dim, tag in gmsh.model.getBoundary([(3, native_volume)], oriented=True)
            if dim == 2
        }
        # Store the temporary observation in the original source-face basis.
        shells[index] = tuple(
            replace(
                ref,
                orientation=(
                    directions[native_surfaces[ref.surface_id]]
                    if surface_basis_parity[ref.surface_id] > 0
                    else "reversed"
                    if directions[native_surfaces[ref.surface_id]] == "forward"
                    else "forward"
                ),
            )
            for ref in shell
        )
        occ.remove([(3, native_volume)], recursive=False)
    return enclosed


def _inner_void_shells(entities, book, vid, shells, exterior_index):
    """Every shell except the largest is a floating PEC void; flip it in place."""
    inner = []
    for index, shell in enumerate(shells):
        if index == exterior_index:
            continue
        metal_owners = tuple(
            sorted(
                {
                    eid
                    for ref in shell
                    for eid in book.surface_data[ref.surface_id]["metadata"][
                        "owner_semantic_ids"
                    ]
                    if entities[eid].material_kind == "conductor"
                }
            )
        )
        # A void boundary faces into the enclosed PEC, opposite to
        # the outward direction of its temporary finite solid.
        shell = tuple(
            replace(
                ref,
                orientation="reversed" if ref.orientation == "forward" else "forward",
            )
            for ref in shell
        )
        shells[index] = shell
        inner.append(
            InnerPecVoidShellRecord(
                identity("VOID__", [vid, sorted(ref.surface_id for ref in shell)]),
                shell,
                identity("COMP__", metal_owners),
                metal_owners,
            )
        )
    return inner
