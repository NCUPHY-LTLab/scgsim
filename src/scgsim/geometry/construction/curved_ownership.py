"""Decide which entity owns each Z slab of each planar cell, and group volumes.

Owns the pure-Python reasoning between the planar arrangement and the Z sweep:
conductor overlap validation, port plane heights, the sorted Z levels, per-slab
conductor and solution-region ownership, and the connected components that
become volumes. Creates no native entities.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from scgsim.geometry.construction.curved_context import SlabOwnership
from scgsim.geometry.construction.curved_kernel import _identity


def build_slab_ownership(session, arrangement):
    """Resolve slab ownership, then connect owned cells into volume groups."""
    entities = session.entities
    route = session.route
    # Route A face metal is a sheet; all other finite conductors remain excluded bodies.
    sheet_ids = {
        eid
        for eid, e in entities.items()
        if route == "A"
        and e.material_kind == "conductor"
        and e.part_role == "face_metal"
    }
    ranges = {eid: _z_range(e) for eid, e in entities.items()}
    _validate_conductor_overlaps(session, arrangement, ranges)
    port_planes = _port_planes(session, ranges)
    z_values = sorted(
        {
            z
            for eid, interval in ranges.items()
            for z in (interval[:1] if eid in sheet_ids else interval)
        }
        | set(port_planes.values())
    )
    occupied, owners = _slab_occupancy(
        session, arrangement, sheet_ids, ranges, z_values
    )
    components, volume_ids, volume_for = _volume_groups(
        arrangement, owners, sheet_ids, ranges, z_values
    )
    return SlabOwnership(
        sheet_ids=sheet_ids,
        ranges=ranges,
        port_planes=port_planes,
        z_values=z_values,
        occupied=occupied,
        owners=owners,
        components=components,
        volume_ids=volume_ids,
        volume_for=volume_for,
    )


def _z_range(entity):
    geometry = entity.geometry
    lower = float(geometry.get("z_min_um", geometry.get("z_um", 0.0)))
    return lower, float(
        geometry.get("z_max_um", lower + float(geometry.get("thickness_um", 0.0)))
    )


def _validate_conductor_overlaps(session, arrangement, ranges):
    """Check contact-pad attachment and overlapping-conductor ownership policy."""
    # Native shared cells establish actual XY overlap; ownership uses the
    # same explicit attachment policy as the polygon conductor path.
    from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
    from scgsim.geometry._primitives.entity_validation import (
        _resolve_contact_pad_attachment,
    )
    from scgsim.geometry.planning.mm_contacts import (
        _validate_volumetric_overlap_ownership,
    )

    membership = arrangement.membership
    cells = arrangement.cells
    conductors = tuple(
        sorted(
            (e for e in session.entities.values() if e.material_kind == "conductor"),
            key=lambda e: e.semantic_id,
        )
    )

    def finite_overlap(first, second):
        return any(
            first.semantic_id in membership[cell]
            and second.semantic_id in membership[cell]
            for cell in cells
        )

    for conductor in conductors:
        if conductor.part_role == "contact_pad":
            _resolve_contact_pad_attachment(
                conductor, conductors, finite_overlap=finite_overlap
            )
    for lower, upper in combinations(conductors, 2):
        lower_min, lower_max = ranges[lower.semantic_id]
        upper_min, upper_max = ranges[upper.semantic_id]
        if min(lower_max, upper_max) - max(lower_min, upper_min) <= _TOPOLOGY_EPS_UM:
            continue
        if finite_overlap(lower, upper):
            _validate_volumetric_overlap_ownership(
                lower, upper, conductors, finite_overlap=finite_overlap
            )


def _port_planes(session, ranges):
    """Z of each lumped-port sheet: the host bottom on Route A, mid-height on B."""
    port_planes = {}
    for port in session.prepared.port_sheet_regions:
        host_ids = tuple(
            dict.fromkeys(overlap.host_semantic_id for overlap in port.overlaps)
        )
        intervals = [ranges[eid] for eid in host_ids]
        if len(host_ids) != 2 or len(set(intervals)) != 1:
            raise ValueError("curved lumped-port hosts must share one authored Z range")
        lo, hi = intervals[0]
        # TODO(human): why Route B places the port sheet at mid-height of the host.
        port_planes[port.port_sheet_id] = lo if session.route == "A" else (lo + hi) / 2
    return port_planes


def _slab_occupancy(session, arrangement, sheet_ids, ranges, z_values):
    """Per (slab, cell): conductors filling it, or the solution region owning it."""
    entities = session.entities
    membership = arrangement.membership
    occupied = {}
    owners = {}
    for slab, (lo, hi) in enumerate(zip(z_values, z_values[1:], strict=False)):
        middle = (lo + hi) / 2
        for cell in arrangement.cells:
            candidates = [
                eid
                for eid in membership[cell]
                if eid in entities
                and eid not in sheet_ids
                and ranges[eid][0] < middle < ranges[eid][1]
            ]
            metals = [
                eid for eid in candidates if entities[eid].material_kind == "conductor"
            ]
            if len({entities[eid].net_id for eid in metals}) > 1:
                raise ValueError(
                    "native curved arrangement found overlapping conductors on different Nets"
                )
            if metals:
                occupied[slab, cell] = sorted(metals)
                continue
            domains = [
                eid for eid in candidates if entities[eid].role == "solution_region"
            ]
            if domains:
                owner = max(
                    domains,
                    key=lambda eid: (
                        entities[eid].material_kind == "dielectric",
                        entities[eid].priority,
                        eid,
                    ),
                )
                owners[slab, cell] = owner
    return occupied, owners


def _volume_groups(arrangement, owners, sheet_ids, ranges, z_values):
    """Join same-owner cells into volumes; return components and their plan ids."""
    membership = arrangement.membership
    cell_loops = arrangement.cell_loops
    edge_cells = arrangement.edge_cells
    cell_ids = arrangement.cell_ids
    # Connected planar/slab cells form solution shells; no 3D Boolean repair.
    parents = {key: key for key in owners}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(a, b):
        if a in owners and b in owners and owners[a] == owners[b]:
            parents[find(b)] = find(a)

    for slab, cell in owners:
        for edge in (abs(e) for loop in cell_loops[cell] for e in loop):
            for other in edge_cells[edge]:
                union((slab, cell), (slab, other))
        if not any(
            eid in membership[cell] and ranges[eid][0] == z_values[slab]
            for eid in sheet_ids
        ):
            union((slab, cell), (slab - 1, cell))
    components = defaultdict(list)
    for key in owners:
        components[find(key)].append(key)
    volume_ids = {
        key: _identity(
            "VOL__",
            {
                "owner": owners[key],
                "cells": sorted((slab, cell_ids[cell]) for slab, cell in members),
            },
        )
        for key, members in components.items()
    }
    volume_for = {key: volume_ids[find(key)] for key in owners}
    return components, volume_ids, volume_for
