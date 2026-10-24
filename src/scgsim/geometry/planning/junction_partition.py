"""Source-domain junction partitioning; independent of native and EPR analysis."""

from __future__ import annotations


import math

from bisect import bisect_left, bisect_right

from collections import defaultdict

from collections.abc import Mapping, Sequence

from dataclasses import fields, replace

from decimal import ROUND_FLOOR, Decimal

from fractions import Fraction

from itertools import pairwise

from typing import Any

import gdstk

from scgsim.geometry.models.input import GeometryBuildInput, LayoutPolygonSpec
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry._primitives.grid import (
    PolygonRegion, polygon_boolean, polygon_world_frame, with_input_grid,
)
from scgsim.geometry.models.physical import (
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
)
from scgsim.semantics.route_a import geometry_z_range

from scgsim.geometry._primitives.loops import (
    _cancel_reversed_planar_edges,
    _simple_planar_loops_from_edges,
)

from scgsim.geometry.models.junctions import JunctionGeometry, RectangularJunctionSpec
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class JunctionPartitionFormat:
    metadata_key: str = "junction_partitions"
    source_hash_key: str = "junction_authored_input_sha256"
    method: str = "scgsim.geometry.junction-partition.v1"
    id_prefix: str = "junction"
    junction_metadata_key: str = "junction_id"
    support_schema: str = "scgsim.geometry.junction-support-plane.v1"


DEFAULT_FORMAT = JunctionPartitionFormat()


def canonical_sha256(value):
    # Preserve the existing AEDT algorithm's canonical identities at its adapter.
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


PRECISION_UM = 1e-9


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


_SUPPORT_PLANE_SCHEMA = DEFAULT_FORMAT.support_schema


def _junction_contact_inventory(
    build_input: GeometryBuildInput,
    source_polygon_id: str,
    terminal_a_net: str,
    terminal_b_net: str,
    stored_binding: Mapping[str, Any] | None = None,
    *,
    support_schema: str = _SUPPORT_PLANE_SCHEMA,
) -> dict[str, Any]:
    """Resolve one authored support and its source/effective incident conductors.

    A projected overlap alone is not a contact. Source intervals remain
    authoritative when profile collapse makes disjoint films share effective Z.
    The common physical plan alone projects an explicitly bound support.
    """
    regions = [
        region
        for region in build_input.port_sheet_regions
        if region.source_polygon_id == source_polygon_id
    ]
    if len(regions) > 1:
        raise ValueError("junction source sheet has ambiguous port-region identity")
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    region = regions[0] if regions else None
    sheet = region if region is not None else polygons[source_polygon_id]
    sheet_rings = _source_contact_rings(sheet)
    raw_plan = build_input.metadata.get("physical_representation")
    plan = (
        PhysicalRepresentationPlan.from_payload(raw_plan)
        if raw_plan is not None
        else None
    )
    facts = {}
    for entity in build_input.entities:
        if entity.material_kind != "conductor":
            continue
        projection = entity.metadata.get("physical_representation")
        if projection is not None:
            if plan is None:
                raise ValueError(
                    "junction lowered source requires its common physical plan"
                )
            source_interval = tuple(projection["source_interval_um"])
            physical_id = projection["physical_layer_id"]
            declaration = (
                plan.declaration_index[physical_id] if physical_id is not None else None
            )
        else:
            source_interval = geometry_z_range(entity.geometry, entity.semantic_id)
            raw = entity.metadata.get("physical_layer")
            declaration = (
                PhysicalLayerDeclaration.from_payload(raw) if raw is not None else None
            )
            physical_id = (
                declaration.physical_layer_id if declaration is not None else None
            )
        anchor = None
        if declaration is not None and declaration.feature_kind in {
            "stack_film",
            "local_film",
        }:
            authored = declaration.attachment_z_um
            if source_interval[0] <= authored <= source_interval[1]:
                anchor = authored
        elif source_interval[0] == source_interval[1]:
            anchor = source_interval[0]
        facts[entity.semantic_id] = {
            "entity": entity,
            "source_interval_um": source_interval,
            "effective_interval_um": geometry_z_range(
                entity.geometry, entity.semantic_id
            ),
            "physical_layer_id": physical_id,
            "anchor": anchor,
        }

    def contains(interval, z):
        return interval[0] <= z <= interval[1]

    def host_record(eid, pid):
        fact = facts[eid]
        return {
            "source_entity_id": eid,
            "source_polygon_id": pid,
            "physical_layer_id": fact["physical_layer_id"],
            "source_interval_um": list(fact["source_interval_um"]),
        }

    hosts = []
    if region is not None and region.overlaps:
        for overlap in region.overlaps:
            if (
                overlap.port_sheet_id != region.port_sheet_id
                or overlap.port_polygon_id != source_polygon_id
            ):
                raise ValueError(
                    "junction port overlap references another source sheet"
                )
            matches = [
                eid
                for eid, fact in facts.items()
                if (
                    eid == overlap.host_semantic_id
                    or fact["entity"].metadata.get("source_semantic_id")
                    == overlap.host_semantic_id
                )
                and overlap.host_polygon_id in fact["entity"].polygon_ids
            ]
            if len(matches) != 1 or overlap.host_polygon_id not in polygons:
                raise ValueError(
                    "junction port host lacks exact Entity/polygon ownership"
                )
            hosts.append(host_record(matches[0], overlap.host_polygon_id))
        anchors = {
            facts[host["source_entity_id"]]["anchor"]
            for host in hosts
            if facts[host["source_entity_id"]]["anchor"] is not None
        }
        if len(anchors) != 1:
            raise ValueError(
                "junction port hosts lack one common authored support plane"
            )
        source_z = next(iter(anchors))
        if any(
            not contains(
                facts[host["source_entity_id"]]["source_interval_um"], source_z
            )
            for host in hosts
        ):
            raise ValueError(
                "junction port host is disjoint from its authored support plane"
            )
        basis = "port_overlap_hosts"
    else:
        contacting = []
        for eid, fact in facts.items():
            entity = fact["entity"]
            if entity.net_id not in {terminal_a_net, terminal_b_net}:
                continue
            for pid in entity.polygon_ids:
                if pid != source_polygon_id and _closed_contact(
                    [sheet_rings], [_source_contact_rings(polygons[pid])]
                ):
                    contacting.append((eid, pid))
        candidates = {
            facts[eid]["anchor"]
            for eid, _ in contacting
            if facts[eid]["anchor"] is not None
        }
        candidates = {
            z
            for z in candidates
            if all(
                any(
                    facts[eid]["entity"].net_id == net
                    and contains(facts[eid]["source_interval_um"], z)
                    and facts[eid]["anchor"] in {None, z}
                    for eid, _ in contacting
                )
                for net in (terminal_a_net, terminal_b_net)
            )
        }
        if len(candidates) != 1:
            raise ValueError(
                "junction terminal contacts lack a unique authored support plane"
            )
        source_z = next(iter(candidates))
        hosts = [
            host_record(eid, pid)
            for eid, pid in contacting
            if contains(facts[eid]["source_interval_um"], source_z)
            and facts[eid]["anchor"] in {None, source_z}
        ]
        basis = "terminal_closed_contact"
    # Store source facts only: a later approximation may change effective Z.
    hosts = sorted(
        {canonical_sha256(host): host for host in hosts}.values(),
        key=lambda item: (item["source_entity_id"], item["source_polygon_id"]),
    )
    binding = {
        "schema_version": support_schema,
        "source_polygon_id": source_polygon_id,
        "source_z_um": source_z,
        "basis": basis,
        "source_hosts": hosts,
    }
    if stored_binding is not None and _plain(stored_binding) != binding:
        raise ValueError(
            "junction stored support-plane binding differs from current source"
        )
    effective_planes = set()
    for host in hosts:
        fact = facts[host["source_entity_id"]]
        if fact["anchor"] is None:
            continue
        if plan is None:
            effective_z = source_z
        elif fact["physical_layer_id"] is None:
            effective_z = plan.map_point((0.0, 0.0, source_z))[2]
        else:
            effective_z = plan.project_support_point(
                (0.0, 0.0, source_z), fact["physical_layer_id"]
            )[2]
        effective_planes.add(effective_z)
    if len(effective_planes) != 1:
        raise ValueError(
            "junction anchor hosts map to different effective support planes"
        )
    effective_z = next(iter(effective_planes))
    if any(
        not contains(
            facts[host["source_entity_id"]]["effective_interval_um"], effective_z
        )
        for host in hosts
    ):
        raise ValueError(
            "junction host does not meet its mapped effective support plane"
        )
    eligible = sorted(
        eid
        for eid, fact in facts.items()
        if contains(fact["source_interval_um"], source_z)
        and contains(fact["effective_interval_um"], effective_z)
    )
    return {
        **binding,
        "effective_z_um": effective_z,
        "eligible_entity_ids": eligible,
    }


def _source_contact_rings(polygon):
    return tuple(
        tuple(tuple(Fraction(str(float(value))) for value in point) for point in ring)
        for ring in (polygon.exterior, *polygon.holes)
    )


def _junction_source_binding(inventory):
    return {
        name: _plain(inventory[name])
        for name in (
            "schema_version",
            "source_polygon_id",
            "source_z_um",
            "basis",
            "source_hosts",
        )
    }


def _edges(rings):
    for ring in rings:
        yield from zip(ring, (*ring[1:], ring[0]))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on(point, a, b):
    return _cross(a, b, point) == 0 and all(
        min(a[i], b[i]) <= point[i] <= max(a[i], b[i]) for i in (0, 1)
    )


def _inside_ring(point, ring):
    inside = False
    for a, b in _edges((ring,)):
        if _on(point, a, b):
            return 0  # Boundary, distinct from inside and outside.
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = Fraction(b[0] - a[0], b[1] - a[1]) * (point[1] - a[1]) + a[0]
            if point[0] < x:
                inside = not inside
    return 1 if inside else -1


def _inside(point, rings):
    return _inside_ring(point, rings[0]) >= 0 and not any(
        _inside_ring(point, hole) == 1 for hole in rings[1:]
    )


def _section(rings, axis, coordinate, low, high):
    """Exact closed contact on a specified edge in Boolean integer coordinates."""
    other = 1 - axis
    marks = {Fraction(low), Fraction(high)}
    for a, b in _edges(rings):
        if a[axis] == b[axis] == coordinate:
            marks.update(Fraction(v[other]) for v in (a, b) if low <= v[other] <= high)
        elif a[axis] != b[axis] and min(a[axis], b[axis]) <= coordinate <= max(
            a[axis], b[axis]
        ):
            v = (
                Fraction(coordinate - a[axis], b[axis] - a[axis])
                * (b[other] - a[other])
                + a[other]
            )
            if low <= v <= high:
                marks.add(v)
    ordered = sorted(marks)

    def point(v):
        return (coordinate, v) if axis == 0 else (v, coordinate)

    intervals = [
        (a, b) for a, b in pairwise(ordered) if _inside(point((a + b) / 2), rings)
    ]
    points = [v for v in ordered if _inside(point(v), rings)]
    return intervals, points


def _covered(polygons, axis, coordinate, low, high):
    intervals = sorted(
        interval
        for rings in polygons
        for interval in _section(rings, axis, coordinate, low, high)[0]
    )
    end = Fraction(low)
    for a, b in intervals:
        if a > end:
            return False
        end = max(end, b)
    return end >= high


def _line_intervals(polygons):
    result = defaultdict(list)
    for rings in polygons:
        for a, b in _edges(rings):
            dx, dy = b[0] - a[0], b[1] - a[1]
            divisor = math.gcd(dx, dy)
            if not divisor:
                continue
            dx, dy = dx // divisor, dy // divisor
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            key = (dx, dy, dx * a[1] - dy * a[0])
            u, v = a[0] * dx + a[1] * dy, b[0] * dx + b[1] * dy
            result[key].append((min(u, v), max(u, v)))
    return result


def _positive_boundary_contact(left, right):
    a, b = _line_intervals(left), _line_intervals(right)
    return any(
        max(x0, y0) < min(x1, y1)
        for key in a.keys() & b.keys()
        for x0, x1 in a[key]
        for y0, y1 in b[key]
    )


def _closed_contact(left, right):
    for a in left:
        for b in right:
            ax = [p[0] for p in a[0]]
            ay = [p[1] for p in a[0]]
            bx = [p[0] for p in b[0]]
            by = [p[1] for p in b[0]]
            if (
                max(ax) < min(bx)
                or max(bx) < min(ax)
                or max(ay) < min(by)
                or max(by) < min(ay)
            ):
                continue
            if {p for ring in a for p in ring} & {p for ring in b for p in ring}:
                return True
            if any(_inside(p, b) for p in a[0]) or any(_inside(p, a) for p in b[0]):
                return True
            for p, q in _edges(a):
                for r, s in _edges(b):
                    c = (
                        _cross(p, q, r),
                        _cross(p, q, s),
                        _cross(r, s, p),
                        _cross(r, s, q),
                    )
                    if any((_on(r, p, q), _on(s, p, q), _on(p, r, s), _on(q, r, s))):
                        return True
                    if c[0] * c[1] < 0 and c[2] * c[3] < 0:
                        return True
    return False


def _region(polygons):
    result = []
    for rings in polygons:
        exterior = gdstk.Polygon(
            [(x * PRECISION_UM, y * PRECISION_UM) for x, y in rings[0]]
        )
        holes = [
            gdstk.Polygon([(x * PRECISION_UM, y * PRECISION_UM) for x, y in ring])
            for ring in rings[1:]
        ]
        result.extend(_boolean([exterior], holes, "not"))
    return result


def _boolean(left, right, operation):
    # SGB's Boolean wrapper filters small areas. Partitioning must retain every
    # residual, so reuse its precision and loop lowering without that filter.
    return list(polygon_boolean(left, right, operation, precision=PRECISION_UM) or ())


def _rings(region):
    result = []
    for polygon in region:
        if isinstance(polygon, PolygonRegion):
            result.append(tuple(tuple((round(x / PRECISION_UM), round(y / PRECISION_UM))
                               for x,y in ring)
                               for ring in (polygon.exterior, *polygon.holes)))
            continue
        points = tuple(
            (round(x / PRECISION_UM), round(y / PRECISION_UM))
            for x, y in polygon.points
        )
        # Atomize at every retained vertex, including vertices inside another
        # edge. This preserves SGB minimum-coherence/point-fragment semantics.
        ordered = sorted(set(points))
        xs = [point[0] for point in ordered]
        edges = []
        for a, b in _edges((points,)):
            candidates = ordered[
                bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
            ]
            on_edge = sorted(
                (p for p in candidates if _on(p, a, b)),
                key=lambda p: (
                    (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
                ),
            )
            edges.extend(pairwise(on_edge))
        loops = _simple_planar_loops_from_edges(_cancel_reversed_planar_edges(edges))

        def area(ring):
            return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,))))

        outer = max(loops, key=area)
        holes = tuple(ring for ring in loops if ring is not outer)
        if any(_inside_ring(hole[0], outer) != 1 for hole in holes):
            raise ValueError(
                "junction residual has point-connected or non-interior components"
            )
        result.append((outer, *holes))
    return result


def _rectangle(x0, y0, x1, y1):
    return ((((x0, y0), (x1, y0), (x1, y1), (x0, y1)),),)


def _contacts(polygons, span, width):
    """Find extrema of area, segment and point contact with closed authored S."""
    region = _boolean(_region(polygons), _region(_rectangle(0, 0, span, width)), "and")
    positions = [round(x / PRECISION_UM) for p in region for x, _ in p.points]
    for rings in polygons:
        for axis, c, low, high in (
            (0, 0, 0, width),
            (0, span, 0, width),
            (1, 0, 0, span),
            (1, width, 0, span),
        ):
            intervals, points = _section(rings, axis, c, low, high)
            if axis == 0 and (intervals or points):
                positions.append(c)
            elif axis == 1:
                positions.extend(points)
                positions.extend(v for pair in intervals for v in pair)
    return (min(positions), max(positions)) if positions else None


def _bound_decimal(bound):
    rational = Fraction(bound)
    return (
        Decimal(rational.numerator)
        / Decimal(rational.denominator)
        * Decimal(str(PRECISION_UM))
    )


def _candidate_indices(bound, step, *, left):
    q = _bound_decimal(bound) / Decimal(str(step))
    floor = int(q.to_integral_value(rounding=ROUND_FLOOR))
    aligned = q == floor
    if left:
        return ([floor] if aligned else []) + [floor + 1]
    return ([floor] if aligned else []) + [floor if not aligned else floor - 1]


def _cut(index, step):
    requested = float(Decimal(index) * Decimal(str(step)))
    return requested, round(requested / PRECISION_UM)


def _end_candidate(free, own, foreign, cut, span, width, *, left):
    clip = _rectangle(0, 0, cut, width) if left else _rectangle(cut, 0, span, width)
    region = (
        _boolean(free, _region(clip), "and")
        if (cut > 0 if left else cut < span)
        else []
    )
    try:
        end = _rings(region)
    except ValueError as exc:
        return None, {"valid": False, "reason": str(exc)}
    if len(end) > 1:
        return None, {
            "valid": False,
            "reason": "nonempty end has disconnected components",
            "components": len(end),
        }
    if end and not _positive_boundary_contact(end, own):
        return None, {
            "valid": False,
            "reason": "end lacks positive-length own-arm contact",
        }
    if _closed_contact(end, foreign):
        return None, {
            "valid": False,
            "reason": "end contacts wrong terminal or third net",
        }
    if not _covered([*own, *end], 0, cut, 0, width):
        return None, {
            "valid": False,
            "reason": "designated end edge is not wholly covered",
        }
    return end, {"valid": True, "empty_residual": not end, "components": len(end)}


def _model_geometry(build_input, originals, records, output_format=DEFAULT_FORMAT):
    """Trace source incidences and owned seams before topology, then slice at two cuts."""
    authored = {r["authored_polygon_id"] for r in records.values()}
    eligible_polygons = {
        jid: {
            pid
            for entity in build_input.entities
            if entity.semantic_id in record["support_plane"]["eligible_entity_ids"]
            for pid in entity.polygon_ids
            if pid not in authored
        }
        for jid, record in records.items()
    }
    sheet_metals = defaultdict(set)
    for jid, record in records.items():
        sheet_metals[record["authored_polygon_id"]].update(eligible_polygons[jid])
    metal_ids = set().union(*eligible_polygons.values())
    needed = metal_ids | authored
    rings = {
        pid: tuple(
            tuple(tuple(round(v / PRECISION_UM) for v in point) for point in ring)
            for ring in (originals[pid].exterior, *originals[pid].holes)
        )
        for pid in needed
    }
    edges, events, members = {}, defaultdict(dict), defaultdict(set)
    intersection_deviation = 0.0

    def event(edge, point, parameter):
        prior = events[edge].get(point)
        if prior is not None and prior != parameter:
            raise ValueError(
                "numerical junction incidence collapses distinct source parameters"
            )
        events[edge][point] = parameter
        members[point].add(edge)

    for pid, region in rings.items():
        for index, ring in enumerate(region):
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,)))
            if len(set(ring)) < 3 or not area:
                raise ValueError(
                    "numerical junction source polygon/hole collapses in model coordinates"
                )
            if index and any(_inside_ring(point, region[0]) != 1 for point in ring):
                raise ValueError(
                    "numerical junction source hole loses model interior topology"
                )
            for n, (a, b) in enumerate(_edges((ring,))):
                key = (pid, index, n)
                edges[key] = a, b
                event(key, a, Fraction(0))
                event(key, b, Fraction(1))
    for arm, (a, b) in edges.items():
        if arm[0] not in metal_ids:
            continue
        ab = b[0] - a[0], b[1] - a[1]
        for sheet, (c, d) in edges.items():
            if arm[0] not in sheet_metals.get(sheet[0], ()):
                continue
            if (
                max(a[0], b[0]) < min(c[0], d[0])
                or max(c[0], d[0]) < min(a[0], b[0])
                or max(a[1], b[1]) < min(c[1], d[1])
                or max(c[1], d[1]) < min(a[1], b[1])
            ):
                continue
            cd = d[0] - c[0], d[1] - c[1]
            determinant = ab[0] * cd[1] - ab[1] * cd[0]
            if determinant:
                ac = c[0] - a[0], c[1] - a[1]
                t = Fraction(ac[0] * cd[1] - ac[1] * cd[0], determinant)
                u = Fraction(ac[0] * ab[1] - ac[1] * ab[0], determinant)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    exact = tuple(a[i] + t * ab[i] for i in (0, 1))
                    point = tuple(round(v) for v in exact)
                    intersection_deviation = max(
                        intersection_deviation,
                        max(float(abs(point[i] - exact[i])) for i in (0, 1))
                        * PRECISION_UM,
                    )
                    event(arm, point, t)
                    event(sheet, point, u)
            elif _cross(a, b, c) == 0:
                for point in {a, b, c, d}:
                    if _on(point, a, b) and _on(point, c, d):
                        for key, start, delta in ((arm, a, ab), (sheet, c, cd)):
                            norm = delta[0] ** 2 + delta[1] ** 2
                            parameter = Fraction(
                                (point[0] - start[0]) * delta[0]
                                + (point[1] - start[1]) * delta[1],
                                norm,
                            )
                            event(key, point, parameter)
    affected_owners = {
        end["source_entity_id"] for record in records.values() for end in record["ends"]
    }
    # A normalized Entity may comprise adjacent authored polygons whose shared
    # seam has different subdivisions. Register those existing endpoint nodes
    # before Boolean topology; no new crossing or contact tolerance is inferred.
    for entity in build_input.entities:
        if entity.semantic_id not in affected_owners:
            continue
        owned = set(entity.polygon_ids) - authored
        owned_edges = [(key, a, b) for key, (a, b) in edges.items() if key[0] in owned]
        for n, (left, a, b) in enumerate(owned_edges):
            for right, c, d in owned_edges[n + 1 :]:
                if left[0] == right[0] or _cross(a, b, c) or _cross(a, b, d):
                    continue
                for point in {a, b, c, d}:
                    if _on(point, a, b) and _on(point, c, d):
                        for key, start, finish in ((left, a, b), (right, c, d)):
                            delta = finish[0] - start[0], finish[1] - start[1]
                            parameter = Fraction(
                                (point[0] - start[0]) * delta[0]
                                + (point[1] - start[1]) * delta[1],
                                delta[0] ** 2 + delta[1] ** 2,
                            )
                            event(key, point, parameter)
    raw_free = {}
    for jid, record in records.items():
        contact_ids = eligible_polygons[jid]
        source_ids = contact_ids | {record["authored_polygon_id"]}
        free = _rings(
            _boolean(
                _region([rings[record["authored_polygon_id"]]]),
                _region([rings[pid] for pid in contact_ids]),
                "not",
            )
        )
        traced = []
        for region in free:
            for index, ring in enumerate(region):
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,)))
                oriented = ring if (area > 0) == (index == 0) else tuple(reversed(ring))
                for a, b in _edges((oriented,)):
                    # Boolean may remove collinear source vertices. Reinsert
                    # registered nodes on this exact segment, then require each
                    # subedge to have its pre-recorded source-edge incidence.
                    delta = b[0] - a[0], b[1] - a[1]
                    nodes = sorted(
                        (
                            point
                            for point, incident in members.items()
                            if any(edge[0] in source_ids for edge in incident)
                            and _on(point, a, b)
                        ),
                        key=lambda point: (
                            (point[0] - a[0]) * delta[0] + (point[1] - a[1]) * delta[1]
                        ),
                    )
                    if not nodes or nodes[0] != a or nodes[-1] != b:
                        raise ValueError(
                            "numerical junction Boolean boundary has untraceable source incidence"
                        )
                    for start, finish in pairwise(nodes):
                        source = {
                            edge
                            for edge in members[start] & members[finish]
                            if edge[0] in source_ids
                        }
                        if not source:
                            raise ValueError(
                                "numerical junction Boolean boundary has untraceable source incidence"
                            )
                        traced.append((start, finish, tuple(sorted(source))))
        raw_free[jid] = traced

    def refine(traced):
        result = []
        for a, b, sources in traced:
            split = {Fraction(0): a, Fraction(1): b}
            for source in sources:
                ta, tb = events[source][a], events[source][b]
                if ta == tb:
                    raise ValueError(
                        "numerical junction source edge has collapsed parameter span"
                    )
                for point, t in events[source].items():
                    u = (t - ta) / (tb - ta)
                    if 0 < u < 1:
                        if u in split and split[u] != point:
                            raise ValueError(
                                "numerical junction source incidences disagree"
                            )
                        split[u] = point
            ordered = [point for _, point in sorted(split.items())]
            result.extend((start, end, sources) for start, end in pairwise(ordered))
        return result

    sliced, free_edges = {}, {}
    for jid, record in records.items():
        dx, dy = (Fraction(str(v)) for v in record["direction_xy"])
        origin = Fraction(str(record["local_origin_projection_um"][0])) / Fraction(
            str(PRECISION_UM)
        )
        cuts = [origin + round(v / PRECISION_UM) for v in record["actual_cuts_um"]]

        def project(p, dx=dx, dy=dy):
            return p[0] * dx + p[1] * dy

        cap = [set(), set()]
        parts = [[], [], []]
        divided = []
        for a, b, sources in refine(raw_free[jid]):
            ua, ub = project(a), project(b)
            split = {Fraction(0): a, Fraction(1): b}
            for k, cut in enumerate(cuts):
                if ua == ub == cut:
                    cap[k].update((a, b))
                elif ua != ub and min(ua, ub) <= cut <= max(ua, ub):
                    u = (cut - ua) / (ub - ua)
                    point = tuple(a[i] + u * (b[i] - a[i]) for i in (0, 1))
                    split[u] = point
                    cap[k].add(point)
                    for source in sources:
                        parameter = events[source][a] + u * (
                            events[source][b] - events[source][a]
                        )
                        event(source, point, parameter)
            ordered = [point for _, point in sorted(split.items())]
            for start, end in pairwise(ordered):
                midpoint = (project(start) + project(end)) / 2
                which = 0 if midpoint < cuts[0] else 2 if midpoint > cuts[1] else 1
                # Source boundaries at an empty residual end belong to R alone.
                if midpoint == cuts[1]:
                    which = 1
                edge = (start, end, sources)
                parts[which].append(edge)
                divided.append(edge)
        for k, points in enumerate(cap):
            ordered = sorted(points, key=lambda p: -p[0] * dy + p[1] * dx)
            if len(ordered) < 2:
                raise ValueError(
                    "numerical junction selected cut has no full source section"
                )
            # Collinear existing arm vertices can split a direct-contact edge.
            if any(_cross(ordered[0], ordered[-1], point) for point in ordered):
                raise ValueError(
                    "numerical junction selected cut has inconsistent section nodes"
                )
            for bottom, top in pairwise(ordered):
                if (bottom, top) not in [(a, b) for a, b, _ in parts[k]] and (
                    top,
                    bottom,
                ) not in [(a, b) for a, b, _ in parts[k + 1]]:
                    parts[k].append((bottom, top, ()))
                    parts[k + 1].append((top, bottom, ()))
        sliced[jid] = parts
        free_edges[jid] = divided
    registry, reverse = {}, {}

    def emit(point):
        if point not in registry:
            result = tuple(float(v * Fraction(str(PRECISION_UM))) for v in point)
            if result in reverse and reverse[result] != point:
                raise ValueError(
                    "numerical junction shared vertices collapse on serialization"
                )
            reverse[result] = point
            registry[point] = result
        return registry[point]

    def lower(traced, *, simplify=False):
        raw = [(a, b) for a, b, _ in traced]
        raw = _cancel_reversed_planar_edges(raw)
        if not raw:
            return []
        loops = _simple_planar_loops_from_edges(raw)
        outer = [
            ring
            for ring in loops
            if not any(
                _inside_ring(ring[0], other) == 1
                for other in loops
                if other is not ring
            )
        ]
        regions = []
        for ring in outer:
            holes = [
                hole
                for hole in loops
                if hole is not ring and _inside_ring(hole[0], ring) == 1
            ]
            if any(
                any(
                    _inside_ring(hole[0], other) == 1
                    for other in holes
                    if other is not hole
                )
                for hole in holes
            ):
                raise ValueError("junction residual has unsupported nested components")
            region = []
            for index, loop in enumerate((ring, *holes)):
                if simplify:
                    loop = tuple(
                        point
                        for n, point in enumerate(loop)
                        if _cross(loop[n - 1], point, loop[(n + 1) % len(loop)]) != 0
                    )
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((loop,)))
                if (area > 0) != (index == 0):
                    loop = tuple(reversed(loop))
                region.append(tuple(emit(point) for point in loop))
            regions.append(tuple(region))
        if sum(len(region) for region in regions) != len(loops):
            raise ValueError("junction residual has point-connected components")
        return regions

    affected_polygons = authored | {
        pid
        for entity in build_input.entities
        if entity.semantic_id in affected_owners
        for pid in entity.polygon_ids
    }
    model = {}
    for pid, polygon in originals.items():
        if pid not in affected_polygons:
            model[pid] = polygon
            continue
        region = []
        for index, ring in enumerate(rings[pid]):
            loop = []
            for n in range(len(ring)):
                key = pid, index, n
                loop.extend(
                    emit(point)
                    for point, _ in sorted(
                        events[key].items(), key=lambda item: item[1]
                    )[:-1]
                )
            region.append(tuple(loop))
        model[pid] = replace(polygon, exterior=region[0], holes=tuple(region[1:]))
    additions, appended = [], defaultdict(list)
    physical_ends = defaultdict(list)
    for jid, record in records.items():
        parts = [lower(refine(part)) for part in sliced[jid]]
        central = lower(refine(sliced[jid][1]), simplify=True)
        if len(central) != 1 or len(central[0]) != 1 or len(central[0][0]) != 4:
            raise ValueError(
                "numerical junction final central remainder is not a four-corner rectangle"
            )
        # Native expects the authored A->B corner ordering.
        dx, dy = record["direction_xy"]
        uv = [(x * dx + y * dy, -x * dy + y * dx) for x, y in central[0][0]]
        amin, amax = min(u for u, _ in uv), max(u for u, _ in uv)
        vmin, vmax = min(v for _, v in uv), max(v for _, v in uv)
        corners = tuple(
            min(
                zip(central[0][0], uv),
                key=lambda item: (item[1][0] - u) ** 2 + (item[1][1] - v) ** 2,
            )[0]
            for u, v in ((amin, vmin), (amax, vmin), (amax, vmax), (amin, vmax))
        )
        sheet = originals[record["authored_polygon_id"]]
        additions.append(
            LayoutPolygonSpec(
                record["central_polygon_id"],
                sheet.layer,
                corners,
                metadata={
                    output_format.junction_metadata_key: jid,
                    "part": "central_rlc",
                },
            )
        )
        for n, part in ((0, parts[0]), (1, parts[2])):
            end = record["ends"][n]
            if len(part) > 1:
                raise ValueError(
                    "unsupported junction end topology in final model coordinates"
                )
            if bool(part) != (end["polygon_id"] is not None):
                raise ValueError(
                    "numerical junction zero-end residual differs from selected local proof"
                )
            if part:
                additions.append(
                    LayoutPolygonSpec(
                        end["polygon_id"],
                        sheet.layer,
                        part[0][0],
                        tuple(part[0][1:]),
                        net_name=end["net_id"],
                        metadata={
                            output_format.junction_metadata_key: jid,
                            "source_entity_id": end["source_entity_id"],
                        },
                    )
                )
                appended[end["source_entity_id"]].append(end["polygon_id"])
                physical_ends[end["source_entity_id"]].extend(
                    refine(sliced[jid][0 if n == 0 else 2])
                )
            end["model_regions"] = _plain(part)
        final_free = lower(refine(free_edges[jid]))
        if not _emitted_partition_coverage(
            final_free, [*parts[0], (corners,), *parts[2]]
        ):
            raise ValueError(
                "numerical junction emitted partition does not cover source free boundary"
            )
        record["free_model_regions"] = _plain(final_free)
        record["central_model_regions"] = [[list(corners)]]
        record["proof_conditions"]["emitted_partition_coverage"] = True
        record["model_coordinate_method"] = "source_edge_incidence_and_two_cuts.v1"
        record["model_boolean_precision_um"] = PRECISION_UM
        record["maximum_model_source_coordinate_deviation_um"] = max(
            abs(round(v / PRECISION_UM) * PRECISION_UM - v)
            for pid in needed
            for ring in (originals[pid].exterior, *originals[pid].holes)
            for point in ring
            for v in point
        )
        record["maximum_model_intersection_coordinate_deviation_um"] = (
            intersection_deviation
        )
        record["authored_source_geometry"] = {
            "exterior": _plain(sheet.exterior),
            "holes": _plain(sheet.holes),
        }
        record["model_cut_projection_ranges_um"] = [
            [
                min(x * dx + y * dy for x, y in edge),
                max(x * dx + y * dy for x, y in edge),
            ]
            for edge in ((corners[3], corners[0]), (corners[1], corners[2]))
        ]
    model.update({p.polygon_id: p for p in additions})
    entities = []
    for entity in build_input.entities:
        ids = tuple(pid for pid in entity.polygon_ids if pid not in authored)
        if not ids and entity.polygon_ids:
            continue
        if entity.material_kind != "conductor":
            entities.append(entity)
            continue
        if entity.semantic_id not in affected_owners and ids == entity.polygon_ids:
            entities.append(entity)
            continue
        # Build support from the actual separate pieces, retaining all boundary
        # subdivisions. No second Boolean, clipping or world regridding occurs.
        emitted_support = []
        for pid in ids:
            for index, ring in enumerate((model[pid].exterior, *model[pid].holes)):
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,)))
                oriented = ring if (area > 0) == (index == 0) else tuple(reversed(ring))
                emitted_support.extend(_edges((oriented,)))
        emitted_support.extend(
            (emit(a), emit(b)) for a, b, _ in physical_ends[entity.semantic_id]
        )
        boundary = _cancel_reversed_planar_edges(emitted_support)
        loops = _simple_planar_loops_from_edges(boundary)
        outer = [
            ring
            for ring in loops
            if not any(
                _inside_ring(
                    tuple(Fraction(str(v)) for v in ring[0]),
                    tuple(tuple(Fraction(str(v)) for v in p) for p in other),
                )
                == 1
                for other in loops
                if other is not ring
            )
        ]
        if len(outer) != 1:
            raise ValueError(
                "junction source Entity support cannot be represented as one connected normalized region"
            )
        geometry = {
            **entity.geometry,
            "outer_loop": outer[0],
            "hole_loops": tuple(ring for ring in loops if ring is not outer[0]),
        }
        entities.append(
            replace(
                entity,
                polygon_ids=(*ids, *appended[entity.semantic_id]),
                geometry=geometry,
            )
        )
    # Check exactly emitted coordinates, independently of local selection.
    actual_rings = {
        pid: tuple(
            tuple(tuple(Fraction(str(v)) for v in point) for point in ring)
            for ring in (polygon.exterior, *polygon.holes)
        )
        for pid, polygon in model.items()
    }
    for record in records.values():
        conductors = [
            (e.net_id, pid, actual_rings[pid])
            for e in entities
            if e.material_kind == "conductor"
            and e.semantic_id in record["support_plane"]["eligible_entity_ids"]
            for pid in e.polygon_ids
        ]
        c = actual_rings[record["central_polygon_id"]][0]
        midpoint = tuple(sum(point[i] for point in c) / 4 for i in (0, 1))

        def strictly_inside(point, region):
            return _inside_ring(point, region[0]) == 1 and all(
                _inside_ring(point, hole) == -1 for hole in region[1:]
            )

        for _, _, region in conductors:
            if (
                any(strictly_inside(point, (c,)) for point in region[0])
                or any(strictly_inside(point, region) for point in (*c, midpoint))
                or any(
                    _cross(a, b, u) * _cross(a, b, v) < 0
                    and _cross(u, v, a) * _cross(u, v, b) < 0
                    for a, b in _edges(region)
                    for u, v in _edges((c,))
                )
            ):
                raise ValueError("final cross-junction PEC intersects RLC interior")
        for end in record["ends"]:
            own = [region for net, _, region in conductors if net == end["net_id"]]
            other = [region for net, _, region in conductors if net != end["net_id"]]
            if end["polygon_id"]:
                piece = actual_rings[end["polygon_id"]]
                arms = [
                    actual_rings[pid]
                    for e in entities
                    if e.semantic_id == end["source_entity_id"]
                    for pid in e.polygon_ids
                    if pid
                    not in {
                        part["polygon_id"]
                        for r in records.values()
                        for part in r["ends"]
                    }
                ]
                if not _fraction_boundary_contact([piece], arms):
                    raise ValueError(
                        "final model end lacks positive-length source-arm contact"
                    )
                if _closed_contact([piece], other):
                    raise ValueError("final cross-junction end contacts another net")
            n = 0 if end["side"] == "A" else 1
            start, finish = (c[3], c[0]) if n == 0 else (c[1], c[2])
            if not _model_edge_covered(own, start, finish):
                raise ValueError(
                    "final model designated end edge is not wholly covered"
                )
            if _model_edge_contacts(other, start, finish):
                raise ValueError("final cross-junction wrong-net end/corner contact")
        for start, finish in ((c[0], c[1]), (c[2], c[3])):
            for net, _, region in conductors:
                intervals, points = _model_edge_section(region, start, finish)
                if intervals or any(0 < point < 1 for point in points):
                    raise ValueError("final cross-junction open-side contact")
        record["proof_conditions"]["all_derived_junctions_checked"] = True
    return model, tuple(entities)


def _fraction_boundary_contact(left, right):
    def lines(regions):
        result = defaultdict(list)
        for region in regions:
            for a, b in _edges(region):
                if a[0] != b[0]:
                    slope = (b[1] - a[1]) / (b[0] - a[0])
                    key, axis = ("x", slope, a[1] - slope * a[0]), 0
                else:
                    key, axis = ("y", a[0]), 1
                result[key].append((min(a[axis], b[axis]), max(a[axis], b[axis])))
        return result

    a, b = lines(left), lines(right)
    return any(
        max(x0, y0) < min(x1, y1)
        for key in a.keys() & b.keys()
        for x0, x1 in a[key]
        for y0, y1 in b[key]
    )


def _emitted_partition_coverage(free, parts):
    """Compare actual emitted boundaries, including central simplification."""

    def rational(regions):
        return [
            tuple(
                tuple(tuple(Fraction(str(v)) for v in point) for point in ring)
                for ring in region
            )
            for region in regions
        ]

    free, parts = rational(free), rational(parts)
    points = sorted(
        {point for region in (*free, *parts) for ring in region for point in ring}
    )
    xs = [point[0] for point in points]

    def edges(regions):
        result = []
        for region in regions:
            for a, b in _edges(region):
                candidates = points[
                    bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
                ]
                ordered = sorted(
                    (point for point in candidates if _on(point, a, b)),
                    key=lambda point: (
                        (point[0] - a[0]) * (b[0] - a[0])
                        + (point[1] - a[1]) * (b[1] - a[1])
                    ),
                )
                result.extend(pairwise(ordered))
        return _cancel_reversed_planar_edges(result)

    return sorted(edges(free)) == sorted(edges(parts))


def _model_edge_coordinates(region, start, finish):
    dx, dy = finish[0] - start[0], finish[1] - start[1]
    norm = dx * dx + dy * dy
    return tuple(
        tuple(
            (
                ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / norm,
                dx * (p[1] - start[1]) - dy * (p[0] - start[0]),
            )
            for p in ring
        )
        for ring in region
    )


def _model_edge_section(region, start, finish):
    return _section(_model_edge_coordinates(region, start, finish), 1, 0, 0, 1)


def _model_edge_covered(regions, start, finish):
    return _covered(
        [_model_edge_coordinates(region, start, finish) for region in regions],
        1,
        0,
        0,
        1,
    )


def _model_edge_contacts(regions, start, finish):
    return any(any(_model_edge_section(region, start, finish)) for region in regions)


@with_input_grid
def partition_junctions(
    build_input: GeometryBuildInput,
    junctions: Sequence[JunctionGeometry],
    step: float,
    *,
    output_format: JunctionPartitionFormat = DEFAULT_FORMAT,
):
    """Derive terminal-owned ends and a complete central rectangle once."""
    if not junctions:
        return build_input
    if output_format.metadata_key in build_input.metadata:
        raise ValueError(
            "junction input is already partitioned; use immutable authored input"
        )
    originals = {p.polygon_id: p for p in build_input.polygons}
    originals.update(
        {
            r.source_polygon_id: LayoutPolygonSpec(
                r.source_polygon_id, r.source_layer, r.exterior, r.holes
            )
            for r in build_input.port_sheet_regions
        }
    )
    authored_ids = {j.source_polygon_id for j in junctions}
    entities = {e.semantic_id: e for e in build_input.entities}
    records = {}
    for junction in junctions:
        support_plane = _junction_contact_inventory(
            build_input,
            junction.source_polygon_id,
            junction.terminal_a_net,
            junction.terminal_b_net,
            junction.metadata.get("junction_support_plane"),
            support_schema=output_format.support_schema,
        )
        sheet = originals[junction.source_polygon_id]
        dx, dy = junction.direction_xy
        px, py = -dy, dx
        along = [x * dx + y * dy for x, y in sheet.exterior]
        across = [x * px + y * py for x, y in sheet.exterior]
        origin = (min(along), min(across))
        span = round((max(along) - origin[0]) / PRECISION_UM)
        width = round((max(across) - origin[1]) / PRECISION_UM)
        if span <= 0 or width <= 0:
            raise ValueError(
                "junction authored geometry collapses at Boolean precision"
            )
        deviations = []

        def local(
            polygon, *, dx=dx, dy=dy, px=px, py=py, origin=origin, deviations=deviations
        ):
            rings = []
            for ring in (polygon.exterior, *polygon.holes):
                converted = []
                for x, y in ring:
                    u, v = x * dx + y * dy - origin[0], x * px + y * py - origin[1]
                    point = (round(u / PRECISION_UM), round(v / PRECISION_UM))
                    deviations.append(
                        max(
                            abs(point[0] * PRECISION_UM - u),
                            abs(point[1] * PRECISION_UM - v),
                        )
                    )
                    converted.append(point)
                rings.append(tuple(converted))
            for index, ring in enumerate(rings):
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,)))
                if len(set(ring)) < 3 or area == 0:
                    raise ValueError(
                        "numerical junction geometry failure: source polygon/hole collapses at Boolean precision"
                    )
                if index and any(_inside_ring(point, rings[0]) != 1 for point in ring):
                    raise ValueError(
                        "numerical junction geometry failure: source hole loses interior topology"
                    )
            return tuple(rings)

        transform = (dx, px, dy, py,
                     origin[0]*dx + origin[1]*px,
                     origin[0]*dy + origin[1]*py)
        with polygon_world_frame(transform):
            by_entity = {
                e.semantic_id: [
                    local(originals[p]) for p in e.polygon_ids if p not in authored_ids
                ]
                for e in entities.values()
                if e.semantic_id in support_plane["eligible_entity_ids"]
            }
            by_net = defaultdict(list)
            for eid, polygons in by_entity.items():
                by_net[entities[eid].net_id].extend(polygons)
            own_a, own_b = by_net[junction.terminal_a_net], by_net[junction.terminal_b_net]
            if not own_a or not own_b:
                raise ValueError(
                    "junction requires both explicit terminal conductor geometries"
                )
            other = [
                p
                for net, polygons in by_net.items()
                if net not in {junction.terminal_a_net, junction.terminal_b_net}
                for p in polygons
            ]
            if _contacts(other, span, width) is not None:
                raise ValueError("third-net closed contact with authored junction sheet")
            ca, cb = _contacts(own_a, span, width), _contacts(own_b, span, width)
            if ca is None or cb is None or ca[1] >= cb[0]:
                raise ValueError("unsupported junction contact ordering or clear corridor")
            lower, upper = ca[1], cb[0]
            for polygons in (own_a, own_b, other):
                for rings in polygons:
                    for y in (0, width):
                        intervals, points = _section(rings, 1, y, lower, upper)
                        if any(a < b for a, b in intervals) or any(
                            lower < v < upper for v in points
                        ):
                            raise ValueError(
                                "unsupported corridor side-boundary metal contact"
                            )
            sheet_region = _region(_rectangle(0, 0, span, width))
            free = _boolean(sheet_region, _region([*own_a, *own_b]), "not")
            corridor = _region(_rectangle(lower, 0, upper, width))
            if _boolean(corridor, free, "not"):
                raise ValueError(
                    "unsupported corridor containment; no general search performed"
                )
            choices = []
            checks = []
            for left, bound, own, foreign in (
                (True, lower, own_a, own_b + other),
                (False, upper, own_b, own_a + other),
            ):
                chosen = None
                for index in _candidate_indices(bound, step, left=left):
                    requested, cut = _cut(index, step)
                    interior = (
                        (Decimal(index) * Decimal(str(step)) > _bound_decimal(bound))
                        if left
                        else (Decimal(index) * Decimal(str(step)) < _bound_decimal(bound))
                    )
                    if interior and cut == bound:
                        raise ValueError(
                            "numerical junction geometry failure: interior grid cut collapses to boundary"
                        )
                    if (
                        cut < lower
                        or cut > upper
                        or (interior and (cut <= lower or cut >= upper))
                    ):
                        continue
                    end, evidence = _end_candidate(
                        free, own, foreign, cut, span, width, left=left
                    )
                    checks.append(
                        {
                            "side": "A" if left else "B",
                            "index": index,
                            "requested_s_um": requested,
                            "actual_s_um": cut * PRECISION_UM,
                            "deviation_um": cut * PRECISION_UM - requested,
                            **evidence,
                        }
                    )
                    if end is not None:
                        chosen = (cut, end, index)
                        break
                if chosen is None:
                    raise ValueError(
                        "unsupported junction end topology on verified corridor/grid"
                    )
                choices.append(chosen)
            (a, end_a, ka), (b, end_b, kb) = choices
            if a >= b:
                raise ValueError("junction DBU grid has no positive central span")
            central = _rectangle(a, 0, b, width)
            for y in (0, width):
                for rings in [*own_a, *own_b, *other, *end_a, *end_b]:
                    intervals, points = _section(rings, 1, y, a, b)
                    if any(x < z for x, z in intervals) or any(a < v < b for v in points):
                        raise ValueError(
                            "RLC open side has forbidden line or point contact"
                        )
            for cut, wrong in ((a, own_b + other + end_b), (b, own_a + other + end_a)):
                if any(any(_section(rings, 0, cut, 0, width)) for rings in wrong):
                    raise ValueError("wrong-net RLC end/corner contact")
            parts = [_region(end_a), _region(central), _region(end_b)]
            union = _boolean([p for part in parts for p in part], [], "or")
            if _boolean(union, free, "xor") or any(
                _boolean(parts[i], parts[j], "and") for i, j in ((0, 1), (0, 2), (1, 2))
            ):
                raise ValueError(
                    "numerical junction geometry failure: partition coverage/interiors differ"
                )

            end_records = []
            for label, end, own, cut, net in (
                ("A", end_a, own_a, a, junction.terminal_a_net),
                ("B", end_b, own_b, b, junction.terminal_b_net),
            ):
                owners = [
                    eid
                    for eid, polygons in by_entity.items()
                    if entities[eid].net_id == net
                    and (
                        _positive_boundary_contact(end, polygons)
                        if end
                        else _covered(polygons, 0, cut, 0, width)
                    )
                ]
                if len(owners) != 1:
                    raise ValueError(
                        "junction end lacks an unambiguous source Entity/material/Z lineage"
                    )
                owner = entities[owners[0]]
                polygon_id = (
                    f"{output_format.id_prefix}_{canonical_sha256([junction.junction_id, label, owner.semantic_id])[:24]}"
                    if end
                    else None
                )
                end_records.append(
                    {
                        "side": label,
                        "source_entity_id": owner.semantic_id,
                        "net_id": net,
                        "material_id": owner.material_id,
                        "geometry": _plain(owner.geometry),
                        "representation": dict(owner.route_representations),
                        "polygon_id": polygon_id,
                        "empty_residual": not end,
                        "local_regions": _plain(end),
                    }
                )
            cid = f"{output_format.id_prefix}_{canonical_sha256([junction.junction_id, 'central'])[:24]}"
            records[junction.junction_id] = {
                "method": output_format.method,
                "support_plane": support_plane,
                "z_um": support_plane["effective_z_um"],
                "authored_polygon_id": junction.source_polygon_id,
                "central_polygon_id": cid,
                "direction_xy": [dx, dy],
                "local_origin_projection_um": list(origin),
                "source_dbu_um": step,
                "boolean_precision_um": PRECISION_UM,
                "span_um": span * PRECISION_UM,
                "width_um": width * PRECISION_UM,
                "corridor_um": [lower * PRECISION_UM, upper * PRECISION_UM],
                "cut_indices": [ka, kb],
                "actual_cuts_um": [a * PRECISION_UM, b * PRECISION_UM],
                "candidate_checks": checks,
                "candidate_count": len(checks),
                "proof_conditions": {
                    "full_width_corridor_containment": True,
                    "closed_corridor_side_clearance": True,
                    "terminal_contact_ordering": "A_left_B_right",
                    "partition_coverage_and_disjoint_interiors": True,
                },
                "maximum_input_coordinate_deviation_um": max(deviations, default=0.0),
                "ends": end_records,
                "free_local_regions": _plain(_rings(free)),
                "central_local_regions": _plain(central),
            }
    model, derived_entities = _model_geometry(
        build_input, originals, records, output_format
    )
    metadata = dict(build_input.metadata)
    metadata[output_format.source_hash_key] = canonical_sha256(_plain(build_input))
    metadata[output_format.metadata_key] = records
    original_ids = {p.polygon_id for p in build_input.polygons}
    derived_polygons = tuple(model[p.polygon_id] for p in build_input.polygons) + tuple(
        p
        for pid, p in model.items()
        if pid not in original_ids and pid not in authored_ids
    )
    regions = tuple(
        replace(
            r,
            exterior=model[r.source_polygon_id].exterior,
            holes=model[r.source_polygon_id].holes,
        )
        if r.source_polygon_id in model
        else r
        for r in build_input.port_sheet_regions
    )
    result = replace(
        build_input,
        polygons=derived_polygons,
        entities=derived_entities,
        port_sheet_regions=regions,
        metadata=metadata,
    )
    context = geometry_grid_context(build_input)
    if context is not None:
        from scgsim.geometry.source.grid import apply_geometry_grid
        result = apply_geometry_grid(result, context)
        actual = {p.polygon_id: p for p in result.polygons}
        for junction in junctions:
            record = records[junction.junction_id]
            polygon = actual[record["central_polygon_id"]]
            try:
                junction_terminal_line({"exterior": polygon.exterior, "holes": polygon.holes},
                                       junction, record["z_um"])
                ends = [actual[end["polygon_id"]] for end in record["ends"] if end["polygon_id"]]
                emitted = [(p.exterior, *p.holes) for p in ends]
                emitted.append((polygon.exterior, *polygon.holes))
                if not _emitted_partition_coverage(record["free_model_regions"], emitted):
                    raise ValueError("junction grid realization changes partition coverage")
            except ValueError as error:
                error.add_note(json.dumps({"junction_id": junction.junction_id,
                    "geometry_grid": context.to_payload(),
                    "requested_direction_xy": list(junction.direction_xy),
                    "requested_width_um": junction.width_um,
                    "realization_status": "incompatible_with_existing_shape_contract"}, sort_keys=True))
                raise
            record["geometry_grid"] = context.to_payload()
            record["requested_central_model_regions"] = record["central_model_regions"]
            record["central_model_regions"] = [[list(polygon.exterior), *map(list, polygon.holes)]]
            record["realized_central_geometry_sha256"] = canonical_sha256(
                {"exterior": polygon.exterior, "holes": polygon.holes})
            for end in record["ends"]:
                if end["polygon_id"]:
                    piece = actual[end["polygon_id"]]
                    end["requested_model_regions"] = end["model_regions"]
                    end["model_regions"] = [[list(piece.exterior), *map(list, piece.holes)]]
    return result


def junction_terminal_line(
    polygon: Mapping[str, Any], junction: JunctionGeometry, z_um: float
) -> tuple[list[list[float]], float]:
    exterior = tuple(
        (float(point[0]), float(point[1])) for point in polygon["exterior"]
    )
    # A common arrangement retains contact nodes on rectangle edges. Use an
    # exact, non-mutating corner witness for shape validation; the physical
    # polygon and all its shared subdivisions remain unchanged.
    rational = lambda p: tuple(Fraction(str(v)) for v in p)
    witness = tuple(point for index, point in enumerate(exterior)
                    if point != exterior[index-1])
    exterior = tuple(point for index, point in enumerate(witness)
                     if _cross(rational(witness[index-1]), rational(point),
                               rational(witness[(index+1) % len(witness)])) != 0)
    if len(exterior) != 4 or polygon.get("holes"):
        raise ValueError(
            f"junction {junction.junction_id!r} requires one rectangular polygon without holes"
        )
    dx, dy = junction.direction_xy
    px, py = -dy, dx
    along = tuple(x * dx + y * dy for x, y in exterior)
    across = tuple(x * px + y * py for x, y in exterior)
    along_min, along_max = min(along), max(along)
    across_min, across_max = min(across), max(across)
    tolerance = 1e-9 * max(
        1.0, abs(along_min), abs(along_max), abs(across_min), abs(across_max)
    )
    if along_max - along_min <= tolerance or across_max - across_min <= tolerance:
        raise ValueError(
            f"junction {junction.junction_id!r} polygon has empty projected extent"
        )
    expected = {
        (along_min, across_min),
        (along_min, across_max),
        (along_max, across_min),
        (along_max, across_max),
    }
    if any(
        not any(
            abs(u - eu) <= tolerance and abs(v - ev) <= tolerance for eu, ev in expected
        )
        for u, v in zip(along, across)
    ):
        raise ValueError(
            f"junction {junction.junction_id!r} polygon is not rectangular in its declared direction"
        )
    width = across_max - across_min
    if not math.isclose(width, junction.width_um, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"junction {junction.junction_id!r} width disagrees with its polygon"
        )
    across_mid = (across_min + across_max) / 2.0
    line = [
        [along_min * dx + across_mid * px, along_min * dy + across_mid * py, z_um],
        [along_max * dx + across_mid * px, along_max * dy + across_mid * py, z_um],
    ]
    return line, along_max - along_min


@with_input_grid
def derive_rectangular_junctions(build_input):
    """Derive once after the single physical interpretation, before any surfaces."""
    requests = tuple(
        RectangularJunctionSpec.from_payload(item)
        for item in build_input.metadata.get("rectangular_junctions", ())
    )
    if not requests:
        return build_input
    regions = {item.port_sheet_id: item for item in build_input.port_sheet_regions}
    geometries = []
    for request in requests:
        region = regions[request.port_sheet_id]
        dx, dy = region.metadata["direction"][:2]
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        across = [-dy * x + dx * y for x, y in region.exterior]
        geometry = JunctionGeometry(
            request.junction_id,
            request.source_polygon_id,
            request.terminal_a.net_id,
            request.terminal_b.net_id,
            (dx, dy),
            max(across) - min(across),
        )
        try:
            junction_terminal_line(
                {"exterior": region.exterior, "holes": region.holes}, geometry, 0.0
            )
        except ValueError as error:
            context = geometry_grid_context(build_input)
            if context is not None:
                error.add_note(json.dumps({"junction_id": request.junction_id,
                    "geometry_grid": context.to_payload(),
                    "requested_direction_xy": list(geometry.direction_xy),
                    "realization_status": "incompatible_with_existing_shape_contract"}, sort_keys=True))
            raise
        geometries.append(geometry)
    result = partition_junctions(
        build_input, geometries, build_input.metadata["source_dbu_um"]
    )
    records = result.metadata["junction_partitions"]
    polygons = {item.polygon_id: item for item in result.polygons}
    regions = []
    by_port = {item.port_sheet_id: item for item in requests}
    for region in result.port_sheet_regions:
        request = by_port.get(region.port_sheet_id)
        if request is None:
            regions.append(region)
            continue
        record = records[request.junction_id]
        central = polygons[record["central_polygon_id"]]
        regions.append(
            replace(
                region,
                exterior=central.exterior,
                holes=central.holes,
                metadata={
                    **region.metadata,
                    "rectangular_junction_partition": request.junction_id,
                    "rectangular_junction_binding": request.to_payload(),
                    "rectangular_junction_source_geometry": record[
                        "authored_source_geometry"
                    ],
                },
            )
        )
    owner_ids = {
        terminal.source_entity_id
        for item in requests
        for terminal in (item.terminal_a, item.terminal_b)
    }
    entities = tuple(
        replace(
            entity,
            metadata={
                **entity.metadata,
                "junction_derived_support": tuple(
                    item.junction_id
                    for item in requests
                    if entity.semantic_id
                    in {
                        item.terminal_a.source_entity_id,
                        item.terminal_b.source_entity_id,
                    }
                ),
            },
        )
        if entity.semantic_id in owner_ids
        else entity
        for entity in result.entities
    )
    return replace(result, entities=entities, port_sheet_regions=tuple(regions))

