"""Plan source incidence and exact two-cut slicing."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import pairwise
from typing import cast

from scgsim.geometry.planning.junction_planning.boolean_geometry import (
    boolean,
    build_polygon_region,
    read_boolean_rings,
)
from scgsim.geometry.planning.junction_planning.core import PRECISION_UM
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    cross,
    inside_ring,
    on,
    region_edges,
)
from scgsim.geometry.planning.junction_planning.model_state import (
    IntegerPoint,
    JunctionModel,
)


def initialize_source(state: JunctionModel) -> None:
    """Collect eligible source polygons and initialize exact incidence ledgers."""
    state.authored = {r["authored_polygon_id"] for r in state.records.values()}
    state.eligible_polygons = {
        jid: {
            pid
            for entity in state.build_input.entities
            if entity.semantic_id in record["support_plane"]["eligible_entity_ids"]
            for pid in entity.polygon_ids
            if pid not in state.authored
        }
        for jid, record in state.records.items()
    }
    state.sheet_metals = defaultdict(set)
    for jid, record in state.records.items():
        state.sheet_metals[record["authored_polygon_id"]].update(
            state.eligible_polygons[jid]
        )
    state.metal_ids = set().union(*state.eligible_polygons.values())
    state.needed = state.metal_ids | state.authored
    state.rings = cast(
        dict[str, tuple[tuple[IntegerPoint, ...], ...]],
        {
            pid: tuple(
                (
                    tuple(
                        (
                            tuple((round(v / PRECISION_UM) for v in point))
                            for point in ring
                        )
                    )
                    for ring in (
                        state.originals[pid].exterior,
                        *state.originals[pid].holes,
                    )
                )
            )
            for pid in state.needed
        },
    )
    state.edges, state.events, state.members = ({}, defaultdict(dict), defaultdict(set))
    state.intersection_deviation = 0.0


def register_source_edges(state: JunctionModel) -> None:
    """Register every directed source edge and its endpoint incidences."""
    for pid, region in state.rings.items():
        for index, ring in enumerate(region):
            area = sum((a[0] * b[1] - b[0] * a[1] for a, b in region_edges((ring,))))
            if len(set(ring)) < 3 or not area:
                raise ValueError(
                    "numerical junction source polygon/hole collapses in model coordinates"
                )
            if index and any((inside_ring(point, region[0]) != 1 for point in ring)):
                raise ValueError(
                    "numerical junction source hole loses model interior topology"
                )
            for n, (a, b) in enumerate(region_edges((ring,))):
                key = (pid, index, n)
                state.edges[key] = (a, b)
                state.event(key, a, Fraction(0))
                state.event(key, b, Fraction(1))


def register_crossings(state: JunctionModel) -> None:
    """Register exact arm-sheet intersections before any Boolean topology."""
    for arm, (a, b) in state.edges.items():
        if arm[0] not in state.metal_ids:
            continue
        ab = (b[0] - a[0], b[1] - a[1])
        for sheet, (c, d) in state.edges.items():
            if arm[0] not in state.sheet_metals.get(sheet[0], ()):
                continue
            if (
                max(a[0], b[0]) < min(c[0], d[0])
                or max(c[0], d[0]) < min(a[0], b[0])
                or max(a[1], b[1]) < min(c[1], d[1])
                or (max(c[1], d[1]) < min(a[1], b[1]))
            ):
                continue
            cd = (d[0] - c[0], d[1] - c[1])
            determinant = ab[0] * cd[1] - ab[1] * cd[0]
            if determinant:
                ac = (c[0] - a[0], c[1] - a[1])
                t = Fraction(ac[0] * cd[1] - ac[1] * cd[0], determinant)
                u = Fraction(ac[0] * ab[1] - ac[1] * ab[0], determinant)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    exact = tuple((a[i] + t * ab[i] for i in (0, 1)))
                    point = tuple((round(v) for v in exact))
                    state.intersection_deviation = max(
                        state.intersection_deviation,
                        max((float(abs(point[i] - exact[i])) for i in (0, 1)))
                        * PRECISION_UM,
                    )
                    state.event(arm, point, t)
                    state.event(sheet, point, u)
            elif cross(a, b, c) == 0:
                for point in {a, b, c, d}:
                    if on(point, a, b) and on(point, c, d):
                        for key, start, delta in ((arm, a, ab), (sheet, c, cd)):
                            norm = delta[0] ** 2 + delta[1] ** 2
                            parameter = Fraction(
                                (point[0] - start[0]) * delta[0]
                                + (point[1] - start[1]) * delta[1],
                                norm,
                            )
                            state.event(key, point, parameter)


def register_owned_seams(state: JunctionModel) -> None:
    """Register existing endpoint nodes on seams of the same source Entity."""
    state.affected_owners = {
        end["source_entity_id"]
        for record in state.records.values()
        for end in record["ends"]
    }
    for entity in state.build_input.entities:
        if entity.semantic_id not in state.affected_owners:
            continue
        owned = set(entity.polygon_ids) - state.authored
        owned_edges = [
            (key, a, b) for key, (a, b) in state.edges.items() if key[0] in owned
        ]
        for n, (left, a, b) in enumerate(owned_edges):
            for right, c, d in owned_edges[n + 1 :]:
                if left[0] == right[0] or cross(a, b, c) or cross(a, b, d):
                    continue
                for point in {a, b, c, d}:
                    if on(point, a, b) and on(point, c, d):
                        for key, start, finish in ((left, a, b), (right, c, d)):
                            delta = (finish[0] - start[0], finish[1] - start[1])
                            parameter = Fraction(
                                (point[0] - start[0]) * delta[0]
                                + (point[1] - start[1]) * delta[1],
                                delta[0] ** 2 + delta[1] ** 2,
                            )
                            state.event(key, point, parameter)


def trace_free_boundaries(state: JunctionModel) -> None:
    """Trace native Boolean free boundaries back to registered source edges."""
    state.raw_free = {}
    for jid, record in state.records.items():
        contact_ids = state.eligible_polygons[jid]
        source_ids = contact_ids | {record["authored_polygon_id"]}
        free = read_boolean_rings(
            boolean(
                build_polygon_region([state.rings[record["authored_polygon_id"]]]),
                build_polygon_region([state.rings[pid] for pid in contact_ids]),
                "not",
            )
        )
        traced = []
        for region in free:
            for index, ring in enumerate(region):
                area = sum(
                    (a[0] * b[1] - b[0] * a[1] for a, b in region_edges((ring,)))
                )
                oriented = ring if (area > 0) == (index == 0) else tuple(reversed(ring))
                for a, b in region_edges((oriented,)):
                    delta = (b[0] - a[0], b[1] - a[1])
                    nodes = sorted(
                        (
                            point
                            for point, incident in state.members.items()
                            if any((edge[0] in source_ids for edge in incident))
                            and on(point, a, b)
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
                            for edge in state.members[start] & state.members[finish]
                            if edge[0] in source_ids
                        }
                        if not source:
                            raise ValueError(
                                "numerical junction Boolean boundary has untraceable source incidence"
                            )
                        traced.append((start, finish, tuple(sorted(source))))
        state.raw_free[jid] = traced


def slice_free_boundaries(state: JunctionModel) -> None:
    """Slice the traced free boundaries at the two selected cut planes."""
    state.sliced, state.free_edges = ({}, {})
    for jid, record in state.records.items():
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
        for a, b, sources in state.refine(state.raw_free[jid]):
            ua, ub = (project(a), project(b))
            split = {Fraction(0): a, Fraction(1): b}
            for k, cut in enumerate(cuts):
                if ua == ub == cut:
                    cap[k].update((a, b))
                elif ua != ub and min(ua, ub) <= cut <= max(ua, ub):
                    u = (cut - ua) / (ub - ua)
                    point = tuple((a[i] + u * (b[i] - a[i]) for i in (0, 1)))
                    split[u] = point
                    cap[k].add(point)
                    for source in sources:
                        parameter = state.events[source][a] + u * (
                            state.events[source][b] - state.events[source][a]
                        )
                        state.event(source, point, parameter)
            ordered = [point for _, point in sorted(split.items())]
            for start, end in pairwise(ordered):
                midpoint = (project(start) + project(end)) / 2
                which = 0 if midpoint < cuts[0] else 2 if midpoint > cuts[1] else 1
                if midpoint == cuts[1]:
                    which = 1
                edge = (start, end, sources)
                parts[which].append(edge)
                divided.append(edge)
        _close_cut_caps(cap, parts, dx, dy)
        state.sliced[jid] = parts
        state.free_edges[jid] = divided


def _close_cut_caps(cap, parts, dx, dy) -> None:
    """Close adjacent cut pieces using their registered exact section nodes."""
    for k, points in enumerate(cap):
        ordered = sorted(points, key=lambda p: -p[0] * dy + p[1] * dx)
        if len(ordered) < 2:
            raise ValueError(
                "numerical junction selected cut has no full source section"
            )
        if any((cross(ordered[0], ordered[-1], point) for point in ordered)):
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
