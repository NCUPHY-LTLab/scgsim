"""Map a junction partition proof back into the model's own coordinates.

`junction_partition.py` proves a three-way split (end A, central RLC rectangle, end B)
of each junction sheet in a local integer frame. This module reproduces that split on
the real model polygons: it traces every source edge incidence before any Boolean
topology exists, cuts the free sheet at the two chosen cut lines, rebuilds the
affected polygons and Entities with the emitted vertices, and hands the result to
`junction_checks.py`, which re-checks the emitted coordinates independently of the
local proof.

Owns: model-coordinate lowering. Does not own: choosing cuts or the local proof
(`junction_partition.py`), exact predicates (`junction_exact.py`) or the final
checks (`junction_checks.py`).

Vocabulary used below. A "source edge" is an edge of an authored polygon, keyed
`(polygon id, ring index, edge index)`. A "parameter" is the exact position
`0..1` of a point along its source edge. An "arm" is a conductor polygon touching a
junction sheet; the "sheet" is the authored junction polygon. "Model coordinates"
are the integer Boolean-grid coordinates of the whole layout; "emitted" coordinates
are the floats finally written to the model polygons.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise

from scgsim.aedt.preparation.junction_checks import (
    FinalModel,
    emitted_partition_coverage,
)
from scgsim.aedt.preparation.junction_exact import (
    PRECISION_UM,
    _boolean,
    _cross,
    _edges,
    _inside_ring,
    _on,
    _oriented_ring,
    _plain_with_dataclasses,
    _region,
    _rings,
    _twice_signed_area,
)
from scgsim.geometry import LayoutPolygonSpec
from scgsim.geometry._primitives.loops import (
    _cancel_reversed_planar_edges,
    _simple_planar_loops_from_edges,
)


def model_geometry(build_input, originals, records):
    """Trace source incidences and owned seams before topology, then slice at two cuts.

    Returns the model polygons by id and the rebuilt Entities. Fills the
    model-coordinate fields of every junction record in place.
    """
    return _JunctionModel(build_input, originals, records).build()


class _SourceIncidence:
    """Exact meeting points of source edges, as parameters along each edge."""

    def __init__(self):
        self.events = defaultdict(dict)  # source edge -> {point: parameter}
        self.members = defaultdict(set)  # point -> source edges through it

    def record(self, edge, point, parameter):
        """Note that `edge` passes through `point` at `parameter`."""
        prior = self.events[edge].get(point)
        if prior is not None and prior != parameter:
            raise ValueError(
                "numerical junction incidence collapses distinct source parameters"
            )
        self.events[edge][point] = parameter
        self.members[point].add(edge)

    def refine(self, traced):
        """Split traced `(start, end, source edges)` edges at every recorded incidence."""
        result = []
        for a, b, sources in traced:
            split = {Fraction(0): a, Fraction(1): b}
            for source in sources:
                ta, tb = self.events[source][a], self.events[source][b]
                if ta == tb:
                    raise ValueError(
                        "numerical junction source edge has collapsed parameter span"
                    )
                for point, t in self.events[source].items():
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


class _VertexRegistry:
    """Converts exact points to emitted float vertices, one float per exact point."""

    def __init__(self):
        self._registry = {}
        self._reverse = {}

    def emit(self, point):
        """Return the emitted vertex for `point`, refusing two points that collide."""
        if point not in self._registry:
            result = tuple(float(v * Fraction(str(PRECISION_UM))) for v in point)
            if result in self._reverse and self._reverse[result] != point:
                raise ValueError(
                    "numerical junction shared vertices collapse on serialization"
                )
            self._reverse[result] = point
            self._registry[point] = result
        return self._registry[point]


class _JunctionModel:
    """One lowering of all junction records onto the model polygons."""

    def __init__(self, build_input, originals, records):
        self.build_input = build_input
        self.originals = originals
        self.records = records
        self.authored = {r["authored_polygon_id"] for r in records.values()}
        self.metal_ids = {
            pid
            for e in build_input.entities
            if e.material_kind == "conductor"
            for pid in e.polygon_ids
            if pid not in self.authored
        }
        self.needed = self.metal_ids | self.authored
        self.rings = {
            pid: tuple(
                tuple(tuple(round(v / PRECISION_UM) for v in point) for point in ring)
                for ring in (originals[pid].exterior, *originals[pid].holes)
            )
            for pid in self.needed
        }
        self.incidence = _SourceIncidence()
        self.intersection_deviation = 0.0
        self.affected_owners = {
            end["source_entity_id"]
            for record in records.values()
            for end in record["ends"]
        }
        self.vertices = _VertexRegistry()
        self.sliced = {}
        self.free_edges = {}
        self.additions = []
        self.appended = defaultdict(list)
        self.physical_ends = defaultdict(list)

    def build(self):
        """Run the lowering steps in order; the order decides which error is seen."""
        source_edges = self._register_source_edges()
        self._register_arm_sheet_incidences(source_edges)
        self._register_owned_seams(source_edges)
        raw_free = self._trace_free_boundaries()
        self._slice_free_boundaries(raw_free)
        model = self._rebuild_affected_polygons()
        self._emit_junction_parts()
        model.update({p.polygon_id: p for p in self.additions})
        entities = self._rebuild_entities(model)
        FinalModel(self.records, model, entities).check()
        return model, tuple(entities)

    # --- source incidences ---------------------------------------------------

    def _register_source_edges(self):
        """Record every source edge's two endpoints, rejecting degenerate rings."""
        edges = {}
        for pid, region in self.rings.items():
            for index, ring in enumerate(region):
                area = _twice_signed_area(ring)
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
                    self.incidence.record(key, a, Fraction(0))
                    self.incidence.record(key, b, Fraction(1))
        return edges

    def _register_arm_sheet_incidences(self, source_edges):
        """Record where each arm edge crosses or overlaps each junction sheet edge."""
        for arm, (a, b) in source_edges.items():
            if arm[0] not in self.metal_ids:
                continue
            for sheet, (c, d) in source_edges.items():
                if sheet[0] not in self.authored:
                    continue
                if (
                    max(a[0], b[0]) < min(c[0], d[0])
                    or max(c[0], d[0]) < min(a[0], b[0])
                    or max(a[1], b[1]) < min(c[1], d[1])
                    or max(c[1], d[1]) < min(a[1], b[1])
                ):
                    continue
                self._register_edge_pair(arm, (a, b), sheet, (c, d))

    def _register_edge_pair(self, arm, arm_edge, sheet, sheet_edge):
        a, b = arm_edge
        c, d = sheet_edge
        ab = b[0] - a[0], b[1] - a[1]
        cd = d[0] - c[0], d[1] - c[1]
        determinant = ab[0] * cd[1] - ab[1] * cd[0]
        if determinant:
            ac = c[0] - a[0], c[1] - a[1]
            t = Fraction(ac[0] * cd[1] - ac[1] * cd[0], determinant)
            u = Fraction(ac[0] * ab[1] - ac[1] * ab[0], determinant)
            if 0 <= t <= 1 and 0 <= u <= 1:
                exact = tuple(a[i] + t * ab[i] for i in (0, 1))
                point = tuple(round(v) for v in exact)
                # The crossing is rounded to the grid; keep the worst rounding
                # error so the record can report it.
                self.intersection_deviation = max(
                    self.intersection_deviation,
                    max(float(abs(point[i] - exact[i])) for i in (0, 1)) * PRECISION_UM,
                )
                self.incidence.record(arm, point, t)
                self.incidence.record(sheet, point, u)
        elif _cross(a, b, c) == 0:
            self._register_shared_points(arm, (a, b), sheet, (c, d))

    def _register_shared_points(self, left, left_edge, right, right_edge):
        """Record the endpoints that lie on both of two collinear source edges."""
        a, b = left_edge
        c, d = right_edge
        for point in {a, b, c, d}:
            if _on(point, a, b) and _on(point, c, d):
                for key, start, finish in ((left, a, b), (right, c, d)):
                    delta = finish[0] - start[0], finish[1] - start[1]
                    parameter = Fraction(
                        (point[0] - start[0]) * delta[0]
                        + (point[1] - start[1]) * delta[1],
                        delta[0] ** 2 + delta[1] ** 2,
                    )
                    self.incidence.record(key, point, parameter)

    def _register_owned_seams(self, source_edges):
        # A normalized Entity may comprise adjacent authored polygons whose shared
        # seam has different subdivisions. Register those existing endpoint nodes
        # before Boolean topology; no new crossing or contact tolerance is inferred.
        for entity in self.build_input.entities:
            if entity.semantic_id not in self.affected_owners:
                continue
            owned = set(entity.polygon_ids) - self.authored
            owned_edges = [
                (key, a, b) for key, (a, b) in source_edges.items() if key[0] in owned
            ]
            for n, (left, a, b) in enumerate(owned_edges):
                for right, c, d in owned_edges[n + 1 :]:
                    if left[0] == right[0] or _cross(a, b, c) or _cross(a, b, d):
                        continue
                    self._register_shared_points(left, (a, b), right, (c, d))

    # --- free sheet boundary ---------------------------------------------------

    def _trace_free_boundaries(self):
        """Trace the free sheet (sheet minus all metal) back to its source edges."""
        raw_free = {}
        for jid, record in self.records.items():
            free = _rings(
                _boolean(
                    _region([self.rings[record["authored_polygon_id"]]]),
                    _region([self.rings[pid] for pid in self.metal_ids]),
                    "not",
                )
            )
            raw_free[jid] = self._trace_to_sources(free)
        return raw_free

    def _trace_to_sources(self, free):
        """List `(start, end, source edges)` for each boundary subedge of `free`."""
        members = self.incidence.members
        traced = []
        for region in free:
            for index, ring in enumerate(region):
                for a, b in _edges((_oriented_ring(ring, index),)):
                    # Boolean may remove collinear source vertices. Reinsert
                    # registered nodes on this exact segment, then require each
                    # subedge to have its pre-recorded source-edge incidence.
                    delta = b[0] - a[0], b[1] - a[1]
                    nodes = sorted(
                        (point for point in members if _on(point, a, b)),
                        key=lambda point: (
                            (point[0] - a[0]) * delta[0] + (point[1] - a[1]) * delta[1]
                        ),
                    )
                    if not nodes or nodes[0] != a or nodes[-1] != b:
                        raise ValueError(
                            "numerical junction Boolean boundary has untraceable source incidence"
                        )
                    for start, finish in pairwise(nodes):
                        source = members[start] & members[finish]
                        if not source:
                            raise ValueError(
                                "numerical junction Boolean boundary has untraceable source incidence"
                            )
                        traced.append((start, finish, tuple(sorted(source))))
        return traced

    def _slice_free_boundaries(self, raw_free):
        """Cut each free boundary at its junction's two cut lines into three parts."""
        for jid, record in self.records.items():
            parts, divided = self._slice_at_cuts(record, raw_free[jid])
            self.sliced[jid] = parts
            self.free_edges[jid] = divided

    def _slice_at_cuts(self, record, traced):
        """Return `(parts, divided)`: edges of end A, centre and end B, and all edges.

        Cuts are lines of constant projection onto the junction direction.
        """
        dx, dy = (Fraction(str(v)) for v in record["direction_xy"])
        origin = Fraction(str(record["local_origin_projection_um"][0])) / Fraction(
            str(PRECISION_UM)
        )
        cuts = [origin + round(v / PRECISION_UM) for v in record["actual_cuts_um"]]

        def project(p):
            return p[0] * dx + p[1] * dy

        cap = [set(), set()]
        parts = [[], [], []]
        divided = []
        for a, b, sources in self.incidence.refine(traced):
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
                        events = self.incidence.events[source]
                        parameter = events[a] + u * (events[b] - events[a])
                        self.incidence.record(source, point, parameter)
            ordered = [point for _, point in sorted(split.items())]
            for start, end in pairwise(ordered):
                midpoint = (project(start) + project(end)) / 2
                which = 0 if midpoint < cuts[0] else 2 if midpoint > cuts[1] else 1
                # Source boundaries at an empty residual end belong to part 1 alone,
                # the central part (added below as the "central_rlc" polygon).
                if midpoint == cuts[1]:
                    which = 1
                edge = (start, end, sources)
                parts[which].append(edge)
                divided.append(edge)
        _close_cut_caps(cap, parts, dx, dy)
        return parts, divided

    # --- emitted model polygons ---------------------------------------------

    def _rebuild_affected_polygons(self):
        """Re-emit every polygon of the junction or its owners with incidence vertices."""
        affected_polygons = self.authored | {
            pid
            for entity in self.build_input.entities
            if entity.semantic_id in self.affected_owners
            for pid in entity.polygon_ids
        }
        emit = self.vertices.emit
        events = self.incidence.events
        model = {}
        for pid, polygon in self.originals.items():
            if pid not in affected_polygons:
                model[pid] = polygon
                continue
            region = []
            for index, ring in enumerate(self.rings[pid]):
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
        return model

    def _emit_junction_parts(self):
        """Create the central and end polygons of every junction and fill its record."""
        for jid, record in self.records.items():
            refine = self.incidence.refine
            parts = [self._lower(refine(part)) for part in self.sliced[jid]]
            central = self._lower(refine(self.sliced[jid][1]), simplify=True)
            if len(central) != 1 or len(central[0]) != 1 or len(central[0][0]) != 4:
                raise ValueError(
                    "numerical junction final central remainder is not a four-corner rectangle"
                )
            corners = _central_corners(central[0][0], record["direction_xy"])
            sheet = self.originals[record["authored_polygon_id"]]
            self.additions.append(
                LayoutPolygonSpec(
                    record["central_polygon_id"],
                    sheet.layer,
                    corners,
                    metadata={"aedt_junction_id": jid, "part": "central_rlc"},
                )
            )
            for n, part in ((0, parts[0]), (1, parts[2])):
                self._emit_end(jid, record, n, part, sheet)
            final_free = self._lower(refine(self.free_edges[jid]))
            if not emitted_partition_coverage(
                final_free, [*parts[0], (corners,), *parts[2]]
            ):
                raise ValueError(
                    "numerical junction emitted partition does not cover source free boundary"
                )
            self._record_model_results(record, sheet, final_free, corners)

    def _emit_end(self, jid, record, n, part, sheet):
        """Create the polygon of end `n` (0 = A, 1 = B), if its residual is nonempty."""
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
            self.additions.append(
                LayoutPolygonSpec(
                    end["polygon_id"],
                    sheet.layer,
                    part[0][0],
                    tuple(part[0][1:]),
                    net_name=end["net_id"],
                    metadata={
                        "aedt_junction_id": jid,
                        "source_entity_id": end["source_entity_id"],
                    },
                )
            )
            self.appended[end["source_entity_id"]].append(end["polygon_id"])
            self.physical_ends[end["source_entity_id"]].extend(
                self.incidence.refine(self.sliced[jid][0 if n == 0 else 2])
            )
        end["model_regions"] = _plain_with_dataclasses(part)

    def _record_model_results(self, record, sheet, final_free, corners):
        record["free_model_regions"] = _plain_with_dataclasses(final_free)
        record["central_model_regions"] = [[list(corners)]]
        record["proof_conditions"]["emitted_partition_coverage"] = True
        record["model_coordinate_method"] = "source_edge_incidence_and_two_cuts.v1"
        record["model_boolean_precision_um"] = PRECISION_UM
        record["maximum_model_source_coordinate_deviation_um"] = max(
            abs(round(v / PRECISION_UM) * PRECISION_UM - v)
            for pid in self.needed
            for ring in (self.originals[pid].exterior, *self.originals[pid].holes)
            for point in ring
            for v in point
        )
        record["maximum_model_intersection_coordinate_deviation_um"] = (
            self.intersection_deviation
        )
        record["authored_source_geometry"] = {
            "exterior": _plain_with_dataclasses(sheet.exterior),
            "holes": _plain_with_dataclasses(sheet.holes),
        }
        dx, dy = record["direction_xy"]
        record["model_cut_projection_ranges_um"] = [
            [
                min(x * dx + y * dy for x, y in edge),
                max(x * dx + y * dy for x, y in edge),
            ]
            for edge in ((corners[3], corners[0]), (corners[1], corners[2]))
        ]

    def _lower(self, traced, *, simplify=False):
        """Turn traced edges into emitted regions `((outer, *holes), ...)`.

        `simplify` drops collinear vertices, which the central rectangle needs so
        that it ends up with exactly four corners.
        """
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
                loop = _oriented_ring(loop, index)
                region.append(tuple(self.vertices.emit(point) for point in loop))
            regions.append(tuple(region))
        if sum(len(region) for region in regions) != len(loops):
            raise ValueError("junction residual has point-connected components")
        return regions

    def _rebuild_entities(self, model):
        """Replace each affected conductor Entity's polygons and support geometry."""
        authored = self.authored
        entities = []
        for entity in self.build_input.entities:
            ids = tuple(pid for pid in entity.polygon_ids if pid not in authored)
            if not ids and entity.polygon_ids:
                continue
            if entity.material_kind != "conductor":
                entities.append(entity)
                continue
            if (
                entity.semantic_id not in self.affected_owners
                and ids == entity.polygon_ids
            ):
                entities.append(entity)
                continue
            outer, loops = self._entity_support_loops(entity, ids, model)
            geometry = {
                **entity.geometry,
                "outer_loop": outer,
                "hole_loops": tuple(ring for ring in loops if ring is not outer),
            }
            entities.append(
                replace(
                    entity,
                    polygon_ids=(*ids, *self.appended[entity.semantic_id]),
                    geometry=geometry,
                )
            )
        return entities

    def _entity_support_loops(self, entity, ids, model):
        """Return `(outer, all loops)` of the Entity's emitted support boundary."""
        # Build support from the actual separate pieces, retaining all boundary
        # subdivisions. No second Boolean, clipping or world regridding occurs.
        emit = self.vertices.emit
        emitted_support = []
        for pid in ids:
            for index, ring in enumerate((model[pid].exterior, *model[pid].holes)):
                emitted_support.extend(_edges((_oriented_ring(ring, index),)))
        emitted_support.extend(
            (emit(a), emit(b)) for a, b, _ in self.physical_ends[entity.semantic_id]
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
        return outer[0], loops


def _close_cut_caps(cap, parts, dx, dy):
    """Ensure each cut line is closed by a full source section on both sides.

    Cut points are ordered along the cut line; a gap between neighbouring points
    that no part edge covers gets a zero-source edge pair so both sides close.
    """
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


def _central_corners(corner_ring, direction):
    """Order the central rectangle's corners in the authored A->B sense.

    Native AEDT expects the corners ordered by (along, across) extremes of the
    junction direction, starting at the minimum corner.
    """
    dx, dy = direction
    uv = [(x * dx + y * dy, -x * dy + y * dx) for x, y in corner_ring]
    amin, amax = min(u for u, _ in uv), max(u for u, _ in uv)
    vmin, vmax = min(v for _, v in uv), max(v for _, v in uv)
    return tuple(
        min(
            zip(corner_ring, uv, strict=False),
            key=lambda item: (item[1][0] - u) ** 2 + (item[1][1] - v) ** 2,
        )[0]
        for u, v in ((amin, vmin), (amax, vmin), (amax, vmax), (amin, vmax))
    )
