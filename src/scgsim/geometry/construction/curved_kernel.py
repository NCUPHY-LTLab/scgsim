"""Native OpenCASCADE (OCC) kernel helpers for curved construction.

Owns stable identity hashing, edge and face topology queries, and `_Kernel`,
which turns one source curve spec into native B-spline edges and one footprint
into a planar face. Every curve kind is built as a degree-3 B-spline so that
lines, arcs and splines share one native edge type. Does not decide which
curves exist (see `curved_source`) or how cells are arranged.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

from scgsim.geometry.construction.curved_source import _circle_points


def _identity(prefix, value):
    """Content-addressed id: `prefix` plus 24 hex digits of the value's SHA-256.

    Ids depend on content, not native tags, so they stay stable when OCC
    renumbers entities.
    """
    return (
        prefix
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )


def _edge_vertices(gmsh, edge):
    """Vertex tags of a native edge, ordered by curve parameter (start first)."""
    vertices = [
        abs(tag)
        for dim, tag in gmsh.model.getBoundary([(1, edge)], oriented=True)
        if dim == 0
    ]
    return sorted(
        vertices,
        key=lambda tag: float(
            gmsh.model.getParametrization(1, edge, gmsh.model.getValue(0, tag, []))[0]
        ),
    )


def _face_loops(gmsh, occ, face):
    """Signed edge tags of each loop of a face, each walked as one closed chain."""
    oriented = {
        abs(tag): tag
        for dim, tag in gmsh.model.getBoundary([(2, face)], oriented=True)
        if dim == 1
    }
    result = []
    for raw in occ.getCurveLoops(face)[1]:
        pending = [oriented[abs(int(tag))] for tag in raw]
        ordered = [pending.pop(0)]

        def endpoints(tag):
            vertices = _edge_vertices(gmsh, abs(tag))
            return vertices if tag > 0 else vertices[::-1]

        while pending:
            end = endpoints(ordered[-1])[-1]
            next_index = next(
                (i for i, tag in enumerate(pending) if endpoints(tag)[0] == end), None
            )
            if next_index is None:
                raise ValueError(
                    "native face boundary does not form one oriented closed chain"
                )
            ordered.append(pending.pop(next_index))
        result.append(tuple(ordered))
    return result


class _Kernel:
    """Builds source curves and footprints as native OCC entities, in call order.

    `points` de-duplicates OCC points by exact coordinates so that curves meeting
    at a source vertex share one native vertex. `source_edges` records, per native
    edge tag, which authored curve it came from.
    """

    def __init__(self, gmsh):
        self.gmsh = gmsh
        self.occ = gmsh.model.occ
        self.points = {}
        self.source_edges = {}

    def point(self, xyz):
        xyz = tuple(float(x) for x in xyz)
        if xyz not in self.points:
            self.points[xyz] = self.occ.addPoint(*xyz)
        return self.points[xyz]

    def line(self, a, b):
        # A straight line is a cubic B-spline with collinear control points so
        # it is the same native edge type as every curved neighbour.
        points = [
            self.point(a),
            self.point(tuple(a[j] + (b[j] - a[j]) / 3 for j in range(3))),
            self.point(tuple(a[j] + 2 * (b[j] - a[j]) / 3 for j in range(3))),
            self.point(b),
        ]
        return self.occ.addBSpline(
            points, degree=3, knots=[0, 1], multiplicities=[4, 4]
        )

    def curve(self, spec, source):
        """Add one source curve; return its open native edge tags."""
        points = [(*xy, 0.0) for xy in spec.points_um]
        native_curve_data = {}
        if spec.kind == "line_segment":
            tags = [self.line(*points)]
        elif spec.kind == "circular_arc":
            tags, native_curve_data = self._arc(spec, points)
        elif spec.kind == "interpolation_spline":
            tags = [self.occ.addSpline([self.point(p) for p in points])]
        elif spec.kind == "bspline":
            tags = [self._bspline(spec, points)]
        else:
            raise ValueError(f"unsupported reconstruction kind {spec.kind!r}")
        for index, tag in enumerate(tags):
            self.source_edges[tag] = {
                "source": source,
                "geometry": asdict(spec),
                "native_piece_index": index,
                **native_curve_data,
            }
        if spec.parameter_interval is not None:
            tags = self._trim_to_parameter_interval(spec, tags[0])
        return self._split_periodic_edges(tags)

    def _arc(self, spec, points):
        controls, pieces = _rational_arc_controls(spec.points_um, points)
        # Join exact rational Bezier spans in one native parameter domain.
        # Internal controls are not arrangement vertices or contact points.
        knots = [i / pieces for i in range(pieces + 1)]
        multiplicities = [4] + [3] * (pieces - 1) + [4]
        tags = [
            self.occ.addBSpline(
                [self.point(tuple(v[j] / v[3] for j in range(3))) for v in controls],
                degree=3,
                weights=[v[3] for v in controls],
                knots=knots,
                multiplicities=multiplicities,
            )
        ]
        native_curve_data = {
            "native_arc_span_knots": knots,
            "native_curve_parameter_bounds": (0.0, 1.0),
        }
        return tags, native_curve_data

    def _bspline(self, spec, points):
        arguments = {}
        if spec.degree is not None:
            arguments["degree"] = spec.degree
        if spec.weights:
            arguments["weights"] = list(spec.weights)
        if spec.knots:
            arguments["knots"] = list(spec.knots)
        if spec.multiplicities:
            arguments["multiplicities"] = list(spec.multiplicities)
        return self.occ.addBSpline([self.point(p) for p in points], **arguments)

    def _trim_to_parameter_interval(self, spec, tag):
        """Keep only the part of `tag` inside the requested parameter interval."""
        self.occ.synchronize()
        lo, hi = spec.parameter_interval
        points = [
            self.occ.addPoint(*self.gmsh.model.getValue(1, tag, [u])) for u in (lo, hi)
        ]
        output, mapping = self.occ.fragment(
            [(1, tag)], [(0, p) for p in points], removeObject=False
        )
        self.occ.synchronize()
        retained = []
        for dim, candidate in mapping[0]:
            if dim != 1:
                continue
            bounds = self.gmsh.model.getParametrizationBounds(1, candidate)
            xyz = self.gmsh.model.getValue(
                1, candidate, [(float(bounds[0][0]) + float(bounds[1][0])) / 2]
            )
            parameter = float(self.gmsh.model.getParametrization(1, tag, xyz)[0])
            if lo <= parameter <= hi:
                retained.append(candidate)
            self.source_edges[candidate] = {
                **self.source_edges[tag],
                "requested_parameter_interval": spec.parameter_interval,
            }
        return retained

    def _split_periodic_edges(self, tags):
        # A periodic edge has no pair of distinct topological endpoints. Split
        # its native parameter domain for the four-edge ruled wall construction;
        # this changes topology only, never the source spline geometry.
        self.occ.synchronize()
        open_tags = []
        for tag in tags:
            if len(_edge_vertices(self.gmsh, tag)) == 2:
                open_tags.append(tag)
                continue
            lower, upper = self.gmsh.model.getParametrizationBounds(1, tag)
            lo, hi = float(lower[0]), float(upper[0])
            split_points = [
                self.occ.addPoint(*self.gmsh.model.getValue(1, tag, [u]))
                for u in (lo, (lo + hi) / 2)
            ]
            _, history = self.occ.fragment(
                [(1, tag)], [(0, p) for p in split_points], removeObject=False
            )
            self.occ.synchronize()
            for dim, piece in history[0]:
                if dim == 1:
                    open_tags.append(piece)
                    self.source_edges[piece] = {
                        **self.source_edges[tag],
                        "periodic_source_parameter_bounds": (lo, hi),
                    }
        return open_tags

    def face(self, chains):
        """Add one planar face from outer-then-hole chains; return it and its edges."""
        loops = []
        edges = []
        for curves, source in chains:
            tags = []
            for index, curve in enumerate(curves):
                tags.extend(self.curve(curve, {**source, "curve_index": index}))
            wire = self.occ.addCurveLoop(tags)
            orientation_face = self.occ.addPlaneSurface([wire])
            self.occ.synchronize()
            bounds = self.gmsh.model.getParametrizationBounds(2, orientation_face)
            normal = self.gmsh.model.getNormal(
                orientation_face,
                [float((a + b) / 2) for a, b in zip(*bounds, strict=False)],
            )
            self.occ.remove([(2, orientation_face)], recursive=False)
            # OCC's plane-surface constructor subtracts subsequent wires in the
            # outer wire's plane basis. Normalize that basis, not authored order.
            if normal[2] < 0:
                wire = self.occ.addCurveLoop([-tag for tag in reversed(tags)])
            loops.append(wire)
            edges.extend(tags)
        return self.occ.addPlaneSurface(loops), edges


def _rational_arc_controls(arc_points_um, points):
    """Exact rational-Bezier controls of a circular arc, in homogeneous form.

    Returns `(controls, pieces)`: one rational quadratic span per quarter turn,
    each degree-elevated to cubic so spans can join in a single B-spline.
    """
    center, radius = _circle_points(arc_points_um)
    cx, cy = center
    angles = [math.atan2(y - cy, x - cx) for x, y in arc_points_um]
    ccw = (angles[2] - angles[0]) % (2 * math.pi)
    mid = (angles[1] - angles[0]) % (2 * math.pi)
    sweep = ccw if mid < ccw else ccw - 2 * math.pi
    pieces = math.ceil(abs(sweep) / (math.pi / 2))
    controls = []
    for i in range(pieces):
        lo = angles[0] + sweep * i / pieces
        hi = angles[0] + sweep * (i + 1) / pieces
        half = (hi - lo) / 2
        w = math.cos(half)
        m = (lo + hi) / 2
        q = [
            (cx + radius * math.cos(lo), cy + radius * math.sin(lo), 0),
            (cx + radius * math.cos(m) / w, cy + radius * math.sin(m) / w, 0),
            (cx + radius * math.cos(hi), cy + radius * math.sin(hi), 0),
        ]
        # Keep the authored end points bit-exact rather than re-evaluated.
        if i == 0:
            q[0] = points[0]
        if i == pieces - 1:
            q[2] = points[-1]
        # Exact homogeneous degree elevation of a rational quadratic arc.
        h = [
            (*[v * weight for v in xyz], weight)
            for xyz, weight in zip(q, [1, w, 1], strict=False)
        ]
        elevated = [
            h[0],
            tuple((h[0][j] + 2 * h[1][j]) / 3 for j in range(4)),
            tuple((2 * h[1][j] + h[2][j]) / 3 for j in range(4)),
            h[2],
        ]
        controls.extend(elevated if i == 0 else elevated[1:])
    return controls, pieces
