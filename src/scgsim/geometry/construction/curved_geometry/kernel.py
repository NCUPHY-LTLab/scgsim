"""Create exact native source curves and detach their canonical identities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    import gmsh

    from scgsim.geometry.construction.curved_geometry.surfaces import SharedSurfaces

from scgsim.geometry.construction.curved_geometry.boundaries import (
    circle_through_points,
)
from scgsim.geometry.models.input import SourceCurveSpec
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
)


class GmshApi(Protocol):
    """Native module surface used by the exact curve construction stages."""

    @property
    def model(self) -> type[gmsh.model]:
        """Expose Gmsh's model namespace without loading it during import."""
        ...

    @property
    def option(self) -> type[gmsh.option]:
        """Expose native options used by shared surface-first shell correction."""
        ...

    @property
    def __version__(self) -> str:
        """Expose the native version recorded in the detached plan."""
        ...

    def write(self, fileName: str) -> None:
        """Write the active native model to its boundary transport file."""
        ...


class _BSplineOptions(TypedDict, total=False):
    """Authored options forwarded to the native B-spline constructor."""

    degree: int
    weights: list[float]
    knots: list[float]
    multiplicities: list[int]


def geometry_identity(prefix: str, value: object) -> str:
    """Name canonical geometry from its unchanged serialized description."""
    return (
        prefix
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )


def edge_vertices(gmsh: GmshApi, edge: int) -> list[int]:
    """Order native vertices in the edge parameter direction."""
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


def face_loops(
    gmsh: GmshApi, occ: type[gmsh.model.occ], face: int
) -> list[tuple[int, ...]]:
    """Recover ordered oriented wires from a native face."""
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
            vertices = edge_vertices(gmsh, abs(tag))
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


class CurveKernel:
    """Share native point identity and source history across exact curves."""

    def __init__(self, gmsh_api: GmshApi):
        self.gmsh = gmsh_api
        self.occ = gmsh_api.model.occ
        self.points = {}
        self.source_edges = {}

    def point(self, xyz):
        """Reuse one OCC point for each exact coordinate tuple."""
        xyz = cast(tuple[float, float, float], tuple(float(x) for x in xyz))
        if xyz not in self.points:
            self.points[xyz] = self.occ.addPoint(*xyz)
        return self.points[xyz]

    def line(self, a, b):
        """Represent a straight edge in the common cubic B-spline basis."""
        points = [
            self.point(a),
            self.point(tuple(a[j] + (b[j] - a[j]) / 3 for j in range(3))),
            self.point(tuple(a[j] + 2 * (b[j] - a[j]) / 3 for j in range(3))),
            self.point(b),
        ]
        return self.occ.addBSpline(
            points, degree=3, knots=[0, 1], multiplicities=[4, 4]
        )

    def curve(self, spec: SourceCurveSpec, source):
        """Create, trim, then open native source curves in the required order."""
        points = [(*xy, 0.0) for xy in spec.points_um]
        native_curve_data = {}
        if spec.kind == "line_segment":
            tags = [self.line(*points)]
        elif spec.kind == "circular_arc":
            tags, native_curve_data = self._circular_arc(spec, points)
        elif spec.kind == "interpolation_spline":
            tags = [self.occ.addSpline([self.point(p) for p in points])]
        elif spec.kind == "bspline":
            tags = self._bspline(spec, points)
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
            tags = self._trim(tags, spec.parameter_interval)
        return self._open_periodic_edges(tags)

    def _circular_arc(self, spec: SourceCurveSpec, points):
        """Lower an arc through exact homogeneous degree elevation."""
        center, radius = circle_through_points(spec.points_um)
        cx, cy = center
        angles = [math.atan2(y - cy, x - cx) for x, y in spec.points_um]
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
            q: list[tuple[float, ...]] = [
                (cx + radius * math.cos(lo), cy + radius * math.sin(lo), 0),
                (cx + radius * math.cos(m) / w, cy + radius * math.sin(m) / w, 0),
                (cx + radius * math.cos(hi), cy + radius * math.sin(hi), 0),
            ]
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

    def _bspline(self, spec: SourceCurveSpec, points):
        """Pass authored B-spline options to OCC without refitting."""
        arguments: _BSplineOptions = {}
        if spec.degree is not None:
            arguments["degree"] = spec.degree
        if spec.weights:
            arguments["weights"] = list(spec.weights)
        if spec.knots:
            arguments["knots"] = list(spec.knots)
        if spec.multiplicities:
            arguments["multiplicities"] = list(spec.multiplicities)
        return [self.occ.addBSpline([self.point(p) for p in points], **arguments)]

    def _trim(self, tags, interval: tuple[float, float]):
        """Retain native children inside the requested source parameter interval."""
        self.occ.synchronize()
        tag = tags[0]
        lo, hi = interval
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
                "requested_parameter_interval": interval,
            }
        return retained

    def _open_periodic_edges(self, tags):
        """Split periodic topology while preserving the native source geometry."""
        # A periodic edge has no pair of distinct topological endpoints. Split
        # its native parameter domain for the four-edge ruled wall construction;
        # this changes topology only, never the source spline geometry.
        self.occ.synchronize()
        open_tags = []
        for tag in tags:
            if len(edge_vertices(self.gmsh, tag)) == 2:
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
        """Create a planar face in the exterior wire orientation basis."""
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


class CurveDetacher:
    """Keep native source and sweep maps together while detaching shared curves."""

    def __init__(self, geometry: SharedSurfaces):
        self.gmsh = geometry.planar.kernel.gmsh
        self.lifted_inverse = {
            tag: (level, edge) for (level, edge), tag in geometry.lifted.items()
        }
        self.vertical_inverse = {tag: key for key, tag in geometry.vertical.items()}
        self.curve_descriptors = geometry.planar.curve_descriptors
        self.native_edges = geometry.planar.native_edges
        self.z_values = geometry.planar.z_values

    def detach(self, edge: int):
        """Detach one canonical curve from retained source and sweep history."""
        endpoints = [
            tuple(float(v) for v in self.gmsh.model.getValue(0, tag, []))
            for tag in edge_vertices(self.gmsh, edge)
        ]
        if len(endpoints) != 2:
            raise ValueError(
                "native arrangement edge lacks two detached topology endpoints"
            )
        if edge in self.lifted_inverse:
            level, source = self.lifted_inverse[edge]
            geometry = {**self.curve_descriptors[source], "z_um": self.z_values[level]}
            kinds = {record["geometry"]["kind"] for record in self.native_edges[source]}
            kind = next(iter(kinds)) if len(kinds) == 1 else "bspline"
        elif edge in self.vertical_inverse:
            geometry = {"kind": "line_segment", "endpoints_um": endpoints}
            kind = "line_segment"
        else:
            raise ValueError(
                "native sidewall construction replaced a planned shared boundary curve"
            )
        cid = geometry_identity("C__", geometry)
        pids = [geometry_identity("P__", xyz) for xyz in endpoints]
        bounds = self.gmsh.model.getParametrizationBounds(1, edge)
        return CurvePlanRecord(
            cid,
            kind,
            pids[0],
            pids[1],
            geometry=geometry,
            parameter_interval=(float(bounds[0][0]), float(bounds[1][0])),
        ), dict(zip(pids, endpoints, strict=False))
