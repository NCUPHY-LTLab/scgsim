"""Verify native prism cap and side incidence without changing geometry."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import pairwise
from typing import Any

from scgsim.aedt.runtime.native.junction_topology import (
    native_coordinates_close,
    native_face_edges,
    native_on_segment,
    native_straight_ring,
    native_z_may_contact,
)
from scgsim.geometry.primitives.planar_loops import (
    cancel_reversed_planar_edges,
    simple_planar_loops_from_edges,
)

_NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM = 1e-5


def native_extrusion_sides(obj, z_range, thickness, observations):
    """Verify straight prism sides against actual caps, independent of subdivision.

    Each actual cap edge must belong to exactly one side face by native ID;
    only associated endpoints are compared under the coordinate policy. IDs are
    scoped to this live object, never source geometry or another session. Raw
    subdivision edges retain their own incidence. Side faces remain simple
    vertical quadrilaterals after comparison-only straight-chain reduction;
    general face tessellation is unsupported.
    """

    return ExtrusionReadback(obj, z_range, thickness, observations).run()


class ExtrusionReadback:
    """Keep cap, side and boundary incidence together through extrusion verification."""

    boundary_edges: Any
    cap_edges: Any
    cap_observations: Any
    cap_z: Any
    obj: Any
    observations: Any
    side_owners: Any
    thickness: Any
    z_range: Any

    def __init__(
        self, obj: Any, z_range: Any, thickness: Any, observations: Any
    ) -> None:
        self.obj = obj
        self.z_range = z_range
        self.thickness = thickness
        self.observations = observations

    def run(self) -> None:
        """Read native state and verify each dependency in its original order."""
        self._read_caps()
        self._index_cap_edges()
        self._read_side_faces()
        self._verify_vertices()
        return self._verify_complete_incidence()

    def _read_caps(self) -> None:
        self.cap_observations = self.observations[-2:]
        self.cap_edges = ({}, {})
        self.side_owners = ({}, {})
        self.cap_z = [
            [p[2] for p in observation["face_vertices_um"]]
            for observation in self.cap_observations
        ]
        self.boundary_edges = [
            tuple(tuple(Fraction(str(v)) for v in p) for p in edge["vertices_um"])
            for observation in self.cap_observations
            for edge in observation["edges"]
        ]
        if any(
            not native_coordinates_close((height,), (self.thickness,))
            for height in (
                max(self.cap_z[1]) - min(self.cap_z[0]),
                min(self.cap_z[1]) - max(self.cap_z[0]),
            )
        ):
            raise RuntimeError("native junction actual top/bottom extrusion differs")

    def _index_cap_edges(self) -> None:
        for n, observation in enumerate(self.cap_observations):
            for edge in observation["edges"]:
                edge_id = edge.get("edge_id")
                if isinstance(edge_id, bool) or not isinstance(edge_id, int):
                    raise self._incidence_error("missing_cap_edge_id", n, cap_edge=edge)
                if edge_id in self.cap_edges[0] or edge_id in self.cap_edges[1]:
                    raise self._incidence_error(
                        "duplicate_cap_edge_id", n, cap_edge=edge
                    )
                self.cap_edges[n][edge_id] = edge

    def _read_side_faces(self) -> None:
        for face in self.obj.faces:
            positions = [vertex.position for vertex in face.vertices]
            if positions and any(
                all(native_coordinates_close((p[2],), (z,)) for p in positions)
                for z in self.z_range
            ):
                continue  # Cap shape and straight edges were already read in full.
            edges, edge_observations = native_face_edges(face, self.obj.name)
            rings = simple_planar_loops_from_edges(cancel_reversed_planar_edges(edges))
            if len(rings) != 1:
                raise RuntimeError(
                    "native junction extrusion side has unsupported loops"
                )
            corners, chains = native_straight_ring(rings[0])
            self._verify_side_height(corners)
            edge_records = {
                tuple(sorted(edge)): observation
                for edge, observation in zip(edges, edge_observations, strict=False)
            }
            if len(edge_records) != len(edges):
                raise RuntimeError(
                    "native junction extrusion side has repeated boundary edges"
                )
            face_state = getattr(face, "__dict__", {})
            face_id = face_state.get("_id", face_state.get("id"))
            for chain in chains:
                n = self._layer(chain[0])
                if self._layer(chain[-1]) != n:
                    continue  # Existing vertical-chain checks retain intermediate nodes.
                for a, b in pairwise(chain):
                    side_edge = edge_records[tuple(sorted((a, b)))]
                    edge_id = side_edge.get("edge_id")
                    cap_edge = (
                        self.cap_edges[n].get(edge_id)
                        if isinstance(edge_id, int) and not isinstance(edge_id, bool)
                        else None
                    )
                    if cap_edge is None:
                        raise self._incidence_error(
                            "unmatched_side_edge",
                            n,
                            side_edge=side_edge,
                            side_face_id=face_id,
                        )
                    wanted = cap_edge["vertices_um"]
                    actual = side_edge["vertices_um"]
                    if not any(
                        all(
                            native_coordinates_close(p, q)
                            for p, q in zip(actual, ordering, strict=False)
                        )
                        for ordering in (wanted, wanted[::-1])
                    ):
                        raise self._incidence_error(
                            "linked_edge_geometry_mismatch",
                            n,
                            cap_edge,
                            side_edge,
                            face_id,
                        )
                    if edge_id in self.side_owners[n]:
                        raise self._incidence_error(
                            "duplicate_side_incidence",
                            n,
                            cap_edge,
                            side_edge,
                            face_id,
                            self.side_owners[n][edge_id],
                        )
                    self.side_owners[n][edge_id] = face_id
            self.boundary_edges.extend(
                zip(corners, (*corners[1:], corners[0]), strict=False)
            )
            self.observations.append(
                {
                    "object_name": self.obj.name,
                    "extrusion_side_vertices_um": [
                        list(map(float, p)) for p in rings[0]
                    ],
                    "edges": edge_observations,
                }
            )

    def _verify_vertices(self) -> None:
        for vertex in self.obj.vertices:
            position = vertex.position
            if (
                position is None
                or len(position) != 3
                or not native_z_may_contact(
                    (position[2], position[2]), (min(self.cap_z[0]), max(self.cap_z[1]))
                )
                or not any(
                    native_on_segment(position, a, b) for a, b in self.boundary_edges
                )
            ):
                raise RuntimeError(
                    "native solid vertex differs from actual extrusion boundary"
                )

    def _verify_complete_incidence(self) -> None:
        for n, edges in enumerate(self.cap_edges):
            for edge_id, edge in edges.items():
                if edge_id not in self.side_owners[n]:
                    raise self._incidence_error(
                        "missing_side_incidence", n, cap_edge=edge
                    )

    def _verify_side_height(self, corners) -> None:
        """Verify vertical side edges and the actual top-to-bottom height."""
        sides = [[p for p in corners if self._layer(p) == n] for n in (0, 1)]
        if len(corners) != 4 or any(len(side) != 2 for side in sides):
            raise RuntimeError(
                "native junction extrusion side is not a vertical quadrilateral"
            )
        if any(
            self._layer(a) != self._layer(b)
            and not native_coordinates_close(a[:2], b[:2])
            for a, b in zip(corners, (*corners[1:], corners[0]), strict=False)
        ):
            raise RuntimeError("native junction extrusion side has nonvertical edges")
        for point in sides[0]:
            mates = [p for p in sides[1] if native_coordinates_close(point[:2], p[:2])]
            if len(mates) != 1 or not native_coordinates_close(
                (float(mates[0][2] - point[2]),), (self.thickness,)
            ):
                raise RuntimeError(
                    "native junction actual top/bottom extrusion differs"
                )

    def _layer(self, point):
        found = [
            n
            for n, z in enumerate(self.z_range)
            if native_coordinates_close((point[2],), (z,))
        ]
        if len(found) != 1 or any(
            not native_coordinates_close((point[2],), (z,))
            for z in self.cap_z[found[0]]
        ):
            raise RuntimeError("native junction actual extrusion vertex Z differs")
        return found[0]

    def _incidence_error(  # noqa: PLR0913 -- each argument is a distinct native incidence diagnostic.
        self,
        condition,
        n,
        cap_edge=None,
        side_edge=None,
        side_face_id=None,
        prior_side_face_id=None,
    ):
        diagnostic = {
            "object_name": self.obj.name,
            "condition": condition,
            "cap_index": n,
            "cap_face_id": self.cap_observations[n].get("face_id"),
            "edge_id": (cap_edge or side_edge or {}).get("edge_id"),
            "cap_edge_endpoints_um": (cap_edge or {}).get("vertices_um"),
            "side_edge_endpoints_um": (side_edge or {}).get("vertices_um"),
            "side_face_id": side_face_id,
            "prior_side_face_id": prior_side_face_id,
            "model_units": "um",
            "absolute_tolerance_um": _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
            "relative_tolerance": 0.0,
        }
        return RuntimeError(
            "native junction extrusion cap-side incidence differs"
            + "; extrusion_incidence_diagnostic="
            + json.dumps(diagnostic, allow_nan=False)
        )
