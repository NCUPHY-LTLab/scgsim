"""Views of a sidecar that readers need and the file stores only once: anything by native tag, the members of
each group, and the orders of the lists v2 wrote.

Owns the computations every reader shares, so no reader rebuilds them its own way.  The tables hold each fact
once (a face names its group, an edge its vertices); a view computes the other direction or order from them:

* :func:`native_tags` and :class:`TagIndex`: planned id <-> native tag, ``tag = 1 + position`` in
  ``binding.ids_by_tag``;
* :func:`group_members`: the faces or volumes of each group, by planned id;
* :func:`expected`: what the load-side checks compare, by native tag (positions, boundaries, areas, volumes);
* :func:`embedded_faces` and :func:`embedded_edges`: the embeddings the mesher re-applies, host by host in tag
  order;
* :func:`ports` and :func:`mesh_features`: the ports and features in the order a reader iterates them;
* :func:`arrangement` and :func:`curved_wall_faces`: the arrangement summary and the curved walls.

Every view takes a sidecar the shared reader accepted (:func:`.read.check_sidecar`).
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

from .records import MeshFeature, Port, Sidecar

_DIMENSIONS = ("0", "1", "2", "3")


@dataclass(frozen=True)
class TagIndex:
    """The native tag of every planned id, by dimension (0 vertices, 1 edges, 2 faces, 3 volumes)."""

    tags: tuple[dict[str, int], ...]

    @classmethod
    def of(cls, sidecar: Sidecar) -> "TagIndex":
        """The tags of ``sidecar``'s binding."""
        by_tag = sidecar["binding"]["ids_by_tag"]
        return cls(
            tuple(
                {planned: tag for tag, planned in enumerate(by_tag[d], start=1)}
                for d in _DIMENSIONS
            )
        )

    def tag(self, dimension: int, planned: str) -> int:
        """The native tag of ``planned`` in ``dimension``."""
        return self.tags[dimension][planned]

    def tags_of(self, dimension: int, planned: Iterable[str]) -> list[int]:
        """The native tags of ``planned``, in the same order."""
        table = self.tags[dimension]
        return [table[p] for p in planned]


def native_tags(sidecar: Sidecar, dimension: int, planned: Iterable[str]) -> list[int]:
    """The native tags of the planned ids ``planned`` of ``dimension``, in the same order."""
    return TagIndex.of(sidecar).tags_of(dimension, planned)


def group_members(sidecar: Sidecar) -> dict[str, list[str]]:
    """Group name -> its members by planned id, sorted: the written faces of a surface group, the emitted
    volumes of a volume group."""
    members: dict[str, list[str]] = {name: [] for name in sidecar["group"]}
    for face_id, face in sorted(sidecar["face"].items()):
        if face["group"] is not None:
            members[face["group"]].append(face_id)
    for volume_id, volume in sorted(sidecar["volume"].items()):
        if volume["emitted"] and volume["group"] is not None:
            members[volume["group"]].append(volume_id)
    return members


class Expected(TypedDict):
    """What a loaded model must hold, by native tag (tag = 1 + position): where each vertex sits, the boundary
    of each edge, face and solid as tag lists (an edge's in written order, the others sorted), and the planned
    face areas and solid volumes."""

    vertices_um: list[list[float]]
    edge_vertices: list[list[int]]
    face_edges: list[list[int]]
    solid_faces: list[list[int]]
    face_area_um2: list[float]
    solid_volume_um3: list[float]


def expected(sidecar: Sidecar) -> Expected:
    """The load-side expectations of ``sidecar``, by native tag."""
    index = TagIndex.of(sidecar)
    by_tag = sidecar["binding"]["ids_by_tag"]
    edges, faces, volumes = sidecar["edge"], sidecar["face"], sidecar["volume"]
    return Expected(
        vertices_um=[list(sidecar["vertex"][v]["position_um"]) for v in by_tag["0"]],
        edge_vertices=[index.tags_of(0, edges[e]["vertices"]) for e in by_tag["1"]],
        face_edges=[
            sorted(
                {index.tag(1, e) for loop in faces[f]["loops"] for e in loop["edges"]}
            )
            for f in by_tag["2"]
        ],
        solid_faces=[
            sorted(index.tags_of(2, volumes[v]["faces"])) for v in by_tag["3"]
        ],
        face_area_um2=[faces[f]["area_um2"] for f in by_tag["2"]],
        solid_volume_um3=[volumes[v]["volume_um3"] for v in by_tag["3"]],
    )


def embedded_faces(sidecar: Sidecar) -> list[tuple[str, str]]:
    """(sheet face, host volume) pairs, host by host in tag order: sheets inside an emitted volume."""
    volumes = sidecar["volume"]
    return [
        (face, volume_id)
        for volume_id in sidecar["binding"]["ids_by_tag"]["3"]
        for face in volumes[volume_id]["embedded_faces"]
    ]


def embedded_edges(sidecar: Sidecar) -> list[tuple[str, str]]:
    """(refinement edge, host face) pairs, host by host in tag order: refinement-only curves inside a face."""
    faces = sidecar["face"]
    return [
        (edge, face_id)
        for face_id in sidecar["binding"]["ids_by_tag"]["2"]
        for edge in faces[face_id]["embedded_edges"]
    ]


def ports(sidecar: Sidecar) -> list[tuple[str, Port]]:
    """(port sheet id, port) pairs by port index, then id."""
    return sorted(
        sidecar["port"].items(), key=lambda item: (item[1]["port_index"], item[0])
    )


def mesh_features(sidecar: Sidecar) -> list[MeshFeature]:
    """The mesh features sorted by kind, source and the canonical JSON of their facts, as the mesher reads them."""
    return sorted(sidecar["mesh_feature"].values(), key=_feature_order)


def _feature_order(feature: MeshFeature) -> tuple[str, str, str]:
    facts = json.dumps(feature["facts"], sort_keys=True, separators=(",", ":"))
    return feature["kind"], feature["source"] or "", facts


class Arrangement(TypedDict):
    """The measured snap drift and arc deviation of the arrangement (grid units), and the number of arc edges."""

    max_snap_drift_grid: float | None
    max_arc_deviation_grid: float | None
    arc_edges: int


def arrangement(sidecar: Sidecar) -> Arrangement:
    """The arrangement summary: snap drift and arc deviation from the sliver summary, and the arc edges."""
    summary = sidecar["summary"]
    slivers = summary["slivers"]
    return Arrangement(
        max_snap_drift_grid=slivers.get("max_snap_drift_grid"),
        max_arc_deviation_grid=slivers.get("max_arc_deviation_grid"),
        arc_edges=summary["arc_edges"],
    )


def curved_wall_faces(sidecar: Sidecar) -> list[str]:
    """The faces on a cylinder, in tag order."""
    faces = sidecar["face"]
    return [f for f in sidecar["binding"]["ids_by_tag"]["2"] if faces[f]["curved"]]
