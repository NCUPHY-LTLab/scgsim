"""Synthetic sidecars, audits and rounds for the meshing tests, so most of them need no Gmsh."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scgsim.contracts.sgb_sidecar import SIDECAR_SCHEMA
from scgsim.meshing.driver import RoundOutcome
from scgsim.meshing.features import SidecarFeatures, read_features
from scgsim.meshing.findings import BadElement, CurveCandidate, MeshAudit

EDGE_COUNT = 12
VERTEX_COUNT = 8
# The faces and the edges of their one loop; edges e9..e11 bound no face.
FACE_LOOPS = {
    "f0": ("e0", "e1", "e2"),
    "f1": ("e3", "e4", "e5"),
    "f2": ("e6", "e7", "e8"),
}
# The table a mesh feature's source names, by kind.
SOURCE_TABLE = {
    "film_sidewall": "entity",
    "arc": "entity",
    "band_boundary": "band_family",
    "refinement_curve": "refinement_request",
    "port_edge": "port",
}


def sidecar_with(
    *features: dict[str, Any],
    embedded_faces: Sequence[tuple[str, str]] = (),
    embedded_edges: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """A minimal valid v3 sidecar: twelve edges e0..e11 on eight vertices, three faces f0..f2, the emitted
    substrate ``vol0`` and vacuum ``vol1`` and the unemitted conductor ``pad`` (``vol2``), with ``features``.

    Each feature's source gets a record of the table its kind names, so every reference resolves;
    ``embedded_faces`` are (face, host volume) pairs and ``embedded_edges`` (edge, host face) pairs.
    """
    document: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "model": {
            "name": "synthetic",
            "identity_digest": "model:0000000000000000",
            "route": "B",
            "route_policy": {
                "conductor_volumes": "excluded",
                "conductor_sheets": False,
            },
            "grid_um": 0.001,
            "tolerance_um": 1e-6,
            "files": {},
        },
        "binding": {
            "rule": "tag = 1 + position",
            "ids_by_tag": {
                "0": [f"v{i}" for i in range(VERTEX_COUNT)],
                "1": [f"e{i}" for i in range(EDGE_COUNT)],
                "2": list(FACE_LOOPS),
                "3": ["vol0", "vol1"],
            },
        },
        "summary": {
            "counts": {
                **dict.fromkeys(
                    ("wires", "shells", "surfaces", "cylinders", "arcs", "pcurves"), 0
                ),
                **dict.fromkeys(("free_faces", "free_edges"), 0),
                "vertices": VERTEX_COUNT,
                "edges": EDGE_COUNT,
                "faces": len(FACE_LOOPS),
                "solids": 2,
                "volumes_total": 3,
            },
            "slivers": {},
            "arc_edges": 0,
        },
        "entity": {
            "substrate": _entity("dielectric"),
            "vacuum": _entity("vacuum"),
            "pad": _entity("conductor", net="N"),
        },
        "source_curve": {},
        "inset_request": {},
        "refinement_request": {},
        "port": {},
        "band_family": {},
        "support_ring": {},
        "support_primitive": {},
        "carrier": {},
        "vertex": {
            f"v{i}": {"position_um": [float(i), 0.0, 0.0]} for i in range(VERTEX_COUNT)
        },
        "edge": {
            f"e{i}": {
                "kind": "horizontal",
                "vertices": [f"v{i % VERTEX_COUNT}", f"v{(i + 1) % VERTEX_COUNT}"],
                "arc": None,
                "length_um": 1.0,
                "carriers": [],
                "entities": [],
                "written_as_free_edge": False,
            }
            for i in range(EDGE_COUNT)
        },
        "face": {
            face: _face(
                "MA::pad__sidewall"
                if face != "f2"
                else "SA::substrate|VACUUM__interface",
                edges,
            )
            for face, edges in FACE_LOOPS.items()
        },
        "volume": {
            "vol0": _volume("substrate", "dielectric", "substrate", list(FACE_LOOPS)),
            "vol1": _volume("vacuum", "vacuum", "vacuum", []),
            "vol2": _volume("pad", "conductor", None, []),
        },
        "group": {
            "MA::pad__sidewall": _surface_group(1, "MA", ["pad"], "sidewall"),
            "SA::substrate|VACUUM__interface": _surface_group(
                2, "SA", ["substrate", "vacuum"], "interface"
            ),
            "substrate": _volume_group(1, "substrate", "dielectric"),
            "vacuum": _volume_group(2, "vacuum", "vacuum"),
        },
        "surface_epr_row": {},
        "mesh_feature": {},
    }
    for number, entry in enumerate(features):
        _add_feature(document, f"mf{number:02d}", entry)
    for face, host in embedded_faces:
        document["volume"][host]["embedded_faces"].append(face)
    for edge, host in embedded_edges:
        document["face"][host]["embedded_edges"].append(edge)
    return document


def _entity(kind: str, net: str | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "material": kind,
        "net": net,
        "layer": None,
        "z_um": [0.0, 0.2],
        "feature_kind": None,
    }


def _face(group: str, edges: Sequence[str]) -> dict[str, Any]:
    return {
        "group": group,
        "kind": "wall",
        "face_kind": "interface" if group.startswith("SA::") else "sidewall",
        "z_um": [0.0, 0.2],
        "negative_volume": "vol0",
        "positive_volume": "vol1",
        "loops": [{"edges": list(edges), "reversed": [False] * len(edges)}],
        "area_um2": 1.0,
        "curved": False,
        "entities": [],
        "families": [],
        "refinements": [],
        "membership": {},
        "bands": [],
        "embedded_edges": [],
        "outer_boundary": None,
    }


def _volume(
    entity: str, kind: str, group: str | None, faces: list[str]
) -> dict[str, Any]:
    return {
        "entity": entity,
        "kind": kind,
        "material": kind,
        "net": "N" if kind == "conductor" else None,
        "emitted": group is not None,
        "group": group,
        "shells": 1,
        "volume_um3": 1.0,
        "aedt_name": entity if group is not None else None,
        "faces": faces,
        "embedded_faces": [],
    }


def _surface_group(
    tag: int, interface_type: str, entities: list[str], face_kind: str
) -> dict[str, Any]:
    return {
        "dim": 2,
        "tag": tag,
        "interface_type": interface_type,
        "entities": entities,
        "net": None,
        "face_kind": face_kind,
        "adjacent_domains": [],
        "adjacent_volumes": [],
        "band": None,
        "area_um2": 1.0,
    }


def _volume_group(tag: int, entity: str, kind: str) -> dict[str, Any]:
    return {
        "dim": 3,
        "tag": tag,
        "interface_type": None,
        "entities": [entity],
        "kind": kind,
        "material": kind,
        "net": None,
    }


def _add_feature(
    document: dict[str, Any], feature_id: str, entry: dict[str, Any]
) -> None:
    """Add a feature and a record for its source in the table its kind names."""
    record = {key: value for key, value in entry.items() if key != "native_tags"}
    document["mesh_feature"][feature_id] = record
    source, table = record.get("source"), SOURCE_TABLE.get(record.get("kind", ""))
    if source is None or table is None or source in document[table]:
        return
    if table == "entity":
        document["entity"][source] = _entity("conductor", net="N")
    elif table == "band_family":
        document["band_family"][source] = {
            "support": {"entity": "pad", "side": "top"},
            "interface": None,
            "margins_um": [0.5],
            "margins_grid": [500],
            "band_count": 2,
            "support_area_um2": None,
            "bands": {},
        }
    elif table == "refinement_request":
        document["refinement_request"][source] = {
            "levels_um": [0.0],
            "points_um": [[0.0, 0.0], [1.0, 0.0]],
            "segments_um": [[[0.0, 0.0], [1.0, 0.0]]],
        }
    else:
        document["port"][source] = {
            "entity": "pad",
            "group": "MA::pad__sidewall",
            "source_name": source,
            "port_index": len(document["port"]) + 1,
            "direction": [1.0, 0.0, 0.0],
            "host_entity_ids": ["pad"],
            "source_layer": "metal",
            "target_layer": "metal",
            "z_um": 0.0,
            "faces": [],
            "host_volumes": [],
            "adjacent_volumes": [],
        }


def feature(
    kind: str,
    tags: list[int],
    facts: dict[str, Any] | None = None,
    source: str | None = "src",
) -> dict[str, Any]:
    """A ``mesh_feature`` record over the edges with the given native tags."""
    return {
        "kind": kind,
        "source": source,
        "edges": [f"e{t - 1}" for t in tags],
        "facts": facts or {},
    }


def film(tags: list[int], thickness_um: float = 0.2) -> dict[str, Any]:
    """A ``film_sidewall`` entry."""
    return feature("film_sidewall", tags, {"thickness_um": thickness_um})


def features_of(*entries: dict[str, Any]) -> SidecarFeatures:
    """The parsed features of a synthetic sidecar with ``entries``."""
    return read_features(sidecar_with(*entries))


def bad_element(  # noqa: PLR0913 - a test builder with independent switches
    element: int,
    candidates: Sequence[tuple[int, float, float]] = (),
    *,
    edge_max: float = 9.0,
    edge_min: float = 0.2,
    location: str = "curve",
    quality: float = 0.001,
    at: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BadElement:
    """A bad element thinnest ``at``, with candidate curves given as (tag, local edge, curve length)."""
    return BadElement(
        element=element,
        quality=quality,
        edge_min=edge_min,
        edge_max=edge_max,
        centroid=at,
        shortest_edge_midpoint=at,
        location=location,
        points=(),
        curves=tuple(t for t, _, _ in candidates),
        surfaces=(),
        volumes=(),
        sgb_ids=(),
        groups=("MA::pad__sidewall",),
        feature_kinds=("film_sidewall",),
        candidates=tuple(
            CurveCandidate(t, edge, length) for t, edge, length in candidates
        ),
    )


def audit_of(
    bad: tuple[BadElement, ...] = (), *, tetrahedra: int = 1000, nodes: int = 300
) -> MeshAudit:
    """An audit whose bad count is that of ``bad``."""
    return MeshAudit(
        tetrahedra=tetrahedra,
        nodes=nodes,
        min_quality=min((b.quality for b in bad), default=0.3),
        min_gamma=0.2,
        bad_count=len(bad),
        bad=bad,
        mesh_seconds=0.5,
        audit_seconds=0.1,
    )


def outcome_of(audit: MeshAudit | None, *, algorithm_3d: int = 10) -> RoundOutcome:
    """A round outcome around ``audit`` (``None`` for a round that made no mesh)."""
    return RoundOutcome(
        audit=audit,
        algorithm_3d=algorithm_3d,
        fallback=None,
        failure=None if audit else "algorithm 10: worker killed by signal 6",
        wall_seconds=0.01,
    )
