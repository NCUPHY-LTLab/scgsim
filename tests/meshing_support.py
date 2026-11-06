"""Synthetic sidecars, audits and rounds for the meshing tests, so most of them need no Gmsh."""

from __future__ import annotations

from typing import Any

from scgsim.meshing.driver import RoundOutcome
from scgsim.meshing.features import SidecarFeatures, read_features
from scgsim.meshing.findings import BadElement, CurveCandidate, MeshAudit

EDGE_COUNT = 12


def sidecar_with(*features: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A minimal v2 sidecar: twelve edges e0..e11, three faces, a substrate and an unemitted conductor ``pad``."""
    document: dict[str, Any] = {
        "schema": "scgsim.sgb.sidecar.v2",
        "binding": {
            "ids_by_tag": {
                "0": [f"v{i}" for i in range(8)],
                "1": [f"e{i}" for i in range(EDGE_COUNT)],
                "2": ["f0", "f1", "f2"],
                "3": ["vol0", "vol1"],
            }
        },
        "physical_groups": {
            "MA::pad__sidewall": {
                "dim": 2,
                "ids": ["f0", "f1"],
                "native_tags": [1, 2],
            },
            "SA::substrate|VACUUM__interface": {
                "dim": 2,
                "ids": ["f2"],
                "native_tags": [3],
            },
            "substrate": {"dim": 3, "ids": ["vol0"], "native_tags": [1]},
        },
        "volumes": {
            "vol0": {"entity": "substrate", "kind": "dielectric", "emitted": True},
            "vol1": {"entity": "pad", "kind": "conductor", "emitted": False},
        },
        "mesh_features": list(features),
        "embedded_faces": [],
        "refinement_edges": [],
    }
    document.update(extra)
    return document


def feature(
    kind: str,
    tags: list[int],
    facts: dict[str, Any] | None = None,
    source: str | None = "src",
) -> dict[str, Any]:
    """A ``mesh_features`` entry over the edges with the given native tags."""
    return {
        "kind": kind,
        "source": source,
        "edges": [f"e{t - 1}" for t in tags],
        "native_tags": tags,
        "facts": facts or {},
    }


def film(tags: list[int], thickness_um: float = 0.2) -> dict[str, Any]:
    """A ``film_sidewall`` entry."""
    return feature("film_sidewall", tags, {"thickness_um": thickness_um})


def features_of(*entries: dict[str, Any]) -> SidecarFeatures:
    """The parsed features of a synthetic sidecar with ``entries``."""
    return read_features(sidecar_with(*entries))


def bad_element(
    element: int,
    candidates: list[tuple[int, float, float]],
    *,
    edge_max: float = 9.0,
    location: str = "curve",
    quality: float = 0.001,
) -> BadElement:
    """A bad element with candidate curves given as (tag, local edge, curve length)."""
    return BadElement(
        element=element,
        quality=quality,
        edge_min=0.2,
        edge_max=edge_max,
        centroid=(0.0, 0.0, 0.0),
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
