"""Geometry facts of an SGB sidecar, as typed frozen records.

Owns what the meshing module reads from a sidecar: the ``mesh_feature`` table, the planned-id binding, the
primary physical groups with their members and the embeddings that Gmsh must re-apply after loading the XAO.
The sidecar is checked first by the shared reader of its contract (``scgsim.contracts.sgb_sidecar``: version,
structure and every reference), and its views give the native tags and orders.  It imports nothing from SGB:
the sidecar JSON is the whole contract, so a sidecar from any worker count, tile layout or machine reads the
same way.

A feature states what is true of a set of edges (this set is the outline of a 0.2 um film wall); it never
states a mesh size.  Sizes come from a sizing recipe (``recipes``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scgsim.contracts.sgb_sidecar import Sidecar, check_sidecar, read_sidecar, views
from scgsim.contracts.sgb_sidecar import SidecarError as ContractError
from scgsim.contracts.sgb_sidecar.records import MeshFeature

from .errors import SidecarError

FEATURE_KINDS = (
    "film_sidewall",
    "band_boundary",
    "arc",
    "refinement_curve",
    "port_edge",
)
_EDGE_DIM = 1


@dataclass(frozen=True)
class Feature:
    """One mesh feature: a kind, its source and the curves it covers.

    ``native_tags`` are Gmsh curve tags of the XAO (tag = 1 + position in the binding's edge list), and
    ``facts`` holds the numbers of that kind, for example ``thickness_um`` of a ``film_sidewall``.
    """

    kind: str
    source: str | None
    edge_ids: tuple[str, ...]
    native_tags: tuple[int, ...]
    facts: Mapping[str, Any]

    def fact(self, name: str) -> float:
        """The numeric fact ``name`` (the contract requires every fact of the kind)."""
        return float(self.facts[name])


@dataclass(frozen=True)
class Binding:
    """Planned id of every entity by dimension; the native tag is ``1 + position`` in its list."""

    ids_by_dimension: tuple[tuple[str, ...], ...]

    def id_of(self, dimension: int, tag: int) -> str | None:
        """The planned id of native entity ``(dimension, tag)``, or ``None`` when the tag is unknown."""
        ids = self.ids_by_dimension[dimension]
        return ids[tag - 1] if 1 <= tag <= len(ids) else None


@dataclass(frozen=True)
class PhysicalGroup:
    """A primary physical group of the sidecar: its name, dimension and member planned ids."""

    name: str
    dimension: int
    ids: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddedFace:
    """A sheet face (native tag) that lies inside the volume ``host_volume`` and must be embedded in it."""

    face: int
    host_volume: int


@dataclass(frozen=True)
class EmbeddedEdge:
    """A refinement-only curve (native tag) that lies in the face ``host_face`` and must be embedded in it."""

    edge: int
    host_face: int


@dataclass(frozen=True)
class SidecarFeatures:
    """Everything the meshing module reads from a sidecar."""

    features: tuple[Feature, ...]
    binding: Binding
    groups: tuple[PhysicalGroup, ...]
    embedded_faces: tuple[EmbeddedFace, ...]
    embedded_edges: tuple[EmbeddedEdge, ...]
    # Ids of the entities whose volume is a conductor, emitted or not: a feature's ``source`` is one of
    # these when a conductor owns the feature (a dielectric or the vacuum is not in the set).
    conductors: frozenset[str] = frozenset()

    def features_by_curve(self) -> dict[int, tuple[Feature, ...]]:
        """The features that cover each native curve tag, in sidecar order."""
        covering: dict[int, list[Feature]] = {}
        for feature in self.features:
            for tag in feature.native_tags:
                covering.setdefault(tag, []).append(feature)
        return {tag: tuple(found) for tag, found in sorted(covering.items())}


def load_features(sidecar_path: Path) -> SidecarFeatures:
    """Read and check the sidecar JSON file at ``sidecar_path``."""
    return _features(load_checked(sidecar_path))


def load_checked(sidecar_path: Path | str) -> Sidecar:
    """The sidecar in the file ``sidecar_path`` once the shared reader accepts it, else :class:`SidecarError`."""
    try:
        return read_sidecar(sidecar_path)
    except ContractError as error:
        raise SidecarError(str(error)) from None


def read_features(sidecar: Mapping[str, Any]) -> SidecarFeatures:
    """Check a decoded sidecar and return its features, binding, groups and embeddings.

    Raises :class:`SidecarError` for a sidecar the shared reader refuses (another version, a structure the
    contract does not describe, an id that names no record) or a band boundary whose margins do not ascend.
    """
    return _features(checked(sidecar))


def checked(sidecar: Mapping[str, Any]) -> Sidecar:
    """``sidecar`` once the shared reader accepts it, else :class:`SidecarError` with its reason."""
    try:
        return check_sidecar(sidecar)
    except ContractError as error:
        raise SidecarError(str(error)) from None


def _features(sidecar: Sidecar) -> SidecarFeatures:
    index = views.TagIndex.of(sidecar)
    by_tag = sidecar["binding"]["ids_by_tag"]
    members = views.group_members(sidecar)
    return SidecarFeatures(
        features=tuple(_feature(f, index) for f in views.mesh_features(sidecar)),
        binding=Binding(tuple(tuple(by_tag[d]) for d in ("0", "1", "2", "3"))),
        groups=tuple(
            PhysicalGroup(name, group["dim"], tuple(members[name]))
            for name, group in sorted(sidecar["group"].items())
        ),
        embedded_faces=tuple(
            EmbeddedFace(index.tag(2, face), index.tag(3, host))
            for face, host in views.embedded_faces(sidecar)
        ),
        embedded_edges=tuple(
            EmbeddedEdge(index.tag(1, edge), index.tag(2, host))
            for edge, host in views.embedded_edges(sidecar)
        ),
        conductors=frozenset(
            volume["entity"]
            for volume in sidecar["volume"].values()
            if volume["kind"] == "conductor"
        ),
    )


def _feature(feature: MeshFeature, index: views.TagIndex) -> Feature:
    facts: dict[str, Any] = dict(feature["facts"])
    if feature["kind"] == "band_boundary":
        _require_ascending(facts["margins_um"])
    edges = tuple(feature["edges"])
    return Feature(
        feature["kind"],
        feature["source"],
        edges,
        tuple(index.tags_of(_EDGE_DIM, edges)),
        MappingProxyType(facts),
    )


def _require_ascending(margins: list[float]) -> None:
    """The band recipes size each band from the margin below it, so the margins must ascend."""
    if any(a >= b for a, b in zip(margins, margins[1:], strict=False)):
        raise SidecarError(
            f"sidecar band_boundary margins_um {margins} must be an ascending list."
        )
