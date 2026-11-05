"""Geometry facts of an SGB sidecar, as typed frozen records.

Owns the reading and validation of the sidecar JSON that the meshing module consumes: the schema name,
the ``mesh_features`` list, the planned-id binding, the primary physical groups and the embeddings that
Gmsh must re-apply after loading the XAO.  It imports nothing from SGB: the sidecar JSON is the whole
contract, so a sidecar from any worker count, tile layout or machine reads the same way.

A feature states what is true of a set of edges (this set is the outline of a 0.2 um film wall); it never
states a mesh size.  Sizes come from the rule set in ``policy``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import SidecarError

SIDECAR_SCHEMA = "scgsim.sgb.sidecar.v2"

# Facts every entry of a kind must carry; a sidecar may add facts later, never drop one of these.
_REQUIRED_FACTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "film_sidewall": ("thickness_um",),
        "band_boundary": ("margins_um", "narrowest_band_um"),
        "arc": ("radius_um",),
        "refinement_curve": (),
        "port_edge": (),
    }
)
FEATURE_KINDS = tuple(_REQUIRED_FACTS)
_EDGE_DIM = 1
_DIMENSIONS = (0, 1, 2, 3)


@dataclass(frozen=True)
class Feature:
    """One ``mesh_features`` entry: a kind, its source and the curves it covers.

    ``native_tags`` are Gmsh curve tags of the XAO (tag = 1 + position in the binding's edge list), and
    ``facts`` holds the numbers of that kind, for example ``thickness_um`` of a ``film_sidewall``.
    """

    kind: str
    source: str | None
    edge_ids: tuple[str, ...]
    native_tags: tuple[int, ...]
    facts: Mapping[str, Any]

    def fact(self, name: str) -> float:
        """The numeric fact ``name`` (validated to exist for the required facts of the kind)."""
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

    def features_by_curve(self) -> dict[int, tuple[Feature, ...]]:
        """The features that cover each native curve tag, in sidecar order."""
        covering: dict[int, list[Feature]] = {}
        for feature in self.features:
            for tag in feature.native_tags:
                covering.setdefault(tag, []).append(feature)
        return {tag: tuple(found) for tag, found in sorted(covering.items())}


def load_features(sidecar_path: Path) -> SidecarFeatures:
    """Read and validate the sidecar JSON file at ``sidecar_path``."""
    document = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    return read_features(document)


def read_features(sidecar: Mapping[str, Any]) -> SidecarFeatures:
    """Validate a decoded sidecar and return its features, binding, groups and embeddings.

    Raises :class:`SidecarError` for another schema name, an unknown feature kind, a missing fact or an
    entry whose edge ids disagree with the binding.
    """
    _require_schema(sidecar)
    binding = _read_binding(sidecar.get("binding"))
    entries = _require_list(sidecar.get("mesh_features"), "mesh_features")
    return SidecarFeatures(
        features=tuple(_read_feature(entry, binding) for entry in entries),
        binding=binding,
        groups=_read_groups(sidecar.get("physical_groups")),
        embedded_faces=_read_embedded_faces(sidecar, binding),
        embedded_edges=_read_embedded_edges(sidecar, binding),
    )


def _require_schema(sidecar: Mapping[str, Any]) -> None:
    schema = sidecar.get("schema")
    if schema != SIDECAR_SCHEMA:
        raise SidecarError(
            f"sidecar schema {schema!r} is not supported; the meshing module reads {SIDECAR_SCHEMA!r}."
        )


def _read_binding(raw: Any) -> Binding:
    ids_by_tag = _require_mapping(
        _require_mapping(raw, "binding").get("ids_by_tag"), "binding.ids_by_tag"
    )
    ids = []
    for dimension in _DIMENSIONS:
        entries = _require_list(
            ids_by_tag.get(str(dimension)), f"binding.ids_by_tag[{dimension}]"
        )
        ids.append(tuple(_require_string(i, "binding id") for i in entries))
    return Binding(tuple(ids))


def _read_feature(raw: Any, binding: Binding) -> Feature:
    entry = _require_mapping(raw, "mesh_features entry")
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind not in _REQUIRED_FACTS:
        raise SidecarError(
            f"mesh_features kind {kind!r} is unknown; expected one of {', '.join(FEATURE_KINDS)}."
        )
    source = entry.get("source")
    if source is not None:
        source = _require_string(source, f"{kind} source")
    edge_ids = tuple(
        _require_string(i, f"{kind} edge")
        for i in _require_list(entry.get("edges"), f"{kind} edges")
    )
    native_tags = tuple(
        _require_tag(t, f"{kind} native tag")
        for t in _require_list(entry.get("native_tags"), f"{kind} native_tags")
    )
    _require_edges_bound(kind, edge_ids, native_tags, binding)
    facts = _read_facts(kind, entry.get("facts"))
    return Feature(kind, source, edge_ids, native_tags, MappingProxyType(facts))


def _require_edges_bound(
    kind: str, edge_ids: Sequence[str], native_tags: Sequence[int], binding: Binding
) -> None:
    """The edge ids and the native tags of an entry must name the same curves, by the binding rule."""
    if len(edge_ids) != len(native_tags):
        raise SidecarError(
            f"{kind} entry has {len(edge_ids)} edges but {len(native_tags)} native tags."
        )
    for edge_id, tag in zip(edge_ids, native_tags, strict=True):
        if binding.id_of(_EDGE_DIM, tag) != edge_id:
            raise SidecarError(
                f"{kind} entry: native tag {tag} is not edge {edge_id!r} in binding.ids_by_tag."
            )


def _read_facts(kind: str, raw: Any) -> dict[str, Any]:
    facts = dict(_require_mapping(raw, f"{kind} facts"))
    for name in _REQUIRED_FACTS[kind]:
        if name not in facts:
            raise SidecarError(f"{kind} entry lacks the fact {name!r}.")
    for name in ("thickness_um", "radius_um", "narrowest_band_um"):
        if name in facts:
            _require_positive(facts[name], f"{kind} {name}")
    if "margins_um" in facts:
        _require_margins(facts["margins_um"], kind)
    return facts


def _require_margins(margins: Any, kind: str) -> None:
    values = _require_list(margins, f"{kind} margins_um")
    for value in values:
        _require_positive(value, f"{kind} margins_um entry")
    if not values or any(a >= b for a, b in zip(values, values[1:], strict=False)):
        raise SidecarError(f"{kind} margins_um must be a non-empty ascending list.")


def _read_groups(raw: Any) -> tuple[PhysicalGroup, ...]:
    groups = _require_mapping(raw, "physical_groups")
    return tuple(
        PhysicalGroup(
            name,
            int(_require_mapping(group, f"physical group {name}")["dim"]),
            tuple(_require_list(group.get("ids", []), f"physical group {name} ids")),
        )
        for name, group in sorted(groups.items())
    )


def _read_embedded_faces(
    sidecar: Mapping[str, Any], binding: Binding
) -> tuple[EmbeddedFace, ...]:
    tags = _tags_by_id(binding)
    return tuple(
        EmbeddedFace(
            _tag_of(tags, 2, entry.get("face")),
            _tag_of(tags, 3, entry.get("host_volume")),
        )
        for entry in _require_list(sidecar.get("embedded_faces", []), "embedded_faces")
    )


def _read_embedded_edges(
    sidecar: Mapping[str, Any], binding: Binding
) -> tuple[EmbeddedEdge, ...]:
    tags = _tags_by_id(binding)
    return tuple(
        EmbeddedEdge(
            _tag_of(tags, 1, entry.get("edge")),
            _tag_of(tags, 2, entry.get("host_face")),
        )
        for entry in _require_list(
            sidecar.get("refinement_edges", []), "refinement_edges"
        )
    )


def _tags_by_id(binding: Binding) -> dict[int, dict[str, int]]:
    return {
        dimension: {entity_id: tag for tag, entity_id in enumerate(ids, start=1)}
        for dimension, ids in enumerate(binding.ids_by_dimension)
    }


def _tag_of(
    tags: Mapping[int, Mapping[str, int]], dimension: int, entity_id: Any
) -> int:
    if not isinstance(entity_id, str) or entity_id not in tags[dimension]:
        raise SidecarError(
            f"embedding names {entity_id!r}, which is not a dimension-{dimension} id in the binding."
        )
    return tags[dimension][entity_id]


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SidecarError(f"sidecar {what} must be an object.")
    return value


def _require_list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise SidecarError(f"sidecar {what} must be a list.")
    return value


def _require_string(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise SidecarError(f"sidecar {what} must be a string.")
    return value


def _require_tag(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SidecarError(f"sidecar {what} must be a positive integer.")
    return value


def _require_positive(value: Any, what: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SidecarError(f"sidecar {what} must be a positive number.")
    return float(value)
