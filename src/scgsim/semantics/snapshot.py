"""Normalize raw semantic facts into immutable, revision-bound snapshots.

Input fact mappings deliberately keep arbitrary nested values at the boundary;
``to_plain`` validates that shape before snapshots are hashed or stored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

from scgsim.refusal import found_text, internal

RevisionKind = Literal["declared_revision", "content_hash_fallback"]


def to_plain(value: Any) -> Any:
    """Return detached plain values in the canonical container form."""
    if isinstance(value, Mapping):
        plain: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    internal(
                        f"semantic fact mapping keys must be strings, got {type(key).__name__}"
                    )
                )
            plain[key] = to_plain(item)
        return plain
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_plain(item) for item in value), key=repr)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        internal(f"semantic facts require plain values, got {type(value).__name__}")
    )


def _freeze(value: Any) -> Any:
    """Recursively replace mutable containers with immutable equivalents."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def freeze_plain(value: Any) -> Any:
    """Detach a plain-data value and freeze each nested mapping and sequence."""
    return _freeze(to_plain(value))


def _detached(value: Any) -> Any:
    """Copy a frozen fact tree back into ordinary dictionaries and lists."""
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detached(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    """Hash deterministic compact JSON for a supported plain-data value."""
    return _canonical_sha256_plain(to_plain(value))


def _canonical_sha256_plain(value: Any) -> str:
    """Hash a value already normalized by ``to_plain``."""
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validate_sha256_digest(name: str, value: str) -> None:
    """Reject digests that are not lowercase SHA-256 hexadecimal strings."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            internal(f"{name} must be 64 lowercase hexadecimal characters")
        )


def _validate_revision(name: str, value: str, kind: RevisionKind) -> None:
    """Validate a declared revision or a content-hash fallback identifier."""
    if not isinstance(value, str) or not value:
        raise ValueError(internal(f"{name} must be a non-empty string"))
    if kind not in ("declared_revision", "content_hash_fallback"):
        raise ValueError(
            internal(f"{name}_kind must be declared_revision or content_hash_fallback")
        )
    if kind == "content_hash_fallback":
        prefix, separator, digest = value.partition(":")
        if (
            prefix != "sha256"
            or separator != ":"
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                internal(
                    f"{name} marked content_hash_fallback must be sha256:<64 lowercase hex>"
                )
            )


@dataclass(frozen=True, slots=True)
class SemanticSnapshotReference:
    """Immutable identity of the exact facts used for one classification."""

    source_revision: str
    source_revision_kind: RevisionKind
    geometry_revision: str
    geometry_revision_kind: RevisionKind
    projection_version: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        _validate_revision(
            "source_revision", self.source_revision, self.source_revision_kind
        )
        _validate_revision(
            "geometry_revision", self.geometry_revision, self.geometry_revision_kind
        )
        if not isinstance(self.projection_version, str) or not self.projection_version:
            raise ValueError(
                internal(
                    "projection_version must be a non-empty string"
                    + ": "
                    + found_text(self.projection_version)
                )
            )
        _validate_sha256_digest("canonical_sha256", self.canonical_sha256)

    def detached(self) -> dict[str, str]:
        """Return a mutable record that can be serialized independently."""
        return {
            "source_revision": self.source_revision,
            "source_revision_kind": self.source_revision_kind,
            "geometry_revision": self.geometry_revision,
            "geometry_revision_kind": self.geometry_revision_kind,
            "projection_version": self.projection_version,
            "canonical_sha256": self.canonical_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticFactsSnapshot:
    """Validated identity and recursively immutable facts for one projection."""

    source_revision: str
    source_revision_kind: RevisionKind
    geometry_revision: str
    geometry_revision_kind: RevisionKind
    projection_version: str
    canonical_sha256: str
    facts: Mapping[str, Any]

    def __post_init__(self) -> None:
        plain = to_plain(self.facts)
        digest = _canonical_sha256_plain(plain)
        _validate_sha256_digest("canonical_sha256", self.canonical_sha256)
        if self.canonical_sha256 != digest:
            raise ValueError(
                internal(
                    "canonical_sha256 does not match semantic facts"
                    + ": "
                    + found_text(self.canonical_sha256, digest)
                )
            )
        _validate_revision(
            "source_revision", self.source_revision, self.source_revision_kind
        )
        _validate_revision(
            "geometry_revision", self.geometry_revision, self.geometry_revision_kind
        )
        if not isinstance(self.projection_version, str) or not self.projection_version:
            raise ValueError(
                internal(
                    "projection_version must be a non-empty string"
                    + ": "
                    + found_text(self.projection_version)
                )
            )
        object.__setattr__(self, "facts", _freeze(plain))

    def reference(self) -> SemanticSnapshotReference:
        """Build the compact immutable identity for consumers of these facts."""
        return SemanticSnapshotReference(
            source_revision=self.source_revision,
            source_revision_kind=self.source_revision_kind,
            geometry_revision=self.geometry_revision,
            geometry_revision_kind=self.geometry_revision_kind,
            projection_version=self.projection_version,
            canonical_sha256=self.canonical_sha256,
        )

    def detached(self) -> dict[str, Any]:
        """Return a fully independent plain-data representation of the snapshot."""
        return {
            "source_revision": self.source_revision,
            "source_revision_kind": self.source_revision_kind,
            "geometry_revision": self.geometry_revision,
            "geometry_revision_kind": self.geometry_revision_kind,
            "projection_version": self.projection_version,
            "canonical_sha256": self.canonical_sha256,
            "facts": _detached(self.facts),
        }


def create_snapshot(  # noqa: PLR0913 -- preserves independent revision values and revision kinds
    facts: Mapping[str, Any],
    *,
    source_revision: str | None,
    geometry_revision: str | None,
    projection_version: str,
    source_revision_kind: RevisionKind | None = None,
    geometry_revision_kind: RevisionKind | None = None,
) -> SemanticFactsSnapshot:
    """Normalize facts, derive missing revisions, and freeze the resulting snapshot.

    The source and geometry revision values and kinds remain separate because
    either source can be declared independently or fall back to the content hash.
    """
    plain = to_plain(facts)
    digest = _canonical_sha256_plain(plain)
    fallback = f"sha256:{digest}"
    resolved_source_revision, resolved_source_kind = _resolve_revision(
        "source_revision", source_revision, source_revision_kind, fallback
    )
    resolved_geometry_revision, resolved_geometry_kind = _resolve_revision(
        "geometry_revision", geometry_revision, geometry_revision_kind, fallback
    )
    return SemanticFactsSnapshot(
        source_revision=resolved_source_revision,
        source_revision_kind=resolved_source_kind,
        geometry_revision=resolved_geometry_revision,
        geometry_revision_kind=resolved_geometry_kind,
        projection_version=projection_version,
        canonical_sha256=digest,
        facts=_freeze(plain),
    )


class _Revision(NamedTuple):
    """A resolved revision: the identifier and whether it was declared or is a content-hash fallback."""

    value: str
    kind: RevisionKind


def _resolve_revision(
    name: str,
    value: str | None,
    kind: RevisionKind | None,
    fallback: str,
) -> _Revision:
    if value is None:
        if kind not in (None, "content_hash_fallback"):
            raise ValueError(
                internal(
                    f"{name}_kind must be content_hash_fallback when {name} is None"
                )
            )
        return _Revision(fallback, "content_hash_fallback")
    return _Revision(value, kind or "declared_revision")
