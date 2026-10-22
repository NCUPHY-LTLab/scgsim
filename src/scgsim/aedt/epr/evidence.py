"""Shared file-evidence interpretation for prepared EPR control and saved cache.

This private codec preserves the existing receipt fields. Saved configuration is
an observation of configuration, independent of adaptive convergence results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from scgsim.aedt.epr.fields import validated_expression_readback


type EvidenceValue = (
    None | bool | int | float | str
    | tuple[EvidenceValue, ...] | Mapping[str, EvidenceValue]
)


def _freeze(value: object) -> EvidenceValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        entries = cast(Mapping[object, object], value)
        frozen: dict[str, EvidenceValue] = {}
        for key, item in entries.items():
            if not isinstance(key, str):
                raise TypeError("prepared evidence mapping keys must be strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in cast(Sequence[object], value))
    raise TypeError(f"prepared evidence contains unsupported {type(value).__name__}")


def _mapping(value: object, label: str) -> Mapping[str, EvidenceValue]:
    frozen = _freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError(f"prepared {label} must be a mapping")
    return frozen


def _records(value: object, label: str) -> tuple[Mapping[str, EvidenceValue], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"prepared {label} must be a sequence")
    return tuple(_mapping(item, label) for item in cast(Sequence[object], value))


def _thaw(value: EvidenceValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _payload(value: Mapping[str, EvidenceValue]) -> dict[str, object]:
    return cast(dict[str, object], _thaw(value))


@dataclass(frozen=True)
class SavedCacheEvidence:
    """Submitted cache requests and saved observations, with no convergence claim."""

    submitted_items: tuple[Mapping[str, EvidenceValue], ...]
    observation: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "submitted_items", _records(self.submitted_items, "cache items"))
        object.__setattr__(self, "observation", _mapping(self.observation, "saved cache"))

    @classmethod
    def from_payload(cls, submitted_items: object, observation: object) -> SavedCacheEvidence:
        submitted = _records(submitted_items, "cache items")
        observed = _mapping(observation, "saved cache")
        if not submitted:
            raise RuntimeError("native preparation lacks submitted cache items")
        source = observed.get("saved_configuration_source")
        if not isinstance(source, Mapping) or not source:
            raise RuntimeError("native preparation lacks fresh saved cache readback")
        expected = tuple(
            {"title": item["title"], "expression": item["expression"], "intrinsics": item["intrinsics"]}
            for item in submitted
        )
        # This is the producer's persisted-subset check, shared by later readers.
        # Optional observed convergence/frequency fields remain untouched.
        if observed.get("use_cache_for") != ("Pass",) or observed.get("items") != expected:
            raise RuntimeError("saved EPR cache persisted-subset readback mismatch")
        return cls(submitted, observed)

    def to_payload(self) -> dict[str, object]:
        return _payload(self.observation)


@dataclass(frozen=True)
class PreparedEprEvidence:
    """One interpretation of setup requests, saved cache and required CLC closure."""

    setup: Mapping[str, EvidenceValue]
    cache: Mapping[str, EvidenceValue]
    expressions: tuple[Mapping[str, EvidenceValue], ...]
    saved_cache: SavedCacheEvidence
    required_definitions: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup", _mapping(self.setup, "setup"))
        object.__setattr__(self, "cache", _mapping(self.cache, "cache"))
        object.__setattr__(self, "expressions", _records(self.expressions, "expressions"))
        object.__setattr__(self, "required_definitions", _mapping(self.required_definitions, "CLC definitions"))

    @classmethod
    def from_payload(cls, receipt: object, *, evidence_root: Path) -> PreparedEprEvidence:
        payload = _mapping(receipt, "receipt")
        setup = _mapping(payload.get("setup"), "setup")
        cache = _mapping(payload.get("cache"), "cache")
        expressions = _records(payload.get("expressions"), "expressions")
        if not setup or not cache or not expressions:
            raise RuntimeError("native preparation lacks setup/cache/expression readback")
        saved = SavedCacheEvidence.from_payload(cache.get("items"), cache.get("serialized_readback"))
        # Keep the historical CLC codec's list/dict wire representation.
        expression_payloads = [cast(dict[str, object], _thaw(item)) for item in expressions]
        required = validated_expression_readback(
            expression_payloads, _payload(cache), evidence_root=evidence_root,
        )
        return cls(setup, cache, expressions, saved, _mapping(required, "CLC definitions"))

    def to_payload(self) -> dict[str, object]:
        return {
            "setup": _payload(self.setup), "cache": self.cache_payload(),
            "expressions": self.expression_payloads(),
        }

    def cache_payload(self) -> dict[str, object]:
        return _payload(self.cache)

    def expression_payloads(self) -> list[dict[str, object]]:
        return [_payload(item) for item in self.expressions]

    def required_definitions_payload(self) -> dict[str, object]:
        return _payload(self.required_definitions)
