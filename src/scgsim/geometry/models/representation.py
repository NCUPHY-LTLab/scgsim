"""Private source/effective representation facts for neutral geometry planning.

These records are an in-process handoff from a preparation owner to shared
planning primitives. They are not source metadata, a serialized schema, or a
backend-specific plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from types import MappingProxyType
from scgsim.semantics.snapshot import freeze_plain


@dataclass(frozen=True, slots=True)
class EntityRepresentation:
    """Physical and effective Z facts for one prepared conductor entity."""

    physical_layer_id: str | None
    representation: Literal["sheet", "solid"]
    source_z_interval_um: tuple[float, float]
    effective_z_interval_um: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_z_interval_um", tuple(self.source_z_interval_um))
        object.__setattr__(self, "effective_z_interval_um", tuple(self.effective_z_interval_um))


@dataclass(frozen=True, slots=True)
class EffectiveEnclosure:
    """Source recipe and already-mapped bounds for one auto enclosure."""

    source_bounds_um: Mapping[str, float]
    effective_bounds_um: Mapping[str, float]
    effective_body_bounds_um: Mapping[str, float]
    source_padding_um: Mapping[str, float]
    effective_padding_um: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, freeze_plain(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RepresentationContext:
    """Prepared representation facts keyed by source semantic identity."""

    entities: Mapping[str, EntityRepresentation]
    enclosures: Mapping[str, EffectiveEnclosure] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", MappingProxyType(dict(self.entities)))
        object.__setattr__(self, "enclosures", MappingProxyType(dict(self.enclosures)))
