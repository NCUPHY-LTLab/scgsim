"""Which inset margins apply to a Surface-EPR face: the spec's own, or those of a customised conductor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def margins_of(spec: Mapping[str, Any], entities: Iterable[str]) -> list[float]:
    """The inset margins (µm) of a face of ``entities`` under ``spec``.

    ``inset_margins_by_entity_um`` customises conductors one by one; every other face takes the spec's
    ``inset_margins_um``.  A spec with neither has no margins.
    """
    by_entity = spec.get("inset_margins_by_entity_um") or {}
    for entity in entities:
        if entity in by_entity:
            return list(by_entity[entity])
    return list(spec.get("inset_margins_um", ()))
