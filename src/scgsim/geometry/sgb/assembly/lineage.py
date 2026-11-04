"""Which source entity a carrier id belongs to (lineage back to the source).

Owns the one lookup the assembly needs to turn the carrier ids on an edge into the
entity ids a face reports: polygon-edge carriers and arc carriers name their entity in
their lineage, cut-family members name their family.
"""

from __future__ import annotations

from scgsim.geometry.sgb.records.carriers import base_carrier_id
from scgsim.geometry.sgb.records.prisms import PrismCatalogue


class CarrierEntities:
    """Maps a carrier id (a witness chord ``<arc>#k`` included) to its source entity or family."""

    def __init__(self, catalogue: PrismCatalogue) -> None:
        self._owner: dict[str, str] = {}
        for prism in catalogue.prisms:
            for carrier in prism.carriers():
                self._owner[base_carrier_id(carrier.id)] = carrier.lineage.entity_id
            for arc in prism.arcs:
                self._owner[arc.id] = arc.lineage.entity_id
        for family in catalogue.families:
            for carrier in family.carriers:
                self._owner[carrier.id] = family.family_id
            for arc in family.arcs:
                self._owner[arc.id] = family.family_id

    def entity_of(self, carrier_id: str) -> str:
        """The entity (or family) id of a carrier; unknown carriers report their own id."""
        base = base_carrier_id(carrier_id)
        return self._owner.get(base, self._owner.get(carrier_id, base))

    def entities_of(self, carrier_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """The sorted distinct entity ids of a set of carriers."""
        return tuple(sorted({self.entity_of(c) for c in carrier_ids}))
