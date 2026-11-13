"""Which source entity a carrier id belongs to (lineage back to the source).

Owns the one lookup the assembly needs to turn the carrier ids on an edge into the
entity ids a face reports: polygon-edge carriers and arc carriers name their entity in
their lineage, cut-family members name their family.  The catalogue's table of known
carriers (``PrismCatalogue.known_carriers``) says what each id is; no id is read.
"""

from __future__ import annotations

from scgsim.geometry.sgb.exact.carriers import KnownCarrier
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue


class CarrierEntities:
    """Maps a carrier id (a witness chord ``<arc>#k`` included) to its source entity or family."""

    def __init__(self, catalogue: PrismCatalogue) -> None:
        self._owner = {
            carrier_id: _owner(known)
            for carrier_id, known in catalogue.known_carriers().items()
        }

    def entity_of(self, carrier_id: str) -> str:
        """The entity (or family) id of a carrier; a carrier the catalogue does not know reports its own id."""
        return self._owner.get(carrier_id, carrier_id)

    def entities_of(self, carrier_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """The sorted distinct entity ids of a set of carriers."""
        return tuple(sorted({self.entity_of(c) for c in carrier_ids}))


def _owner(known: KnownCarrier) -> str:
    """A source carrier belongs to its Entity, a cut-family member to its family."""
    lineage = known.lineage
    return lineage.family_id if lineage.kind == "family_member" else lineage.entity_id
