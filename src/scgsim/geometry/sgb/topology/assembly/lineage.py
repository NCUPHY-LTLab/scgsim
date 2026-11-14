"""Which source entity a carrier id belongs to (lineage back to the source).

Owns the two lookups the assembly needs to turn the carrier ids on an edge into the
ids a face reports.  What drew an edge (``owners_of``, a face's ``sources``): polygon-edge
and arc carriers name their Entity, cut-family members name their family.  Which Entities
an edge belongs to (``entities_of``, a wall's membership and its columns): the source
carriers' Entities only, since a cut-family member (a Surface-EPR band seam, say) draws a
curve but is no Entity.  The catalogue's table of known carriers
(``PrismCatalogue.known_carriers``) says what each id is; no id is read.
"""

from __future__ import annotations

from scgsim.geometry.exact.carriers import KnownCarrier
from scgsim.geometry.sgb.records.prisms import PrismCatalogue


class CarrierEntities:
    """Maps a carrier id (a witness chord ``<arc>#k`` included) to its source entity or family."""

    def __init__(self, catalogue: PrismCatalogue) -> None:
        known = catalogue.known_carriers()
        self._owner = {
            carrier_id: _owner(carrier) for carrier_id, carrier in known.items()
        }
        self._family_members = frozenset(
            carrier_id
            for carrier_id, carrier in known.items()
            if carrier.lineage.kind == "family_member"
        )

    def owner_of(self, carrier_id: str) -> str:
        """The entity (or family) id of a carrier; a carrier the catalogue does not know reports its own id."""
        return self._owner.get(carrier_id, carrier_id)

    def owners_of(self, carrier_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """The sorted distinct entity and family ids of a set of carriers: what drew them."""
        return tuple(sorted({self.owner_of(c) for c in carrier_ids}))

    def entities_of(self, carrier_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """The sorted distinct entity ids of a set of carriers; a cut-family member names none."""
        return self.owners_of([c for c in carrier_ids if c not in self._family_members])


def _owner(known: KnownCarrier) -> str:
    """A source carrier belongs to its Entity, a cut-family member to its family."""
    lineage = known.lineage
    return lineage.family_id if lineage.kind == "family_member" else lineage.entity_id
