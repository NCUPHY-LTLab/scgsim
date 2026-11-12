"""Where each carrier of a model comes from (design brief section 7), for the sidecar's ``carrier`` table.

Owns the one lookup the writer needs to describe a carrier id that lies on an emitted edge: its lineage (a
source polygon edge or arc, or a member of a cut family) and, for a source carrier, the source curve it lies
on.  SGB names carriers itself, so this is the one place that knows how a witness chord's id (``<arc id>#k``)
or a derived arc relates to the curve it comes from; the sidecar states the relation as a field, and no
reader reads it from an id.
"""

from __future__ import annotations

from dataclasses import dataclass

from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.carriers import ArcCarrier, Lineage, base_carrier_id

BAND_FAMILY_ROLE = "inset_band"
REFINEMENT_FAMILY_ROLE = "refinement"


@dataclass(frozen=True)
class CarrierLineage:
    """The lineage of every carrier id a model knows, the source curves, and the role of each cut family."""

    known: dict[str, Lineage]
    source_curves: frozenset[str]
    derived_arcs: dict[str, ArcCarrier]
    roles: dict[str, str]

    @classmethod
    def of(cls, model: ConformalModel) -> CarrierLineage:
        """The carriers of ``model``: source line carriers and arcs, family members, derived arcs."""
        catalogue = model.labelled.catalogue
        known: dict[str, Lineage] = {}
        sources: set[str] = set()
        for prism in catalogue.prisms:
            for carrier in prism.carriers():
                known[carrier.id] = carrier.lineage
                # A straight source edge is named by its lineage; a witness chord of an arc is not.
                if carrier.id == carrier.lineage.describe():
                    sources.add(carrier.id)
            for arc in prism.arcs:
                known[arc.id] = arc.lineage
                sources.add(arc.id)
        for family in catalogue.families:
            for carrier in family.carriers:
                known[carrier.id] = carrier.lineage
            for arc in family.arcs:
                known[arc.id] = arc.lineage
        for arc_id, arc in model.labelled.arcs.items():
            known.setdefault(arc_id, arc.lineage)
        return cls(
            known,
            frozenset(sources),
            dict(model.labelled.arcs),
            {f.family_id: f.role for f in catalogue.families},
        )

    def lineage(self, carrier_id: str) -> Lineage | None:
        """The lineage of a carrier id, a witness chord's included; ``None`` for a carrier SGB does not know."""
        return self.known.get(carrier_id) or self.known.get(base_carrier_id(carrier_id))

    def source_curve(self, carrier_id: str) -> str:
        """The source curve a source carrier lies on: itself, the arc of a witness chord, or the root of a
        derived arc."""
        for candidate in (carrier_id, base_carrier_id(carrier_id)):
            if candidate in self.source_curves:
                return candidate
            arc = self.derived_arcs.get(candidate)
            if arc is not None and arc.root_id in self.source_curves:
                return arc.root_id
        raise SgbError(
            f"internal: the source carrier {carrier_id!r} lies on no source curve of the catalogue"
        )

    def role(self, family_id: str) -> str | None:
        """The role of the cut family ``family_id`` (``inset_band``, ``refinement`` or any other the catalogue
        gives); ``None`` for a family a lineage names that the catalogue does not hold."""
        return self.roles.get(family_id)
