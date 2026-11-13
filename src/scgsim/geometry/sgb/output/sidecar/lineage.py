"""Where each carrier of a model comes from (design brief section 7), for the sidecar's ``carrier`` table.

Owns the one lookup the writer needs to describe a carrier id that lies on an emitted edge: its lineage (a
source polygon edge or arc, or a member of a cut family) and, for a source carrier, the source curve it lies
on.  The catalogue's table of known carriers (``PrismCatalogue.known_carriers``) and the derived arcs of the
model say which curve a witness chord or a derived arc belongs to; the sidecar states the relation as a field,
and neither the writer nor a reader reads it from an id.
"""

from __future__ import annotations

from dataclasses import dataclass

from scgsim.geometry.models.refusals import internal
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.exact.carriers import ArcCarrier, Lineage
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue
from scgsim.geometry.sgb.model import ConformalModel

BAND_FAMILY_ROLE = "inset_band"


@dataclass(frozen=True)
class CarrierLineage:
    """The lineage of every carrier id a model knows, the curve each lies on, the source curves, and the role of
    each cut family."""

    known: dict[str, Lineage]
    curves: dict[str, str]
    source_curves: frozenset[str]
    derived_arcs: dict[str, ArcCarrier]
    roles: dict[str, str]

    @classmethod
    def of(cls, model: ConformalModel) -> CarrierLineage:
        """The carriers of ``model``: source line carriers and arcs, family members, derived arcs."""
        catalogue = model.labelled.catalogue
        table = catalogue.known_carriers()
        known = {carrier_id: entry.lineage for carrier_id, entry in table.items()}
        curves = {carrier_id: entry.curve for carrier_id, entry in table.items()}
        for arc_id, arc in model.labelled.arcs.items():
            known.setdefault(arc_id, arc.lineage)
            for piece in arc.pieces():
                known.setdefault(piece.id, arc.lineage)
                curves.setdefault(piece.id, arc_id)
        return cls(
            known,
            curves,
            _source_curves(catalogue),
            dict(model.labelled.arcs),
            {f.family_id: f.role for f in catalogue.families},
        )

    def lineage(self, carrier_id: str) -> Lineage | None:
        """The lineage of a carrier id, a witness chord's included; ``None`` for a carrier SGB does not know."""
        return self.known.get(carrier_id)

    def source_curve(self, carrier_id: str) -> str:
        """The source curve a source carrier lies on: itself, the arc of a witness chord, or the root of a
        derived arc."""
        for candidate in (carrier_id, self.curves.get(carrier_id, carrier_id)):
            if candidate in self.source_curves:
                return candidate
            arc = self.derived_arcs.get(candidate)
            if arc is not None and arc.root_id in self.source_curves:
                return arc.root_id
        raise SgbError(
            internal(
                f"the source carrier {carrier_id!r} lies on no source curve of the catalogue"
            )
        )

    def role(self, family_id: str) -> str | None:
        """The role of the cut family ``family_id`` (``inset_band``, or any other the catalogue gives); ``None``
        for a family a lineage names that the catalogue does not hold."""
        return self.roles.get(family_id)


def _source_curves(catalogue: PrismCatalogue) -> frozenset[str]:
    """The source curves: the straight source edges and the arcs of every prism.

    A witness chord of an arc is not a source curve: it lies on its arc.
    """
    sources: set[str] = set()
    for prism in catalogue.prisms:
        sources.update(c.id for c in prism.carriers() if c.role == "source_edge")
        sources.update(arc.id for arc in prism.arcs)
    return frozenset(sources)
