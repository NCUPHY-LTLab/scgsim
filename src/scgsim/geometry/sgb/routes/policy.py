"""Route A, B and C as labelling policies (design brief sections 8 and 14, stage S7).

Owns the decisions that differ between routes and nothing else.  The acceptance rules follow
today's contract, ``planning.representation.validate_selected_route`` with
``models.common.ROUTE_ALLOWED_REPRESENTATIONS`` (Route A: ``surface_sheet`` and
``cutout_boundary_shell``; Route B: ``cutout_boundary_shell``; Route C: ``material_volume``):

* **Route A** -- face-metal *films* are zero-thickness sheets and the finite *bodies* (bumps,
  posts, TSV-like volumes) keep their volume, which is excluded like in Route B.  A sheet cell of
  a level becomes a PEC face between the volumes below and above.  A conductor whose physical
  declaration is a film (``stack_film`` or ``local_film``) and that is still finite after the
  physical interpretation is rejected: the selected profile did not collapse it.  A finite
  conductor with a ``finite_volume`` declaration -- or with no declaration at all, as today's
  ``cutout_boundary_shell`` representation allows -- is accepted.
* **Route B** -- every conductor is a finite prism whose volume is *excluded* from the emitted
  model; its boundary faces become faces of the neighbouring volumes.  Conductor sheets are
  rejected (today's ``cutout_boundary_shell`` is the only representation).  Lumped-port sheets
  are accepted: a port lies on the support plane of its hosts (the surface under the metal), so
  it is a face of the interface there, never a free face.
* **Route C** -- every conductor is a finite prism whose volume is *retained and meshed*:
  conductor-dielectric and conductor-vacuum interfaces are shared faces, the interior is a
  solution domain (finite conductivity today, superconducting material models later).  Conductor
  sheets are rejected.  Lumped-port sheets are accepted and represented as in Route B: a face of
  the interface in the gap at the support plane whose edges are shared with the conductor walls.

Route is an early input: it changes which volumes are emitted and which primary group a face
joins, never the arrangement or assembly algorithms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scgsim.geometry.sgb.errors import RouteError
from scgsim.geometry.sgb.records.labels import Domain
from scgsim.geometry.sgb.records.prisms import FILM_FEATURE_KINDS, PrismRecord


@dataclass(frozen=True)
class RoutePolicy:
    """What a route emits and accepts."""

    name: str
    exclude_conductor_volumes: bool
    allows_sheets: bool
    rejects_finite_films: bool

    def is_emitted(self, domain: Domain | None) -> bool:
        """Whether a volume of this domain is part of the emitted model."""
        if domain is None:
            return False
        return not (domain.kind == "conductor" and self.exclude_conductor_volumes)


ROUTE_A = RoutePolicy(
    "A", exclude_conductor_volumes=True, allows_sheets=True, rejects_finite_films=True
)
ROUTE_B = RoutePolicy(
    "B", exclude_conductor_volumes=True, allows_sheets=False, rejects_finite_films=False
)
ROUTE_C = RoutePolicy(
    "C",
    exclude_conductor_volumes=False,
    allows_sheets=False,
    rejects_finite_films=False,
)
ROUTES = {"A": ROUTE_A, "B": ROUTE_B, "C": ROUTE_C}


def route_policy(name: str) -> RoutePolicy:
    """The policy of a route name; an unknown route is an error."""
    try:
        return ROUTES[name]
    except KeyError:
        raise RouteError(f"unknown route {name!r}; choose one of A, B or C") from None


def check_route(policy: RoutePolicy, prisms: Sequence[PrismRecord]) -> None:
    """Fail closed when the prisms are not valid for the route (before any arrangement is built)."""
    conductors = [p for p in prisms if p.material_kind == "conductor"]
    if policy.rejects_finite_films:
        films = sorted(
            {
                p.entity_id
                for p in conductors
                if not p.is_sheet and p.feature_kind in FILM_FEATURE_KINDS
            }
        )
        if films:
            raise RouteError(
                f"Route {policy.name} needs film conductors as sheets, but these films are still "
                f"finite after the physical interpretation: {', '.join(films)}; select a profile "
                "that collapses them to sheets (for example thin_film) or choose Route B or C"
            )
    if not policy.allows_sheets:
        sheets = sorted({p.entity_id for p in conductors if p.is_sheet})
        if sheets:
            raise RouteError(
                f"Route {policy.name} keeps conductors finite and has no conductor sheets, but these "
                f"are sheets: {', '.join(sheets)}; select a profile that keeps them finite or choose Route A"
            )
