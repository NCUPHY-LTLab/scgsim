"""Route A, B and C as labelling policies (design brief sections 8 and 14, stage S7).

Owns the decisions that differ between routes and nothing else:

* **Route A** -- conductors are zero-thickness sheets.  A sheet cell of a level becomes a
  PEC face between the volumes below and above; no conductor volume exists.  A conductor
  that is still finite after the physical interpretation is rejected with a clear error.
* **Route B** -- conductors are finite prisms whose volumes are *excluded* from the emitted
  model; their boundary faces become faces of the neighbouring dielectric and vacuum
  volumes.  Sheets (a profile that collapses a film) are PEC faces as in Route A.
* **Route C** -- conductors are finite prisms whose volumes are *retained and meshed*:
  conductor-dielectric and conductor-vacuum interfaces are shared faces, the interior is
  a solution domain (finite conductivity today, superconducting material models later).
  Sheets and lumped-port sheets have no meaning here and are rejected.

Route is an early input: it changes which volumes are emitted and which primary group a
face joins, never the arrangement or assembly algorithms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scgsim.geometry.sgb.errors import RouteError
from scgsim.geometry.sgb.records.labels import Domain
from scgsim.geometry.sgb.records.prisms import PrismRecord


@dataclass(frozen=True)
class RoutePolicy:
    """What a route emits and accepts."""

    name: str
    exclude_conductor_volumes: bool
    allows_finite_conductors: bool
    allows_sheets: bool

    def is_emitted(self, domain: Domain | None) -> bool:
        """Whether a volume of this domain is part of the emitted model."""
        if domain is None:
            return False
        return not (domain.kind == "conductor" and self.exclude_conductor_volumes)


ROUTE_A = RoutePolicy(
    "A",
    exclude_conductor_volumes=False,
    allows_finite_conductors=False,
    allows_sheets=True,
)
ROUTE_B = RoutePolicy(
    "B",
    exclude_conductor_volumes=True,
    allows_finite_conductors=True,
    allows_sheets=True,
)
ROUTE_C = RoutePolicy(
    "C",
    exclude_conductor_volumes=False,
    allows_finite_conductors=True,
    allows_sheets=False,
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
    if not policy.allows_finite_conductors:
        finite = sorted({p.entity_id for p in conductors if not p.is_sheet})
        if finite:
            raise RouteError(
                f"Route {policy.name} needs sheet conductors, but these are still finite after the "
                f"physical interpretation: {', '.join(finite)}; select a profile that collapses them "
                "to sheets (for example thin_film) or choose Route B or C"
            )
    if not policy.allows_sheets:
        sheets = sorted(
            {p.entity_id for p in prisms if p.is_sheet and p.material_kind != "port"}
        )
        ports = sorted({p.entity_id for p in prisms if p.material_kind == "port"})
        if sheets or ports:
            raise RouteError(
                f"Route {policy.name} keeps conductor volumes and has no sheets: "
                f"sheet conductors {sheets or 'none'}, lumped-port sheets {ports or 'none'}"
            )
