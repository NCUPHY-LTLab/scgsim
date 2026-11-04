"""Route A / Route B as labelling policies (brief section 8, stage S7; section 7 names).

Owns the decisions that differ between routes and nothing else:

* **Route A** -- conductors are zero-thickness sheets. A sheet cell of a level
  arrangement becomes a PEC face between the volumes below and above; no
  conductor volume exists.
* **Route B** -- conductors are finite prisms. Their volumes are *excluded*
  from the emitted model; their boundary faces become faces of the
  neighbouring dielectric/vacuum volumes.

Route is an early input: it changes which volumes are emitted and which
primary group a face joins, never the arrangement or assembly algorithms.

Physical groups *partition* each dimension: a face has exactly one primary
group. Semantic names follow brief section 7: ``MA::<net>`` metal-air,
``MS::<net>`` metal-substrate, ``SA::<a>|<b>`` substrate-air, ``PORT::<name>``,
``EXT::<domain>`` for the outer model boundary, ``VOL::<domain>`` for volumes,
and in Route A ``PEC::<net>`` for sheet faces. "All PEC faces of Net X" in
Route B is a *union* -- a list of primary group names recorded in the sidecar,
never an overlapping Gmsh group.
"""

from __future__ import annotations

from dataclasses import dataclass

from sgbproto.errors import OverlapError, RouteError
from sgbproto.records import Domain, SheetOwner


@dataclass(frozen=True)
class RoutePolicy:
    name: str
    exclude_conductor_volumes: bool
    allow_finite_conductors: bool

    def is_emitted(self, domain: Domain | None) -> bool:
        """Whether a volume of this domain is part of the emitted model."""
        if domain is None:
            return False
        return not (domain.kind == "conductor" and self.exclude_conductor_volumes)


ROUTE_A = RoutePolicy("A", exclude_conductor_volumes=False, allow_finite_conductors=False)
ROUTE_B = RoutePolicy("B", exclude_conductor_volumes=True, allow_finite_conductors=True)
ROUTES = {"A": ROUTE_A, "B": ROUTE_B}


def check_route(policy: RoutePolicy, prisms) -> None:
    """Route A accepts only sheet conductors; fail closed otherwise."""
    if policy.allow_finite_conductors:
        return
    finite = [p.entity_id for p in prisms if p.material_kind == "conductor" and not p.is_sheet]
    if finite:
        raise RouteError(f"Route {policy.name} needs sheet conductors; finite conductors: {finite}")


def surface_group(
    neg: Domain | None, pos: Domain | None, owner: SheetOwner | None, policy: RoutePolicy, bands: tuple = ()
) -> str | None:
    """The single primary physical-group name of a face from the domains on its two sides.

    A face inside an inset support region carries its band: ``MA::pad::band=2`` (top face) or
    ``MS::pad::band=2`` (bottom face) in Route B. A Route A conductor sheet in a banded region is one face that is
    both the metal-air and the metal-substrate surface: its group is ``SHEET::net::band=k`` and both Surface-EPR
    rows reference it (see ``surface_epr_rows``).
    """
    name = _base_group(neg, pos, owner, policy, bool(bands))
    port = owner is not None and owner.kind == "port"  # a port stays one surface even where bands cut it
    return name if name is None or not bands or port else f"{name}::band={bands[0][1]}"


def _base_group(neg, pos, owner, policy, banded: bool) -> str | None:
    if owner is not None:
        if owner.kind == "port":
            return f"PORT::{owner.name}"
        return f"SHEET::{owner.name}" if banded else f"PEC::{owner.name}"
    if neg is None and pos is None:
        return None
    if neg is None or pos is None:
        return f"EXT::{(pos if neg is None else neg).name}"
    conductors = [d for d in (neg, pos) if d.kind == "conductor"]
    if len(conductors) == 2:
        raise OverlapError(f"different-Net conductors touch: {neg.name} / {pos.name}")
    if conductors:
        metal = conductors[0]
        other = pos if metal == neg else neg
        return f"{'MA' if other.kind == 'vacuum' else 'MS'}::{metal.name}"
    first, second = sorted((neg, pos), key=lambda d: (d.kind == "vacuum", d.name))
    return f"SA::{first.name}|{second.name}"


def surface_epr_rows(group_names) -> dict[str, str]:
    """Surface-EPR row -> the primary group that carries it, for every banded group.

    A Route A sheet group (``SHEET::net::band=k``) carries both the ``MA::net::band=k`` and the
    ``MS::net::band=k`` row; every other banded group (``MA``, ``MS``, ``SA``) carries its own row.
    """
    rows: dict[str, str] = {}
    for name in sorted(group_names):
        if "::band=" not in name:
            continue
        kind, rest = name.split("::", 1)
        if kind == "SHEET":
            rows[f"MA::{rest}"] = name
            rows[f"MS::{rest}"] = name
        else:
            rows[name] = name
    return rows


def pec_net(neg: Domain | None, pos: Domain | None, owner: SheetOwner | None) -> str | None:
    """Net whose PEC union a face belongs to (a conductor sheet, or a face of an excluded conductor)."""
    if owner is not None:
        return owner.name if owner.kind == "conductor" else None
    for d in (neg, pos):
        if d is not None and d.kind == "conductor":
            return d.name
    return None
