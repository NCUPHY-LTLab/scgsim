"""Stage S3: from requests to ``CurveFamilyRecord`` s (design brief sections 5 and 8).

Owns the one place requests become cut families.  Physical boundaries need no family -- they are the
prism footprints and implicit.  Everything else is explicit:

* an inset request becomes a ``required`` family of boundaries of ``R_m`` (``inset``), one family per
  support face, targeting the level the face lies on; the seams stop at the boundary of every lumped port
  on that level, so a port stays one face.  A type-level request is first expanded into its faces, and a face
  that several requests land on is one family (``inset_plan``);
* a refinement request becomes a ``refinement_only`` family of line members at its height; it is a user
  request and is never clipped, so it cuts a port it crosses.

A family whose Z no prism defines adds an ordinary level (``prisms.levels``); the arrangement code never
learns where a family came from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from scgsim.geometry.sgb.cutfamilies.inset import PortClip, inset_family
from scgsim.geometry.sgb.cutfamilies.inset_plan import InsetPlan, plan_insets
from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest, RefinementRequest
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import Grid, exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismCatalogue


@dataclass(frozen=True)
class CutFamilies:
    """A catalogue with its requested cut families, and how the inset requests became band families."""

    catalogue: PrismCatalogue
    insets: InsetPlan


def with_cut_families(
    catalogue: PrismCatalogue,
    insets: tuple[InsetRequest, ...] = (),
    refinements: tuple[RefinementRequest, ...] = (),
) -> PrismCatalogue:
    """The catalogue with the families its requests generate (ids must be unique)."""
    return build_cut_families(catalogue, insets, refinements).catalogue


def build_cut_families(
    catalogue: PrismCatalogue,
    insets: tuple[InsetRequest, ...] = (),
    refinements: tuple[RefinementRequest, ...] = (),
) -> CutFamilies:
    """The catalogue with the families its requests generate, and the plan the inset requests became."""
    _check_unique_ids([r.request_id for r in (*insets, *refinements)], catalogue)
    plan = plan_insets(catalogue, insets)
    _check_unique_ids([r.request_id for r in (*plan.requests, *refinements)], catalogue)
    families = [
        inset_family(
            support.request.request_id,
            support.region,
            (support.level,),
            _port_footprints(catalogue, support.level),
        )
        for support in plan.supports
    ]
    families.extend(
        _refinement_family(request, catalogue.grid) for request in refinements
    )
    return CutFamilies(
        replace(catalogue, families=(*catalogue.families, *families)), plan
    )


def _check_unique_ids(request_ids: list[str], catalogue: PrismCatalogue) -> None:
    ids = [*request_ids, *(f.family_id for f in catalogue.families)]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise InputError(
            f"cut-family request ids must be unique, repeated: {', '.join(duplicates)}"
        )


def _port_footprints(
    catalogue: PrismCatalogue, level: Fraction
) -> tuple[PortClip, ...]:
    """The lumped-port sheets on a level, which the inset seams must not enter off their hosts."""
    return tuple(
        PortClip(
            prism.footprint,
            _landings(catalogue, prism.port.host_entity_ids, level)
            if prism.port.lands_on_hosts
            else (),
        )
        for prism in catalogue.prisms
        if prism.port is not None and prism.z0 == level
    )


def _landings(
    catalogue: PrismCatalogue, host_entity_ids: tuple[str, ...], level: Fraction
) -> tuple[Footprint, ...]:
    """The footprints of the host conductors that meet a port's plane (a body above or below it, or a sheet on it)."""
    return tuple(
        prism.footprint
        for prism in catalogue.prisms
        if prism.entity_id in host_entity_ids and prism.z0 <= level <= prism.z1
    )


def _refinement_family(request: RefinementRequest, grid: Grid) -> CurveFamilyRecord:
    if len(request.points_um) < 2:
        raise InputError(
            f"refinement request {request.request_id!r} needs at least two points"
        )
    points = []
    for x_um, y_um in request.points_um:
        x, y = grid.lattice_units(x_um), grid.lattice_units(y_um)
        if x is None or y is None:
            raise InputError(
                f"refinement request {request.request_id!r}: point ({x_um}, {y_um}) um is off the "
                f"{grid.size_um} um model grid"
            )
        points.append((x, y))
    carriers = tuple(
        LineCarrier(
            f"{request.request_id}/seg{k}",
            p,
            q,
            Lineage(
                "family_member",
                family_id=request.request_id,
                member=f"{request.request_id}/seg{k}",
            ),
        )
        for k, (p, q) in enumerate(zip(points, points[1:], strict=False))
        if p != q
    )
    if not carriers:
        raise InputError(
            f"refinement request {request.request_id!r} has no segment of positive length"
        )
    return CurveFamilyRecord(
        request.request_id,
        "refinement",
        carriers,
        (exact_z(request.z_um),),
        "refinement_only",
    )
