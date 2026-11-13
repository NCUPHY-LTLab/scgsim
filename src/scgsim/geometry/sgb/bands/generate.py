"""Stage S3: from requests to ``CurveFamilyRecord`` s (design brief sections 5 and 8).

Owns the one place requests become cut families.  Physical boundaries need no family -- they are the
prism footprints and implicit.  Everything else is explicit:

an inset request becomes a family of boundaries of ``R_m`` (``inset``), one family per support face,
targeting the level the face lies on; the seams stop at the boundary of every lumped port on that level, so a
port stays one face.  A type-level request is first expanded into its faces, and a face that several requests
land on is one family (``inset_plan``).

A family whose Z no prism defines adds an ordinary level (``prisms.levels``); the arrangement code never
learns where a family came from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from scgsim.geometry.sgb.bands.inset import inset_family
from scgsim.geometry.sgb.bands.inset_plan import InsetPlan
from scgsim.geometry.sgb.bands.inset_regions import PortClip
from scgsim.geometry.sgb.bands.requests import InsetRequest
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.exact.prisms import Footprint, PrismCatalogue


@dataclass(frozen=True)
class CutFamilies:
    """A catalogue with its requested cut families, and how the inset requests became band families."""

    catalogue: PrismCatalogue
    plan: InsetPlan


def check_request_ids(
    insets: tuple[InsetRequest, ...], catalogue: PrismCatalogue
) -> None:
    """Refuse inset requests whose ids repeat, or repeat a cut family the catalogue already carries."""
    _check_unique_ids([r.request_id for r in insets], catalogue)


def build_cut_families(catalogue: PrismCatalogue, plan: InsetPlan) -> CutFamilies:
    """The catalogue with one band family per resolved support of ``plan``, and the plan itself."""
    _check_unique_ids([r.request_id for r in plan.requests], catalogue)
    families = [
        inset_family(
            support.request.request_id,
            support.region,
            (support.level,),
            _port_footprints(catalogue, support.level),
            catalogue.grid.size_um,
        )
        for support in plan.supports
    ]
    return CutFamilies(
        replace(catalogue, families=(*catalogue.families, *families)), plan
    )


def _check_unique_ids(request_ids: list[str], catalogue: PrismCatalogue) -> None:
    ids = [*request_ids, *(f.family_id for f in catalogue.families)]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise InputError(
            f"inset request ids must be unique and differ from the cut families the catalogue already "
            f"carries; repeated: {', '.join(duplicates)}. Fix the requests: give each its own request_id"
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
