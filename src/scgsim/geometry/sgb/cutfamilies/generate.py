"""Stage S3: from requests to ``CurveFamilyRecord`` s (design brief sections 5 and 8).

Owns the one place requests become cut families.  Physical boundaries need no family -- they are the
prism footprints and implicit.  Everything else is explicit:

* an inset request becomes a ``required`` family of boundaries of ``R_m`` (``inset``), one family per
  support face, targeting the level the face lies on;
* a refinement request becomes a ``refinement_only`` family of line members at its height.

A family whose Z no prism defines adds an ordinary level (``prisms.levels``); the arrangement code never
learns where a family came from.
"""

from __future__ import annotations

from dataclasses import replace

from scgsim.geometry.sgb.cutfamilies.inset import inset_family
from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest, RefinementRequest
from scgsim.geometry.sgb.cutfamilies.supports import resolve_supports
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import Grid, exact_z
from scgsim.geometry.sgb.records.prisms import PrismCatalogue


def with_cut_families(
    catalogue: PrismCatalogue,
    insets: tuple[InsetRequest, ...] = (),
    refinements: tuple[RefinementRequest, ...] = (),
) -> PrismCatalogue:
    """The catalogue with the families its requests generate (ids must be unique)."""
    ids = [r.request_id for r in (*insets, *refinements)] + [
        f.family_id for f in catalogue.families
    ]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise InputError(
            f"cut-family request ids must be unique, repeated: {', '.join(duplicates)}"
        )
    families = [
        inset_family(support.request.request_id, support.region, (support.level,))
        for support in resolve_supports(catalogue, insets)
    ]
    families.extend(
        _refinement_family(request, catalogue.grid) for request in refinements
    )
    return replace(catalogue, families=(*catalogue.families, *families))


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
