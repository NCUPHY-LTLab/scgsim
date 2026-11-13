"""Surface-EPR bands on a flipped die: a conductor hanging under its substrate is attached by its top face.

A flip chip has two dies across a vacuum gap.  The lower die (D0) carries its metal on the top of its substrate,
so a conductor's bottom face is metal-substrate (MS) and its top face metal-air (MA).  The upper die (D1) is turned
over: its metal hangs under its substrate, so the top face of a D1 conductor is MS and its bottom face, toward the
gap, is MA.  Bands are measured from the gap edges on the plane the metal is attached to, which for D1 is the plane
of its top face.  The bump joining the dies touches metal at both ends and is never banded.
"""

from __future__ import annotations

import math

import pytest
from sgb_scenes import GRID, footprint, rect_um

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
)
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.topology.routes.naming import interface_type_of

MARGINS = (0.5, 1.0)
HALF = 20.0  # the die is 40 x 40 um
HOLE = rect_um(-8, -8, 8, 8)  # the ground opening, 16 x 16 um
PAD = rect_um(-4, -4, 4, 4)  # an 8 x 8 um pad centred in the opening
BUMP = rect_um(12, -2, 16, 2)  # a 4 x 4 um bump on the grounds, 4 um from the opening


def metal(  # noqa: PLR0913 - a test builder
    entity: str, outer, z0: str, z1: str, net: str, holes=()
) -> PrismRecord:
    """A finite aluminium conductor."""
    return PrismRecord(
        entity,
        footprint(outer, holes),
        exact_z(z0),
        exact_z(z1),
        "Al",
        "conductor",
        net_id=net,
    )


def flip_chip() -> PrismCatalogue:
    """Two dies 8 um apart: on each a ground with an opening and a pad in it, the D1 pair facing down, one bump.

    D0 metal is 0 .. 0.2 um on the D0 substrate (below z = 0); D1 metal is 8.0 .. 8.2 um under the D1 substrate
    (above z = 8.2); the bump joins the grounds from 0.2 to 8.0 um.
    """
    box = footprint(rect_um(-HALF, -HALF, HALF, HALF))
    die = rect_um(-HALF, -HALF, HALF, HALF)
    return PrismCatalogue(
        GRID,
        (
            PrismRecord(
                "substrate_d0", box, exact_z(-10), exact_z(0), "substrate", "dielectric"
            ),
            PrismRecord("gap", box, exact_z(0), exact_z("8.2"), "vacuum", "vacuum"),
            PrismRecord(
                "substrate_d1",
                box,
                exact_z("8.2"),
                exact_z("18.2"),
                "substrate",
                "dielectric",
            ),
            metal("d0_ground", die, "0", "0.2", "G", [HOLE]),
            metal("d0_pad", PAD, "0", "0.2", "P"),
            metal("d1_ground", die, "8", "8.2", "G", [HOLE]),
            metal("d1_pad", PAD, "8", "8.2", "Q"),
            metal("bump", BUMP, "0.2", "8", "G"),
        ),
        name="flip_chip",
    )


def by_type() -> tuple[InsetRequest, ...]:
    """One type-level request per Surface-EPR type, id = the lower-case type."""
    return tuple(
        InsetRequest(t.lower(), InterfaceSurfaces(t), MARGINS)
        for t in ("MA", "MS", "SA")
    )


def expansion_of(model: ConformalModel, request_id: str):
    """The banded family ids and the skipped supports of one type-level request."""
    (found,) = (
        e for e in model.inset_plan.expansions if e.request.request_id == request_id
    )
    return list(found.banded), [s.support for s in found.skipped]


def band_areas(model: ConformalModel, family_id: str, row_type: str) -> list[float]:
    """Area (um^2) of each band of a family, counting only its cells of the given interface type.

    The cells where the bump stands on a ground are metal-metal (MM): they carry the family's band label but
    belong to no band group, so they are left out like the groups leave them out.
    """
    areas = [0.0] * (len(MARGINS) + 1)
    for layer in model.labelled.level_layers:
        for fi in range(1, len(layer.arr.faces)):
            label = layer.labels[fi]
            if interface_type_of(label.below, label.above, label.sheet) != row_type:
                continue
            for family, band in label.bands:
                if family == family_id:
                    areas[band] += (
                        face_exact_area(layer.arr, layer.edge_curve, fi)
                        * GRID.size_um**2
                    )
    return areas


def pad_bands() -> list[float]:
    """The bands of the 8 um pad, every edge a gap edge: the 8, 7 and 6 um squares nested."""
    return [64.0 - 49.0, 49.0 - 36.0, 36.0]


def ground_bands(covered: float = 0.0) -> list[float]:
    """The bands of the ground round the 16 um opening (the die edge is no origin), less an area ``covered``.

    The points within ``m`` of the opening on the metal side are the opening grown by ``m`` minus the opening:
    ``4 * 16 * m + pi m^2``.
    """
    near = [4 * 16 * m + math.pi * m * m for m in MARGINS]
    total = 40.0 * 40.0 - 16.0 * 16.0 - covered
    return [near[0], near[1] - near[0], total - near[1]]


def assert_bands(got: list[float], expected: list[float]) -> None:
    """The band areas agree to 0.001 um^2 (the arcs are realised exactly, the lattice is 1 nm)."""
    assert got == [pytest.approx(value, abs=1e-3) for value in expected]


def test_the_upper_die_conductors_are_banded_ma_toward_the_gap_and_ms_toward_their_substrate():
    """Every D1 conductor face of a type is banded, none skipped; D0 keeps its faces; the bump has no band."""
    model = build_model_from_catalogue(flip_chip(), "B", insets=by_type())
    assert expansion_of(model, "ma") == (
        [
            "ma::d1_ground::bottom",
            "ma::d1_pad::bottom",
            "ma::d0_ground::top",
            "ma::d0_pad::top",
        ],
        [],
    )
    assert expansion_of(model, "ms") == (
        [
            "ms::d0_ground::bottom",
            "ms::d0_pad::bottom",
            "ms::d1_ground::top",
            "ms::d1_pad::top",
        ],
        [],
    )
    groups = set(model.routed.groups)
    for kind, side in (("MA", "bottom"), ("MS", "top")):
        for entity in ("d1_ground", "d1_pad"):
            assert {f"{kind}::{entity}__{side}::band={k}" for k in range(3)} <= groups
    assert not [g for g in groups if "::bump__" in g and "::band=" in g]


def test_the_upper_die_bands_have_the_areas_of_the_gap_edges_on_its_substrate():
    """D1 bands mirror D0: the bump only takes its footprint off the gap-facing face of each ground."""
    model = build_model_from_catalogue(flip_chip(), "B", insets=by_type())
    bump_area = 16.0
    expected = {
        ("ma::d1_ground::bottom", "MA"): ground_bands(covered=bump_area),
        ("ms::d1_ground::top", "MS"): ground_bands(),
        ("ma::d1_pad::bottom", "MA"): pad_bands(),
        ("ms::d1_pad::top", "MS"): pad_bands(),
        ("ma::d0_ground::top", "MA"): ground_bands(covered=bump_area),
        ("ms::d0_ground::bottom", "MS"): ground_bands(),
        ("ma::d0_pad::top", "MA"): pad_bands(),
        ("ms::d0_pad::bottom", "MS"): pad_bands(),
    }
    for (family, row_type), areas in expected.items():
        assert_bands(band_areas(model, family, row_type), areas)


def test_a_concrete_request_bands_the_gap_facing_face_of_an_upper_die_pad():
    """Naming the face works too: the bottom of a D1 pad is MA, measured from the gap edges above it."""
    request = InsetRequest(
        "pad_gap_side", ConductorSurface("d1_pad", "bottom"), MARGINS, interface="MA"
    )
    model = build_model_from_catalogue(flip_chip(), "B", insets=(request,))
    assert_bands(band_areas(model, "pad_gap_side", "MA"), pad_bands())


def back_side_film() -> PrismCatalogue:
    """A substrate with vacuum on both sides and a film with an opening and a pad on its underside."""
    box = footprint(rect_um(-HALF, -HALF, HALF, HALF))
    return PrismCatalogue(
        GRID,
        (
            PrismRecord(
                "vacuum_below", box, exact_z(-20), exact_z(-10), "vacuum", "vacuum"
            ),
            PrismRecord(
                "substrate", box, exact_z(-10), exact_z(0), "substrate", "dielectric"
            ),
            PrismRecord("vacuum", box, exact_z(0), exact_z(10), "vacuum", "vacuum"),
            metal(
                "back_ground",
                rect_um(-HALF, -HALF, HALF, HALF),
                "-10.2",
                "-10",
                "G",
                [HOLE],
            ),
            metal("back_pad", PAD, "-10.2", "-10", "P"),
        ),
        name="back_side_film",
    )


def test_a_film_on_the_underside_of_a_substrate_is_banded_like_the_upper_die():
    """The back face of a substrate (a TSV landing) carries its metal the same way a flipped die does."""
    model = build_model_from_catalogue(back_side_film(), "B", insets=by_type())
    assert expansion_of(model, "ma") == (
        ["ma::back_ground::bottom", "ma::back_pad::bottom"],
        [],
    )
    assert expansion_of(model, "ms") == (
        ["ms::back_ground::top", "ms::back_pad::top"],
        [],
    )
    assert_bands(band_areas(model, "ma::back_pad::bottom", "MA"), pad_bands())
    assert_bands(band_areas(model, "ms::back_ground::top", "MS"), ground_bands())
