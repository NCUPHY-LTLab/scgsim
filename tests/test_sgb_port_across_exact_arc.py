"""A lumped-port sheet whose ends reach one grid unit into two curved terminals builds like one on straight terminals (Issue #108).

The scene is ``circular_pad_capacitor`` at a tenth of its size: a disk pad of exact arcs inside a
circular ground opening, and a port sheet across the gap that starts one grid unit inside the pad,
where the pad's edge is at the sheet's corner, and ends one grid unit inside the ground.  The sheet's
long edges cross both exact circles.  The port splits each circle's arc there, while the concentric
band arcs beside it, 0.5 and 1.0 um away, stay whole; the chord of such a band arc then crossed the
split arc's chord and the port edge, and the dissolved arrangement was not planar.

Areas are hand-calculated geometry: annuli, and the parts of them the straight sheet covers.  Every
realised contour lies within sqrt(2)/2 grid of its exact circle or line, so a band area is within
that distance times the band's contour length.
"""

from __future__ import annotations

import math

import pytest
from sgb_scenes import GRID, chip_prisms, exact_band_areas, footprint, rect_um, units

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.bands.requests import InsetRequest, InterfaceSurfaces
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.exact.carriers import Lineage, circle_arcs
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import (
    Footprint,
    LineEdge,
    PortIdentity,
    PrismCatalogue,
    PrismRecord,
)

PAD_RADIUS = 10.0
GAP = 2.0
OPENING_RADIUS = PAD_RADIUS + GAP
HALF_WIDTH = 1.0  # the sheet spans |y| <= 1 um
DIE_HALF = 20.0
MARGINS = (0.5, 1.0)
SA_FAMILY = "sa::substrate|vacuum::z=0.0"
# One grid unit into each terminal, measured where the sheet's corner meets the curved edge.
SHEET_START = (
    units(math.sqrt(PAD_RADIUS**2 - HALF_WIDTH**2)) * GRID.size_um - GRID.size_um
)
SHEET_END = OPENING_RADIUS + GRID.size_um
SNAP_UM = math.sqrt(2) / 2 * GRID.size_um


def curved_port_scene(route: str) -> PrismCatalogue:
    """The pad, the ground with its opening, and the sheet on the hosts' support plane z = 0."""
    top = exact_z(0 if route == "A" else "0.2")
    prisms = chip_prisms(GRID, DIE_HALF)
    pad_footprint, pad_arcs = Footprint.from_primitives(
        circle_arcs(
            "pad/edge",
            (0, 0),
            units(PAD_RADIUS),
            True,
            Lineage("polygon_edge", "pad", 0, 0, 0),
        )
    )
    prisms.append(
        PrismRecord(
            "pad",
            pad_footprint,
            exact_z(0),
            top,
            "Al",
            "conductor",
            net_id="pad",
            arcs=pad_arcs,
        )
    )
    die = [
        (units(x), units(y))
        for x, y in rect_um(-DIE_HALF, -DIE_HALF, DIE_HALF, DIE_HALF)
    ]
    opening = circle_arcs(
        "ground/opening",
        (0, 0),
        units(OPENING_RADIUS),
        False,
        Lineage("polygon_edge", "ground", 0, 1, 0),
    )
    ground_footprint, ground_arcs = Footprint.from_primitives(
        [LineEdge(die[i], die[(i + 1) % 4], i) for i in range(4)], [opening]
    )
    prisms.append(
        PrismRecord(
            "ground",
            ground_footprint,
            exact_z(0),
            top,
            "Al",
            "conductor",
            net_id="Ground",
            arcs=ground_arcs,
        )
    )
    identity = PortIdentity(
        "p1", "P1", 1, (1.0, 0.0, 0.0), ("pad", "ground"), "metal", "port"
    )
    sheet = footprint(rect_um(SHEET_START, -HALF_WIDTH, SHEET_END, HALF_WIDTH))
    prisms.append(
        PrismRecord(
            "p1", sheet, exact_z(0), exact_z(0), "lumped_port", "port", port=identity
        )
    )
    return PrismCatalogue(GRID, tuple(prisms), name=f"curved_port_{route}")


def type_bands() -> tuple[InsetRequest, ...]:
    """MA, MS and SA bands at 0.5 and 1.0 um."""
    return tuple(
        InsetRequest(t.lower(), InterfaceSurfaces(t), MARGINS)
        for t in ("MA", "MS", "SA")
    )


def build(route: str, *, bands: bool) -> ConformalModel:
    """The scene's model, with or without Surface-EPR bands."""
    return build_model_from_catalogue(
        curved_port_scene(route), route, insets=type_bands() if bands else ()
    )


def strip_under_circle(radius: float) -> float:
    """Area of the circle of ``radius`` inside |y| <= HALF_WIDTH on the side x > 0."""
    h = HALF_WIDTH
    return h * math.sqrt(radius**2 - h**2) + radius**2 * math.asin(h / radius)


def annulus(inner: float, outer: float) -> float:
    """Area between two concentric circles."""
    return math.pi * (outer**2 - inner**2)


def covered_annulus(inner: float, outer: float) -> float:
    """The annulus between two circles in the gap, less the part the sheet covers there."""
    return annulus(inner, outer) - (
        strip_under_circle(outer) - strip_under_circle(inner)
    )


def contour(*radii: float) -> float:
    """Length of the circles bounding a band."""
    return sum(2 * math.pi * r for r in radii)


def port_faces(model: ConformalModel) -> list[tuple[str, float]]:
    """(face id, exact area in um^2) of each face of the port group."""
    (level,) = [layer for layer in model.labelled.level_layers if layer.level_z == 0]
    found = []
    for fi in range(1, len(level.arr.faces)):
        sheet = level.labels[fi].sheet
        if sheet is not None and sheet.kind == "port":
            found.append(
                (
                    level.face_ids[fi],
                    face_exact_area(level.arr, level.edge_curve, fi) * GRID.size_um**2,
                )
            )
    return found


def bands_um2(model: ConformalModel, family: str) -> dict[int, float]:
    """Band areas of a family in um^2, port faces left out."""
    return {
        band: area * GRID.size_um**2
        for band, area in exact_band_areas(model, family, include_ports=False).items()
    }


@pytest.mark.parametrize("route", ["A", "B"])
@pytest.mark.parametrize("bands", [False, True], ids=["no_bands", "bands"])
def test_a_sheet_across_two_exact_arcs_builds_and_validates(route, bands):
    """Every invariant holds, the planarity of the dissolved arrangements included."""
    model = build(route, bands=bands)
    assert model.report.failures == []
    assert "arrangements_planar" in {c.name for c in model.report.checks}


@pytest.mark.parametrize("bands", [False, True], ids=["no_bands", "bands"])
def test_the_port_is_one_face_over_the_gap(bands):
    """The port is the sheet between the two circles: the strip under the opening less the strip under the pad."""
    model = build("B", bands=bands)
    (port,) = port_faces(model)
    expected = strip_under_circle(OPENING_RADIUS) - strip_under_circle(PAD_RADIUS)
    assert port[1] == pytest.approx(expected, abs=SNAP_UM * (4 * HALF_WIDTH + 2 * GAP))
    assert len(model.routed.groups["PORT::p1"].ids) == 1


def test_bands_on_the_terminals_and_in_the_gap_are_the_hand_calculated_areas():
    """MA and MS bands are the terminals' annuli, the 1-unit landings included; SA bands leave out the sheet."""
    model = build("B", bands=True)
    r, o = PAD_RADIUS, OPENING_RADIUS
    pad = {
        0: annulus(r - 0.5, r),
        1: annulus(r - 1, r - 0.5),
        2: math.pi * (r - 1) ** 2,
    }
    pad_bound = {0: contour(r - 0.5, r), 1: contour(r - 1, r - 0.5), 2: contour(r - 1)}
    ground = {
        0: annulus(o, o + 0.5),
        1: annulus(o + 0.5, o + 1),
        2: (2 * DIE_HALF) ** 2 - math.pi * (o + 1) ** 2,
    }
    ground_bound = {
        0: contour(o, o + 0.5),
        1: contour(o + 0.5, o + 1),
        2: contour(o + 1),
    }
    # The gap is 2 um, so the SA bands from its two edges meet at its middle and there is no band 2.
    sa = {
        0: covered_annulus(r, r + 0.5) + covered_annulus(o - 0.5, o),
        1: covered_annulus(r + 0.5, r + 1) + covered_annulus(o - 1, o - 0.5),
    }
    sa_bound = {
        0: contour(r, r + 0.5, o - 0.5, o),
        1: contour(r + 0.5, r + 1, o - 1, o - 0.5),
    }
    for family, expected, bound in (
        ("ma::pad::top", pad, pad_bound),
        ("ms::pad::bottom", pad, pad_bound),
        ("ma::ground::top", ground, ground_bound),
        ("ms::ground::bottom", ground, ground_bound),
        (SA_FAMILY, sa, sa_bound),
    ):
        got = bands_um2(model, family)
        assert sorted(got) == sorted(expected), family
        for band, area in expected.items():
            assert abs(got[band] - area) <= bound[band] * SNAP_UM, (
                family,
                band,
                got[band],
                area,
            )
