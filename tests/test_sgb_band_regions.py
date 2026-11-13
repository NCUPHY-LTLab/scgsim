"""Band regions are established before Imprint, only the seams between different bands are emitted, and Imprint must keep them.

Example A is a straight gap between two gap edges; example B a square whose four edges are all origins.  Their
areas are hand-calculated geometry, not simulation results.  The injected faults drop seams from the level's
noding input -- the real pipeline otherwise -- to show the Imprint check refuses a lost segment or loop.
"""

from __future__ import annotations

import math

import pytest
from sgb_scenes import (
    GRID,
    chip_prisms,
    conductor,
    footprint,
    rect_um,
    slab_model,
    units,
)

import scgsim.geometry.sgb.arrangement.derivation.level_layers as level_layers
from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
    SolutionInterface,
)
from scgsim.geometry.sgb.errors import ValidationError
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.exact.plane import point_in_ring
from scgsim.geometry.sgb.exact.carriers import Lineage, circle_arcs
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import (
    Footprint,
    LineEdge,
    PrismCatalogue,
    PrismRecord,
)

MARGINS = (0.5, 1.0)
SHEET_HEIGHT = 10.0
SHEET_WIDTH = 9.0


def gap_model(width: float) -> PrismCatalogue:
    """Two 9 x 10 um sheets (Route A) facing each other across a straight gap ``width`` um wide, centred on x = 0."""
    half = width / 2
    return slab_model(
        "A",
        [
            ("left", "N1", (-half - SHEET_WIDTH, 0.0, -half, SHEET_HEIGHT)),
            ("right", "N2", (half, 0.0, half + SHEET_WIDTH, SHEET_HEIGHT)),
        ],
    )


def sa_bands(margins=MARGINS) -> InsetRequest:
    """Bands on the substrate-to-vacuum interface at z = 0, measured from the sheets' edges."""
    return InsetRequest("sa", SolutionInterface("substrate", "vacuum", z_um=0), margins)


def band_faces(
    model: ConformalModel, family_id: str
) -> list[tuple[int, float, object]]:
    """``(band, area in um^2, face index)`` of every dissolved level face of a family."""
    grid = model.labelled.catalogue.grid.size_um
    out = []
    for layer in model.labelled.level_layers:
        for fi in range(1, len(layer.arr.faces)):
            for family, band in layer.labels[fi].bands:
                if family == family_id:
                    area = face_exact_area(layer.arr, layer.edge_curve, fi) * grid**2
                    out.append((band, area, (layer, fi)))
    return out


def band_areas(model: ConformalModel, family_id: str) -> dict[int, float]:
    """Area per band of a family (um^2)."""
    areas: dict[int, float] = {}
    for band, area, _ in band_faces(model, family_id):
        areas[band] = areas.get(band, 0.0) + area
    return areas


def band_at(model: ConformalModel, family_id: str, x_um: float, y_um: float) -> int:
    """The band of the dissolved face holding a point (off every edge)."""
    grid = model.labelled.catalogue.grid.size_um
    den = 1000
    point = (round(x_um / grid * den), round(y_um / grid * den), den)
    (found,) = [
        band
        for band, _, (layer, fi) in band_faces(model, family_id)
        if sum(
            point_in_ring(layer.arr.cycle_points(c), point)
            for c in layer.arr.face_cycles(fi)
        )
        % 2
    ]
    return found


def seams_of(model: ConformalModel, family_id: str):
    """The required seams the family recorded."""
    family = next(
        f for f in model.labelled.catalogue.families if f.family_id == family_id
    )
    return family.regions.seams


def grown(r: float) -> float:
    """Area of one 9 x 10 um sheet grown by r um: the sheet, four strips and four quarter discs."""
    return (
        SHEET_WIDTH * SHEET_HEIGHT
        + 2 * r * (SHEET_WIDTH + SHEET_HEIGHT)
        + math.pi * r * r
    )


def overlap(r: float, width: float) -> float:
    """Area where the r-neighbourhoods of the two sheets overlap across the gap (zero unless 2r > width)."""
    if 2 * r <= width:
        return 0.0
    lens = 2 * r * r * math.acos(width / (2 * r)) - width / 2 * math.sqrt(
        4 * r * r - width * width
    )
    return SHEET_HEIGHT * (2 * r - width) + lens


def expected_sa_areas(width: float) -> dict[int, float]:
    """Hand-calculated SA band areas: the die (80 x 80 um) less the sheets, cut at 0.5 and 1.0 um."""
    within = [
        2 * grown(r) - overlap(r, width) - 2 * SHEET_WIDTH * SHEET_HEIGHT
        for r in MARGINS
    ]
    support = 80.0 * 80.0 - 2 * SHEET_WIDTH * SHEET_HEIGHT
    return {0: within[0], 1: within[1] - within[0], 2: support - within[1]}


# --------------------------------------------------------------------------- example A: a straight gap


@pytest.mark.parametrize(
    ("width", "across"),
    [
        # positions across the gap at mid height (x from the gap centre, um) and their band
        (1.6, {-0.6: 0, -0.2: 1, 0.0: 1, 0.2: 1, 0.6: 0}),
        (2.0, {-0.75: 0, -0.25: 1, 0.0: 1, 0.25: 1, 0.75: 0}),
        (2.6, {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 1, 1.0: 0}),
    ],
)
def test_example_a_a_straight_gap_has_the_bands_of_its_distance_and_no_centre_seam(
    width, across
):
    """Gaps narrower than, equal to and wider than 2 x 1.0 um: band 0 | band 1 (| band 2 |) band 1 | band 0."""
    model = build_model_from_catalogue(gap_model(width), "A", insets=(sa_bands(),))
    assert not model.report.failures
    for x, band in across.items():
        assert band_at(model, "sa", x, SHEET_HEIGHT / 2) == band, x
    assert band_areas(model, "sa") == pytest.approx(expected_sa_areas(width), rel=1e-6)
    # no seam lies on the centre line: an edge there would end on x = 0 at mid height
    grid = model.labelled.catalogue.grid.size_um
    for layer in model.labelled.level_layers:
        for edge in layer.arr.edges:
            assert not (
                edge.a[0] == edge.b[0] == 0 and edge.a[1] < 5 / grid < edge.b[1]
            )


def test_example_a_the_coincident_offsets_of_a_gap_exactly_twice_the_margin_are_no_seam():
    """At width 2 m the two 1.0 um offsets meet on x = 0 with band 1 on both sides: no carrier lies there."""

    def vertical_seams(model):
        """x (grid units) of the straight margin-1.0 seams within 1 um of the gap centre."""
        family = next(
            f for f in model.labelled.catalogue.families if f.family_id == "sa"
        )
        outer = {s.carrier_id for s in family.regions.seams if s.band == 2}
        return sorted(
            c.p[0]
            for c in family.carriers
            if c.id in outer and c.p[0] == c.q[0] and abs(c.p[0]) < 1000
        )

    exact = build_model_from_catalogue(gap_model(2.0), "A", insets=(sa_bands(),))
    assert vertical_seams(exact) == []
    wide = build_model_from_catalogue(gap_model(2.6), "A", insets=(sa_bands(),))
    assert vertical_seams(wide) == [
        -300,
        300,
    ]  # the band-2 strip's two seams, 0.3 um from the centre


# --------------------------------------------------------------------------- example B: a square


def square_model() -> PrismCatalogue:
    """A 10 x 10 um Route A sheet in the middle of an 80 um die: every edge faces exposed substrate."""
    return slab_model("A", [("pad", "N", (-5.0, -5.0, 5.0, 5.0))])


def square_bands() -> InsetRequest:
    """Bands on the sheet at 2 and 4 um from its edges."""
    return InsetRequest("pad", ConductorSurface("pad", "sheet"), (2.0, 4.0))


def test_example_b_a_square_has_bands_of_64_32_and_4_square_micrometres():
    """The 2 um contour encloses 6 x 6 um, the 4 um contour 2 x 2 um: 100 - 36, 36 - 4 and 4 um^2."""
    model = build_model_from_catalogue(square_model(), "A", insets=(square_bands(),))
    assert not model.report.failures
    assert band_areas(model, "pad") == pytest.approx(
        {0: 64.0, 1: 32.0, 2: 4.0}, rel=1e-9
    )
    assert sorted(band for band, *_ in band_faces(model, "pad")) == [0, 1, 2]
    # each contour is one loop of four seams; the offsets' overlaps past the corners are no seam
    assert sorted(s.band for s in seams_of(model, "pad")) == [1] * 4 + [2] * 4
    assert band_at(model, "pad", 0.0, 0.0) == 2


# --------------------------------------------------------------------------- injected faults


def dropping(monkeypatch, family_id: str, keep):
    """Drop some seams of one family from the level's noding input, leaving the family's records as built."""
    original = level_layers.family_segments

    def without(family):
        segments = original(family)
        if family.family_id != family_id:
            return segments
        seams = {s.carrier_id: s for s in family.regions.seams}
        return [s for s in segments if keep(seams[next(iter(s.carriers)).curve_id])]

    monkeypatch.setattr(level_layers, "family_segments", without)


def test_a_missing_segment_of_a_seam_is_refused(monkeypatch):
    """Example B with one side of the 4 um square lost before Imprint."""
    model = build_model_from_catalogue(square_model(), "A", insets=(square_bands(),))
    lost = next(s.carrier_id for s in seams_of(model, "pad") if s.band == 2)
    dropping(monkeypatch, "pad", lambda s: s.carrier_id != lost)
    with pytest.raises(
        ValidationError,
        match=r"Imprint lost band boundaries .* not covered by level edges",
    ):
        build_model_from_catalogue(square_model(), "A", insets=(square_bands(),))


def test_a_missing_loop_is_refused(monkeypatch):
    """Example B without its whole 4 um loop: the 6 x 6 um face would span bands 1 and 2 with every corner at 2 um."""
    dropping(monkeypatch, "pad", lambda s: s.band != 2)
    with pytest.raises(ValidationError, match="not covered by level edges") as refused:
        build_model_from_catalogue(square_model(), "A", insets=(square_bands(),))
    assert "(4.0000" in str(refused.value) or "(-1.0000" in str(refused.value)


def test_a_missing_seam_of_a_gap_band_is_refused(monkeypatch):
    """Example A, the gap wider than 2 m: the band-2 strip's seams lost."""
    dropping(monkeypatch, "sa", lambda s: s.band != 2)
    with pytest.raises(ValidationError, match="not covered by level edges"):
        build_model_from_catalogue(gap_model(2.6), "A", insets=(sa_bands(),))


def test_a_seam_ending_on_the_die_edge_is_clipping_not_loss():
    """A sheet 0.3 um from the die edge: its SA seams reach the die edge (no origin) and the build passes."""
    catalogue = slab_model("A", [("pad", "N", (-39.7, -10.0, -20.0, 10.0))])
    model = build_model_from_catalogue(catalogue, "A", insets=(sa_bands(),))
    assert not model.report.failures
    assert band_at(model, "sa", -39.85, 0.0) == 0
    assert band_at(model, "sa", -19.25, 0.0) == 1
    assert band_at(model, "sa", -15.0, 0.0) == 2


def test_a_conductor_beside_the_square_changes_no_band_of_it():
    """Another conductor on the level is an independent edge: the square keeps its regions and areas."""
    catalogue = slab_model("A", [("pad", "N", (-5.0, -5.0, 5.0, 5.0))])
    other = conductor("other", rect_um(10.0, -5.0, 20.0, 5.0), 0, 0, "M")
    catalogue = PrismCatalogue(
        catalogue.grid, (*catalogue.prisms, other), name=catalogue.name
    )
    model = build_model_from_catalogue(catalogue, "A", insets=(square_bands(),))
    assert band_areas(model, "pad") == pytest.approx(
        {0: 64.0, 1: 32.0, 2: 4.0}, rel=1e-9
    )


# --------------------------------------------------------------------------- hand-calculated erosions


def route_b_pad(points=None, *, fp=None, arcs=()) -> PrismCatalogue:
    """One 0.2 um Route B conductor ``pad`` on the substrate surface of an 80 um die."""
    footprint_ = fp if fp is not None else footprint(points)
    pad = PrismRecord(
        "pad",
        footprint_,
        exact_z(0),
        exact_z("0.2"),
        "Al",
        "conductor",
        net_id="N",
        arcs=arcs,
    )
    return PrismCatalogue(GRID, (*chip_prisms(GRID, 40.0), pad), name="pad")


def type_bands() -> tuple[InsetRequest, ...]:
    """MA, MS and SA bands at (0.5, 1.0) um, asked for by type."""
    return tuple(
        InsetRequest(t.lower(), InterfaceSurfaces(t), MARGINS)
        for t in ("MA", "MS", "SA")
    )


def interior_angles(points) -> list[float]:
    """Interior angle (radians) at each vertex of a counter-clockwise polygon."""
    out = []
    for i, (bx, by) in enumerate(points):
        ax, ay = points[i - 1]
        cx, cy = points[(i + 1) % len(points)]
        turn = math.atan2(
            (bx - ax) * (cy - by) - (by - ay) * (cx - bx),
            (bx - ax) * (cx - bx) + (by - ay) * (cy - by),
        )
        out.append(math.pi - turn)
    return out


def polygon_bands(points) -> tuple[dict[int, float], dict[int, float]]:
    """Band areas on a polygon's top face and on the SA interface round it, for features wider than 2 m.

    At least r inside: A - P r + r^2 (sum of cot(a/2) over convex corners - sum of (a - pi)/2 over reflex
    ones).  Within r outside: P r + r^2 (sum of (pi - a)/2 over convex corners - sum of cot((2 pi - a)/2) over
    reflex ones).  The die is 80 x 80 um.
    """
    area = (
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True)
        )
        / 2
    )
    perimeter = sum(
        math.dist(p, q) for p, q in zip(points, points[1:] + points[:1], strict=True)
    )
    alphas = interior_angles(points)
    inner_corner = sum(
        1 / math.tan(a / 2) if a < math.pi else -(a - math.pi) / 2 for a in alphas
    )
    outer_corner = sum(
        (math.pi - a) / 2 if a < math.pi else -1 / math.tan((2 * math.pi - a) / 2)
        for a in alphas
    )
    inner = [area - perimeter * r + r * r * inner_corner for r in MARGINS]
    outer = [perimeter * r + r * r * outer_corner for r in MARGINS]
    top = {0: area - inner[0], 1: inner[0] - inner[1], 2: inner[1]}
    sa = {0: outer[0], 1: outer[1] - outer[0], 2: 6400 - area - outer[1]}
    return top, sa


SA_FAMILY = "sa::substrate|vacuum::z=0.0"
L_SHAPE = [
    (-10.0, -10.0),
    (10.0, -10.0),
    (10.0, 0.0),
    (0.0, 0.0),
    (0.0, 10.0),
    (-10.0, 10.0),
]
U_SHAPE = [
    (-12.0, -10.0),
    (12.0, -10.0),
    (12.0, 10.0),
    (6.0, 10.0),
    (6.0, -4.0),
    (-6.0, -4.0),
    (-6.0, 10.0),
    (-12.0, 10.0),
]


@pytest.mark.parametrize("points", [L_SHAPE, U_SHAPE], ids=["L", "U"])
def test_reflex_corners_give_the_hand_calculated_bands(points):
    """L and U conductors: the reflex corners' arcs inside, their sharp convex counterparts outside."""
    model = build_model_from_catalogue(route_b_pad(points), "B", insets=type_bands())
    top, sa = polygon_bands(points)
    for family in ("ma::pad::top", "ms::pad::bottom"):
        assert band_areas(model, family) == pytest.approx(top, rel=1e-9)
    assert band_areas(model, SA_FAMILY) == pytest.approx(sa, rel=1e-9)


@pytest.mark.parametrize(
    "points",
    [
        [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0)],
        [
            (
                10 * math.cos(math.pi / 8 + k * math.pi / 4),
                10 * math.sin(math.pi / 8 + k * math.pi / 4),
            )
            for k in range(8)
        ],
    ],
    ids=["45-degree triangle", "octagon"],
)
def test_oblique_edges_give_the_hand_calculated_bands_within_the_snap_bound(points):
    """An oblique offset ends on rounded lattice points: each contour lies within sqrt(2)/2 grid of the exact one."""
    points = [(round(x, 3), round(y, 3)) for x, y in points]
    model = build_model_from_catalogue(route_b_pad(points), "B", insets=type_bands())
    top, sa = polygon_bands(points)
    perimeter = sum(
        math.dist(p, q) for p, q in zip(points, points[1:] + points[:1], strict=True)
    )
    bound = 2 * perimeter * math.sqrt(2) / 2 * GRID.size_um  # two contours bound a band
    for family, expected in (("ma::pad::top", top), (SA_FAMILY, sa)):
        got = band_areas(model, family)
        assert sorted(got) == [0, 1, 2]
        for band, area in expected.items():
            assert abs(got[band] - area) <= bound, (family, band, got[band], area)


def test_a_strip_narrower_than_twice_the_margin_has_an_empty_upper_band():
    """A 1.6 x 20 um strip: nothing is 1.0 um from both long edges, so band 2 holds no face; it is no error."""
    model = build_model_from_catalogue(
        route_b_pad(rect_um(-0.8, -10.0, 0.8, 10.0)), "B", insets=type_bands()
    )
    half_inside = (1.6 - 1.0) * (20 - 1.0)
    assert band_areas(model, "ma::pad::top") == pytest.approx(
        {0: 32 - half_inside, 1: half_inside}, rel=1e-9
    )
    family = next(
        f for f in model.labelled.catalogue.families if f.family_id == "ma::pad::top"
    )
    assert [r.rings for r in family.regions.regions][1] == ()
    assert not [s for s in family.regions.seams if s.band == 2]


def test_a_pad_with_a_circular_hole_gives_the_hand_calculated_bands():
    """A 30 um square with a hole of radius 5 um (exact arcs): offsets of the hole are concentric circles."""
    outer = [(units(x), units(y)) for x, y in rect_um(-15, -15, 15, 15)]
    ring = [LineEdge(outer[i], outer[(i + 1) % 4], i) for i in range(4)]
    hole = circle_arcs(
        "pad/hole", (0, 0), units(5), False, Lineage("polygon_edge", "pad", 0, 1, 0)
    )
    fp, carried = Footprint.from_primitives(ring, [hole])
    model = build_model_from_catalogue(
        route_b_pad(fp=fp, arcs=carried), "B", insets=type_bands()
    )
    inner = [(30 - 2 * r) ** 2 - math.pi * (5 + r) ** 2 for r in MARGINS]
    area = 900 - 25 * math.pi
    assert band_areas(model, "ma::pad::top") == pytest.approx(
        {0: area - inner[0], 1: inner[0] - inner[1], 2: inner[1]}, rel=1e-9
    )
    ring_faces = [band for band, *_ in band_faces(model, "ma::pad::top")]
    assert sorted(ring_faces) == [
        0,
        0,
        1,
        1,
        2,
    ]  # bands 0 and 1 run round the edge and round the hole


def test_a_narrow_neck_splits_the_inner_band_into_two_islands():
    """Two 10 um squares joined by a 1.5 um wide neck: the 1.0 um region is two islands, mirror images."""
    neck = [
        (-15.0, -5.0),
        (-5.0, -5.0),
        (-5.0, -0.75),
        (5.0, -0.75),
        (5.0, -5.0),
        (15.0, -5.0),
        (15.0, 5.0),
        (5.0, 5.0),
        (5.0, 0.75),
        (-5.0, 0.75),
        (-5.0, 5.0),
        (-15.0, 5.0),
    ]
    model = build_model_from_catalogue(route_b_pad(neck), "B", insets=type_bands())
    islands = [area for band, area, _ in band_faces(model, "ma::pad::top") if band == 2]
    assert len(islands) == 2 and islands[0] == pytest.approx(islands[1], rel=1e-12)
    assert (
        band_at(model, "ma::pad::top", 0.0, 0.0) == 1
    )  # the neck centre is 0.75 um from its edges


def test_a_polygon_circle_with_small_turns_erodes_to_one_ring_of_as_many_vertices():
    """A 256-gon of radius 20 um, 1.41 degrees per vertex: R_m is one 256-gon at both margins.

    Its neighbouring offsets used to miss each other on the lattice at 0.5 um, so R_m and its outside were one
    cell and the build refused.  The offsets of two lines now end at their crossing, one shared lattice point
    per vertex (Issue #138), so the curve closes with one vertex per source vertex.
    """
    points = [
        (20 * math.cos(2 * math.pi * k / 256), 20 * math.sin(2 * math.pi * k / 256))
        for k in range(256)
    ]
    model = build_model_from_catalogue(route_b_pad(points), "B", insets=type_bands())
    family = next(
        f for f in model.labelled.catalogue.families if f.family_id == "ma::pad::top"
    )
    assert [
        (r.margin, [len(ring) for ring in r.rings]) for r in family.regions.regions
    ] == [(500, [256]), (1000, [256])]
