"""SGB's 2D models in memory: the plane section of a validated 3D model along a cut line (issue 188).

Synthetic catalogues in grid units (``sgb_scenes``) are built in 3D and cut.  A coplanar waveguide (CPW) is
checked exactly: every edge's group and place, the levels, the faces and their areas, the groups, unions,
Surface-EPR rows and band families.  Each cut rule is broken once at a known place, and the place is read back
out of the refusal.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from sgb_scenes import (
    GRID,
    airbridge,
    chip_prisms,
    conductor,
    footprint,
    pad_in_ground,
    rect_um,
    units,
)
from test_located_refusals import on, places

from scgsim.geometry.models.section import CutLine
from scgsim.geometry.sgb.cutfamilies.requests import (
    InsetRequest,
    InterfaceSurfaces,
    RefinementRequest,
)
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.section import SectionModel, build_section_model

MARGINS = (0.5, 1.0)
CPW_CUT = CutLine((0.0, -91.0), (0.0, 91.0))


def by_type(*types: str) -> tuple[InsetRequest, ...]:
    """Bands of every face of each Surface-EPR type, at :data:`MARGINS`."""
    return tuple(InsetRequest(t.lower(), InterfaceSurfaces(t), MARGINS) for t in types)


def cpw(*, along_y: bool = False):
    """A Route B CPW 100 um long on 500 um of substrate, MA, MS and SA banded: the acceptance cross-section.

    Grounds 80 um, gaps 6 um, signal 10 um, film 0.1 um; it runs along x (``along_y``: along y) and spans the
    whole model across, so a cut from one edge to the other crosses it at a right angle.
    """

    def box(x0, y0, x1, y1):
        return rect_um(y0, x0, y1, x1) if along_y else rect_um(x0, y0, x1, y1)

    outline = footprint(box(-50, -91, 50, 91))
    prisms = [
        PrismRecord(
            "substrate", outline, exact_z(-500), exact_z(0), "si", "dielectric"
        ),
        PrismRecord("vacuum", outline, exact_z(0), exact_z(100), "vacuum", "vacuum"),
        conductor("gnd_minus", box(-50, -91, 50, -11), 0, "0.1", "ground"),
        conductor("signal", box(-50, -5, 50, 5), 0, "0.1", "signal"),
        conductor("gnd_plus", box(-50, 11, 50, 91), 0, "0.1", "ground"),
    ]
    catalogue = PrismCatalogue(GRID, tuple(prisms), name="cpw")
    return build_model_from_catalogue(catalogue, "B", insets=by_type("MA", "MS", "SA"))


@pytest.fixture(scope="module")
def cpw_model():
    """The CPW along x, built once."""
    return cpw()


@pytest.fixture(scope="module")
def cpw_section(cpw_model):
    """The CPW cut across at x = 0."""
    return build_section_model(cpw_model, CPW_CUT)


def edges_by_place(
    section: SectionModel,
) -> list[tuple[str, int, int, Fraction, Fraction]]:
    """Every edge as ``(group, u0, u1, z0, z1)``, ``u`` in grid units, sorted."""
    out = []
    for edge in section.edges.values():
        a, b = section.vertices[edge.v0], section.vertices[edge.v1]
        out.append((edge.group, a.u, b.u, a.z, b.z))
    return sorted(out)


def faces_by_domain(section: SectionModel) -> dict[str, list]:
    """Owning Entity -> the faces it fills."""
    out: dict[str, list] = {}
    for face in section.faces.values():
        out.setdefault(face.domain.name, []).append(face)
    return out


def refusal(model, cut: CutLine) -> str:
    """The refusal of a cut, as text."""
    with pytest.raises(InputError) as error:
        build_section_model(model, cut)
    return str(error.value)


def at(x: float, y: float) -> tuple[float, float, float, float]:
    """A defect that is one point, as a box."""
    return (x, y, x, y)


# --------------------------------------------------------------------------- the cut line


def test_a_cut_line_runs_along_x_or_y():
    """The record keeps two float ends and knows its axis and length."""
    cut = CutLine((0, -91), (0, 91))
    assert (cut.start_um, cut.end_um, cut.axis, cut.length_um) == (
        (0.0, -91.0),
        (0.0, 91.0),
        "y",
        182.0,
    )
    assert CutLine((5.0, 2.0), (-3.0, 2.0)).axis == "x"


@pytest.mark.parametrize(
    ("ends", "error", "text"),
    [
        (((0.0, -91.0), (10.0, 91.0)), ValueError, "runs neither along x nor along y"),
        (((1.0, 1.0), (1.0, 1.0)), ValueError, "has no length"),
        (((0.0, math.inf), (0.0, 1.0)), ValueError, "must be finite"),
        ((("0", 1.0), (0.0, 2.0)), TypeError, "pair of numbers"),
        (((True, 1.0), (0.0, 2.0)), TypeError, "pair of numbers"),
        (((0.0,), (0.0, 2.0)), TypeError, "(x, y) pair"),
    ],
)
def test_a_cut_line_that_is_no_straight_cut_along_an_axis_is_refused(ends, error, text):
    """An oblique, empty or malformed cut fails where it is written."""
    with pytest.raises(error, match=text.replace("(", r"\(").replace(")", r"\)")):
        CutLine(*ends)


def test_an_oblique_cut_line_names_both_ends():
    """The message carries the two ends, in um."""
    with pytest.raises(ValueError) as error:
        CutLine((0.0, -91.0), (10.0, 91.0))
    assert places(str(error.value)) == [(0.0, -91.0, 10.0, 91.0)]


# --------------------------------------------------------------------------- the CPW, exactly


def u(value_um: float) -> int:
    """A distance along the cut in grid units."""
    return units(value_um)


Z0, ZF = Fraction(0), Fraction(1, 10)
# The CPW's place along the cut x = 0 from y = -91 um: ground [0, 80], gap [80, 86], signal [86, 96],
# gap [96, 102], ground [102, 182] um.  Each band k of a face lies between margin k - 1 and margin k of the
# nearest gap edge (a conductor wall in the section): bands 0 and 1 are 0.5 um wide, band 2 is the rest.
HORIZONTAL = [
    ("EXT::substrate__bottom", 0, 182, Fraction(-500)),
    ("EXT::vacuum__top", 0, 182, Fraction(100)),
    ("MS::gnd_minus__bottom::band=2", 0, 79, Z0),
    ("MS::gnd_minus__bottom::band=1", 79, 79.5, Z0),
    ("MS::gnd_minus__bottom::band=0", 79.5, 80, Z0),
    ("SA::substrate|vacuum__interface::band=0", 80, 80.5, Z0),
    ("SA::substrate|vacuum__interface::band=1", 80.5, 81, Z0),
    ("SA::substrate|vacuum__interface::band=2", 81, 85, Z0),
    ("SA::substrate|vacuum__interface::band=1", 85, 85.5, Z0),
    ("SA::substrate|vacuum__interface::band=0", 85.5, 86, Z0),
    ("MS::signal__bottom::band=0", 86, 86.5, Z0),
    ("MS::signal__bottom::band=1", 86.5, 87, Z0),
    ("MS::signal__bottom::band=2", 87, 95, Z0),
    ("MS::signal__bottom::band=1", 95, 95.5, Z0),
    ("MS::signal__bottom::band=0", 95.5, 96, Z0),
    ("SA::substrate|vacuum__interface::band=0", 96, 96.5, Z0),
    ("SA::substrate|vacuum__interface::band=1", 96.5, 97, Z0),
    ("SA::substrate|vacuum__interface::band=2", 97, 101, Z0),
    ("SA::substrate|vacuum__interface::band=1", 101, 101.5, Z0),
    ("SA::substrate|vacuum__interface::band=0", 101.5, 102, Z0),
    ("MS::gnd_plus__bottom::band=0", 102, 102.5, Z0),
    ("MS::gnd_plus__bottom::band=1", 102.5, 103, Z0),
    ("MS::gnd_plus__bottom::band=2", 103, 182, Z0),
    ("MA::gnd_minus__top::band=2", 0, 79, ZF),
    ("MA::gnd_minus__top::band=1", 79, 79.5, ZF),
    ("MA::gnd_minus__top::band=0", 79.5, 80, ZF),
    ("MA::signal__top::band=0", 86, 86.5, ZF),
    ("MA::signal__top::band=1", 86.5, 87, ZF),
    ("MA::signal__top::band=2", 87, 95, ZF),
    ("MA::signal__top::band=1", 95, 95.5, ZF),
    ("MA::signal__top::band=0", 95.5, 96, ZF),
    ("MA::gnd_plus__top::band=0", 102, 102.5, ZF),
    ("MA::gnd_plus__top::band=1", 102.5, 103, ZF),
    ("MA::gnd_plus__top::band=2", 103, 182, ZF),
]
# The walls the cut crosses: the film's sidewalls (unbanded) and the model's two ends.
VERTICAL = [
    ("EXT::substrate__sidewall", 0, Fraction(-500), Z0),
    ("EXT::substrate__sidewall", 182, Fraction(-500), Z0),
    ("EXT::vacuum__sidewall", 0, ZF, Fraction(100)),
    ("EXT::vacuum__sidewall", 182, ZF, Fraction(100)),
    ("MA::gnd_minus__sidewall", 80, Z0, ZF),
    ("MA::signal__sidewall", 86, Z0, ZF),
    ("MA::signal__sidewall", 96, Z0, ZF),
    ("MA::gnd_plus__sidewall", 102, Z0, ZF),
]


def expected_cpw_edges():
    """:data:`HORIZONTAL` and :data:`VERTICAL` in the form :func:`edges_by_place` gives."""
    return sorted(
        [(g, u(a), u(b), z, z) for g, a, b, z in HORIZONTAL]
        + [(g, u(a), u(a), z0, z1) for g, a, z0, z1 in VERTICAL]
    )


def test_the_cpw_section_has_the_trace_of_every_face_the_cut_crosses(cpw_section):
    """Each edge lies where its 3D face meets the cut, in that face's group; the film is 0.1 um at z = 0."""
    assert edges_by_place(cpw_section) == expected_cpw_edges()
    levels = sorted({v.z for v in cpw_section.vertices.values()})
    assert levels == [Fraction(-500), Z0, ZF, Fraction(100)]


def test_every_edge_is_the_trace_of_one_emitted_face_of_the_3d_model(
    cpw_model, cpw_section
):
    """An edge's group is its 3D face's, and a wall's edge stands where the cut crosses its source curve."""
    for edge in cpw_section.edges.values():
        face = cpw_model.topology.faces[edge.trace]
        assert cpw_model.routed.face_group[edge.trace] == edge.group
        assert face.kind == ("horizontal" if edge.kind == "horizontal" else "wall")
        assert edge.bands == face.bands
        assert bool(edge.curves) == (edge.kind == "vertical")
    sidewall = next(
        e for e in cpw_section.edges.values() if e.group == "MA::gnd_minus__sidewall"
    )
    assert sidewall.curves == ("gnd_minus/polygon0/ring0/edge2",)


def test_the_cpw_section_has_one_substrate_face_and_one_vacuum_face(cpw_section):
    """The conductors are excluded (Route B): the vacuum reaches into both gaps, the substrate is a rectangle."""
    faces = faces_by_domain(cpw_section)
    assert sorted(faces) == ["substrate", "vacuum"]
    (substrate,), (vacuum,) = faces["substrate"], faces["vacuum"]
    to_um2 = Fraction(1, 1000)  # grid units x um -> um^2
    assert substrate.twice_area * to_um2 / 2 == 182 * 500
    above_the_film, in_the_gaps = 182 * Fraction("99.9"), 2 * 6 * Fraction("0.1")
    assert vacuum.twice_area * to_um2 / 2 == above_the_film + in_the_gaps
    assert [len(loop) for loop in substrate.loops] == [24]
    assert [len(loop) for loop in vacuum.loops] == [28]
    sides = {
        (e.group, e.left is not None, e.right is not None)
        for e in cpw_section.edges.values()
    }
    assert (
        "MA::gnd_minus__sidewall",
        False,
        True,
    ) in sides  # metal before it, the gap's vacuum after
    assert (
        "MS::signal__bottom::band=2",
        False,
        True,
    ) in sides  # metal above it, the substrate below
    assert ("SA::substrate|vacuum__interface::band=2", True, True) in sides


def test_the_cpw_section_keeps_the_3d_groups_unions_rows_and_bands_it_crosses(
    cpw_section,
):
    """Groups have dim 1 (edges) and 2 (faces); unions, rows and band families are the 3D ones, restricted."""
    groups = cpw_section.groups
    assert {name for name, g in groups.items() if g.dim == 2} == {"substrate", "vacuum"}
    assert {g.dim for g in groups.values()} == {1, 2}
    assert len(groups["SA::substrate|vacuum__interface::band=0"].ids) == 4
    assert sorted(cpw_section.unions) == [
        "EXT",
        "MA",
        "MS",
        "NET::ground",
        "NET::signal",
        "SA",
    ]
    assert cpw_section.unions["SA"] == [
        f"SA::substrate|vacuum__interface::band={k}" for k in range(3)
    ]
    assert "MA::signal__sidewall" in cpw_section.unions["NET::signal"]
    rows = cpw_section.surface_epr_rows
    assert len(rows) == 21 and all(rows[row] == row for row in rows)
    assert sorted(r.request_id for r in cpw_section.band_families) == sorted(
        f"{t}::{e}::{s}"
        for t, s in (("ma", "top"), ("ms", "bottom"))
        for e in ("gnd_minus", "gnd_plus", "signal")
    ) + ["sa::substrate|vacuum::z=0.0"]
    assert {r.margins_um for r in cpw_section.band_families} == {MARGINS}
    assert cpw_section.report.failures == []


def test_a_cut_along_x_of_a_line_along_y_is_the_same_section():
    """Turning the layout and the cut together changes nothing in the section's frame."""
    section = build_section_model(cpw(along_y=True), CutLine((-91.0, 0.0), (91.0, 0.0)))
    assert edges_by_place(section) == expected_cpw_edges()


def test_a_cut_drawn_the_other_way_mirrors_u(cpw_model):
    """``u`` runs from the cut's start, so swapping its ends mirrors the section."""
    section = build_section_model(cpw_model, CutLine((0.0, 91.0), (0.0, -91.0)))
    length = u(182)
    mirrored = sorted(
        (g, length - b, length - a, z0, z1) for g, a, b, z0, z1 in expected_cpw_edges()
    )
    assert edges_by_place(section) == mirrored


# --------------------------------------------------------------------------- faces, holes and a route A sheet


def test_a_floating_conductor_is_a_hole_in_the_vacuum_face():
    """The vacuum face has its outer loop first (counter-clockwise) and the island as a clockwise hole."""
    prisms = chip_prisms(GRID, 40.0)
    prisms.append(conductor("island", rect_um(-10, -10, 10, 10), 5, "5.2", "F"))
    model = build_model_from_catalogue(PrismCatalogue(GRID, tuple(prisms)), "B")
    section = build_section_model(model, CutLine((0.0, -40.0), (0.0, 40.0)))
    (vacuum,) = faces_by_domain(section)["vacuum"]
    assert [len(loop) for loop in vacuum.loops] == [4, 4]
    hole = {section.edges[e].group for e, _ in vacuum.loops[1]}
    assert hole == {"MA::island__bottom", "MA::island__top", "MA::island__sidewall"}
    assert vacuum.twice_area * Fraction(1, 1000) / 2 == 80 * 100 - 20 * Fraction(2, 10)


def test_a_volume_the_cut_crosses_twice_is_two_faces():
    """Under the airbridge the vacuum is cut off from the rest in the section, though it is one 3D volume."""
    model = build_model_from_catalogue(airbridge(), "B")
    section = build_section_model(model, CutLine((-150.0, 0.0), (150.0, 0.0)))
    vacuum = faces_by_domain(section)["vacuum"]
    assert len(vacuum) == 2 and len({face.volume for face in vacuum}) == 1
    under = min(vacuum, key=lambda face: face.twice_area)
    # Between the posts (|x| < 25 um) from the ground's top to the deck, and the 40 um gap down to the substrate.
    assert under.twice_area * Fraction(1, 1000) / 2 == 50 * Fraction(
        28, 10
    ) + 40 * Fraction(2, 10)
    assert len(section.groups["vacuum"].ids) == 2


def test_a_route_a_sheet_is_an_edge_between_the_substrate_and_the_vacuum():
    """A zero-thickness sheet's trace carries its owner and its group, the bare entity id."""
    prisms = chip_prisms(GRID, 40.0)
    prisms.append(conductor("pad", rect_um(-10, -10, 10, 10), 0, 0, "N"))
    model = build_model_from_catalogue(PrismCatalogue(GRID, tuple(prisms)), "A")
    section = build_section_model(model, CutLine((0.0, -40.0), (0.0, 40.0)))
    (sheet,) = [e for e in section.edges.values() if e.group == "pad"]
    assert sheet.owner is not None and sheet.owner.name == "pad"
    assert (sheet.left_domain.name, sheet.right_domain.name) == ("vacuum", "substrate")
    a, b = section.vertices[sheet.v0], section.vertices[sheet.v1]
    assert (a.u, b.u, a.z) == (u(30), u(50), 0)


# --------------------------------------------------------------------------- the cut rules, located


def test_a_cut_along_an_edge_is_refused_with_the_stretch(cpw_model):
    """A cut along the signal's lower edge lies on the edge, so it has no section."""
    message = refusal(cpw_model, CutLine((-50.0, -5.0), (50.0, -5.0)))
    assert "runs along an edge of" in message and "signal" in message
    assert "on level z = 0 um" in message and "Fix the cut line" in message
    assert on(places(message)[1], (-50, -5, 50, -5))


def test_a_cut_through_a_corner_is_refused_at_the_corner():
    """The cut x = 0 meets the diamond pad only at its two tips; it names the nearest one."""
    prisms = chip_prisms(GRID, 40.0)
    diamond = [(0, -20), (20, 0), (0, 20), (-20, 0)]
    prisms.append(conductor("diamond", diamond, 0, "0.2", "N"))
    model = build_model_from_catalogue(PrismCatalogue(GRID, tuple(prisms)), "B")
    message = refusal(model, CutLine((0.0, -40.0), (0.0, 40.0)))
    assert "passes through a corner at (0.000, -20.000) um on level z = 0 um" in message
    assert "edges of diamond meet" in message
    assert "at 3 more places" in message  # the other tip, on both film levels


def test_an_oblique_crossing_is_refused_with_its_angle():
    """A slanted strip crosses the cut at about 79 degrees, not 90: its section would be too wide."""
    prisms = chip_prisms(GRID, 40.0)
    strip = [(-40, -12), (40, 4), (40, 12), (-40, -4)]
    prisms.append(conductor("strip", strip, 0, "0.2", "N"))
    model = build_model_from_catalogue(PrismCatalogue(GRID, tuple(prisms)), "B")
    message = refusal(model, CutLine((0.0, -40.0), (0.0, 40.0)))
    assert "crosses an edge of strip at (0.000, -4.000) um on level z = 0 um" in message
    assert (
        f"at {math.degrees(math.atan2(80, 16)):.1f} degrees instead of a right angle"
        in message
    )
    assert "plausible but wrong L and C" in message


def test_a_cut_across_an_arc_is_refused_at_the_arc():
    """The cut x = -22 um crosses the pad's circular hole (centre (-25, 5), radius 8) at y = 5 - sqrt(55)."""
    model = build_model_from_catalogue(pad_in_ground("B"), "B")
    message = refusal(model, CutLine((-22.0, -150.0), (-22.0, 150.0)))
    assert "crosses an arc of pad" in message and "radius 8 um" in message
    y = 5 - math.sqrt(64 - 9)
    assert on(places(message)[1], at(-22, y))
    assert on(places(message)[2], at(-25, 5))  # the centre
    assert "move it onto a straight section of the line" in message


def test_a_cut_across_a_band_arc_is_refused_with_its_family_and_margin():
    """MA bands of an L-shaped pad turn round its reflex corner (0, 0) on arcs of radius 0.5 and 1 um."""
    prisms = chip_prisms(GRID, 40.0)
    pad = [(-10, -10), (10, -10), (10, 0), (0, 0), (0, 10), (-10, 10)]
    prisms.append(conductor("pad", pad, 0, "0.2", "N"))
    catalogue = PrismCatalogue(GRID, tuple(prisms))
    model = build_model_from_catalogue(catalogue, "B", insets=by_type("MA"))
    message = refusal(model, CutLine((-0.3, -40.0), (-0.3, 40.0)))
    assert (
        "crosses the arc of the Surface-EPR band seam of family 'ma::pad::top' at margin 1 um"
        in message
    )
    assert on(places(message)[1], at(-0.3, -math.sqrt(1 - 0.09)))
    assert "on level z = 0.2 um" in message
    assert "farther than the largest margin from the corner" in message


def test_a_cut_beyond_the_model_is_refused_with_the_extent():
    """An end 10 um past the chip's edge lies outside the model's planar extent."""
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(chip_prisms(GRID, 40.0))), "B"
    )
    message = refusal(model, CutLine((0.0, -50.0), (0.0, 40.0)))
    assert "runs outside the model" in message
    assert places(message)[1:] == [at(0, -50), (-40.0, -40.0, 40.0, 40.0)]


def test_a_cut_that_ends_inside_the_model_is_refused_at_its_end():
    """A section is the whole cross-section: an end 10 um inside the chip is refused."""
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(chip_prisms(GRID, 40.0))), "B"
    )
    message = refusal(model, CutLine((0.0, -30.0), (0.0, 40.0)))
    assert "ends at (0.000, -30.000) um, inside the model" in message
    assert (
        "extend it to the edge of the extent (-40.000, -40.000)..(40.000, 40.000) um"
        in message
    )


def wide_vacuum(*conductors: PrismRecord):
    """A 80 x 80 um substrate under a vacuum that reaches 20 um past its edge at y = 40 um."""
    substrate = footprint(rect_um(-40, -40, 40, 40))
    vacuum = footprint(rect_um(-40, -40, 40, 60))
    prisms = [
        PrismRecord(
            "substrate", substrate, exact_z(-100), exact_z(0), "si", "dielectric"
        ),
        PrismRecord("vacuum", vacuum, exact_z(0), exact_z(100), "vacuum", "vacuum"),
        *conductors,
    ]
    return build_model_from_catalogue(PrismCatalogue(GRID, tuple(prisms)), "B")


def test_a_cut_that_ends_where_one_region_goes_on_is_refused_with_that_region():
    """Ending at the substrate's edge leaves out the vacuum beyond it."""
    message = refusal(wide_vacuum(), CutLine((0.0, -40.0), (0.0, 40.0)))
    assert (
        "ends at (0.000, 40.000) um, inside the model: vacuum continues past it"
        in message
    )
    assert "at z 0..100 um" in message


def test_a_conductor_that_touches_the_model_edge_at_a_point_in_the_section_is_refused():
    """A pad flush with the substrate's edge, under a vacuum that goes on, meets the outside only at a point."""
    pad = conductor("pad", rect_um(-10, 20, 10, 40), 0, "0.2", "N")
    message = refusal(wide_vacuum(pad), CutLine((0.0, -40.0), (0.0, 60.0)))
    assert (
        "cuts pad and the edge of the model where they touch only at a point" in message
    )
    assert "(0.000, 40.000) um at z = 0 um" in message
    assert "Fix the layout or the cut line" in message


def test_a_cut_end_off_the_grid_is_refused():
    """Both ends must be lattice points of the model grid."""
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(chip_prisms(GRID, 40.0))), "B"
    )
    message = refusal(model, CutLine((0.0005, -40.0), (0.0005, 40.0)))
    assert "start_um (0.0005, -40.0) um is off the 0.001 um model grid" in message


def test_a_model_with_refinement_curves_is_refused():
    """A 2D model has no place for mesh-size curves, so they are refused rather than lost."""
    catalogue = PrismCatalogue(GRID, tuple(chip_prisms(GRID, 40.0)))
    line = RefinementRequest("r", 0.0, ((-5.0, 0.0), (5.0, 0.0)))
    model = build_model_from_catalogue(catalogue, "B", refinements=(line,))
    message = refusal(model, CutLine((0.0, -40.0), (0.0, 40.0)))
    assert "carries refinement curves (r)" in message


def test_every_refusal_links_to_the_cut_rules():
    """The located refusals end with the link to their explanation when the docs are published."""
    model = build_model_from_catalogue(
        PrismCatalogue(GRID, tuple(chip_prisms(GRID, 40.0))), "B"
    )
    message = refusal(model, CutLine((0.0, -30.0), (0.0, 40.0)))
    assert message.endswith("geometry-sgb.html#cut-rules") or "See " not in message
