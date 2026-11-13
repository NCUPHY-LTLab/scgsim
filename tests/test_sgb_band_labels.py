"""A face's band comes from the band regions built before Imprint, never from one sample point on a band boundary.

Where two offsets coincide (a gap exactly twice the margin wide) the distance to the origins equals
the margin on a line, and a point sampled on that line would put its face one band too high.  The region
``dist >= m`` has no area there, so it has no cell, no seam and no face.  Bands are measured from gap edges
only: conductor edges with exposed dielectric next to them.
"""

from __future__ import annotations

import math

import pytest
from sgb_gmsh_oracle import gmsh_session, mesh_coarsely
from sgb_inputs import gds_pad_in_ground, thin_film_pair
from sgb_scenes import slab_model

from scgsim.contracts.sgb_sidecar import read_sidecar, views
from scgsim.geometry.sgb import build_sgb_geometry
from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.arrangement.dcel import (
    Arrangement,
    build_arrangement,
    interior_point,
    strip_cells,
)
from scgsim.geometry.sgb.arrangement.labelling.bands import (
    check_seams_kept,
    face_bands,
    level_supports,
)
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import Segment, snap_round
from scgsim.geometry.sgb.bands.inset import inset_family
from scgsim.geometry.sgb.bands.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.checks.verification.loaded import (
    apply_sidecar_embeds,
    verify_xao,
)
from scgsim.geometry.sgb.errors import InputError, ValidationError
from scgsim.geometry.sgb.exact.carriers import EdgeCarrier
from scgsim.geometry.sgb.exact.grid import Grid
from scgsim.geometry.sgb.exact.prisms import LineEdge
from scgsim.geometry.sgb.exact.support import DISTANCE_SLACK, SupportRegion
from scgsim.geometry.sgb.model import (
    ConformalModel,
    build_conformal_model,
    build_model_from_catalogue,
)
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue

MARGINS = (0.5, 1.0)
SA = SolutionInterface("substrate", "upper_air", z_um=0)


def rectangle(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Counter-clockwise corners."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def arrange(ring: list[tuple[int, int]]) -> Arrangement:
    """The arrangement of one closed ring: a single bounded face."""
    count = len(ring)
    segments = [
        Segment(ring[i], ring[(i + 1) % count], frozenset([EdgeCarrier(f"e{i}")]))
        for i in range(count)
    ]
    return build_arrangement(snap_round(segments).edges)


def support_of(ring: list[tuple[int, int]], margins: tuple[int, ...]) -> SupportRegion:
    """A support region bounded by one polygonal ring."""
    count = len(ring)
    edges = tuple(LineEdge(ring[i], ring[(i + 1) % count]) for i in range(count))
    return SupportRegion((edges,), margins)


# --------------------------------------------------------------------------- the rule on one face


def banded(ring: list[tuple[int, int]], margins: tuple[int, ...]):
    """The band family of a support bounded by one polygonal ring, ready for labelling."""
    family = inset_family("f", support_of(ring, margins), (0,))
    return family, level_supports([family])


def test_a_face_on_the_medial_axis_takes_the_band_of_its_area_not_of_its_midline():
    """A strip twice as wide as the margin: d == margin on the midline only, so no region and no seam."""
    ring = rectangle(0, 0, 2000, 10_000)
    arr = arrange(ring)
    support = support_of(ring, (1000,))
    assert support.band_of(interior_point(arr, 1)) == 1  # the single-point rule
    family, supports = banded(ring, (1000,))
    assert family.carriers == () and family.arcs == ()
    assert family.regions.regions[0].rings == ()
    assert face_bands(interior_point(arr, 1), supports) == (("f", 0),)


def test_a_face_outside_the_support_has_no_band():
    """Outside the witness polygon of the region, whatever the distance."""
    arr = arrange(rectangle(5000, 0, 6000, 1000))
    _, supports = banded(rectangle(0, 0, 2000, 2000), (500,))
    assert face_bands(interior_point(arr, 1), supports) == ()


def test_a_face_that_holds_area_in_two_bands_fails_closed():
    """A missing cut: the level holds the support but not the seam, so the Imprint check refuses it."""
    ring = rectangle(0, 0, 10_000, 10_000)
    family, supports = banded(ring, (2000,))
    assert len(family.regions.seams) == 4  # the 6 x 6 square at 2000 from every edge
    arr = arrange(ring)
    bands = [{}, dict(face_bands(interior_point(arr, 1), supports))]
    with pytest.raises(ValidationError, match="not covered by level edges"):
        check_seams_kept(arr, bands, set(), [family], Grid(0.001), "z = 0")


def test_a_port_face_across_two_bands_is_nominally_in_band_zero_inside_the_support():
    """The band seams stop at a port's boundary, so its face may straddle: it carries band 0, none outside."""
    ring = rectangle(0, 0, 10_000, 10_000)
    arr = arrange(ring)
    _, supports = banded(ring, (2000,))
    assert face_bands(interior_point(arr, 1), supports, nominal=True) == (("f", 0),)
    _, outside = banded(rectangle(20_000, 0, 30_000, 10_000), (2000,))
    assert face_bands(interior_point(arr, 1), outside, nominal=True) == ()


def test_strip_cells_add_up_to_the_exact_face_area():
    """The trapezoids are an exact partition: area weights are exact fractions of the face."""
    arr = arrange([(0, 0), (6, 0), (6, 3), (3, 3), (3, 7), (0, 7)])
    assert sum(c.area for c in strip_cells(arr, 1)) * 2 == arr.face_twice_area(1)


# --------------------------------------------------------------------------- real models


Rect = tuple[float, float, float, float]


def build(catalogue, route, *insets):
    """A model that keeps each layer's raw (pre-dissolution) arrangement, the cells the bands are decided on."""
    return build_model_from_catalogue(catalogue, route, insets=insets, keep_raw=True)


def build_input(effective, route, *insets):
    """The same from an effective input (source stages S0 to S2)."""
    return build(
        build_prism_catalogue(effective.build_input, effective.representation),
        route,
        *insets,
    )


def distance_to_rects(x: float, y: float, rects: list[Rect]) -> float:
    """Distance from a point to the nearest of axis-aligned rectangles, in the rectangles' unit."""
    return min(
        math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))
        for x0, y0, x1, y1 in rects
    )


def margins_of(model: ConformalModel, family_id: str) -> tuple[int, ...]:
    """The margins (grid units) of an inset family."""
    return next(
        f.support.margins
        for f in model.labelled.catalogue.families
        if f.family_id == family_id
    )


def box_um(arr, fi, grid):
    """Bounding box of a face in um."""
    points = [p for s in arr.face_segments(fi) for p in s]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return tuple(round(v * grid, 6) for v in (min(xs), min(ys), max(xs), max(ys)))


def band_rows(model: ConformalModel, family_id: str):
    """``(bbox in um, band, sheet owner, area in um^2)`` of every dissolved level face labelled with a band."""
    grid = model.labelled.catalogue.grid.size_um
    rows = []
    for layer in model.labelled.level_layers:
        for fi in range(1, len(layer.arr.faces)):
            label = layer.labels[fi]
            for family, band in label.bands:
                if family == family_id:
                    area = face_exact_area(layer.arr, layer.edge_curve, fi) * grid**2
                    rows.append((box_um(layer.arr, fi, grid), band, label.sheet, area))
    return rows


def assert_labels_match_the_distance_to(model, family_id, rects: list[Rect]):
    """Every raw cell's clear interior samples lie in the band of their distance to the rectangles (um)."""
    grid = model.labelled.catalogue.grid.size_um
    margins = [m * grid for m in margins_of(model, family_id)]
    clearance = DISTANCE_SLACK * grid
    checked = 0
    for layer in model.labelled.level_layers:
        arr = layer.raw_arrangement
        for fi in range(1, len(arr.faces)):
            bands = [b for f, b in layer.raw_labels[fi].bands if f == family_id]
            if not bands:
                continue
            found = set()
            for cell in strip_cells(arr, fi):
                for x, y, den in cell.points:
                    d = distance_to_rects(x / den * grid, y / den * grid, rects)
                    if all(abs(d - m) > clearance for m in margins):
                        found.add(sum(1 for m in margins if d > m))
            assert found <= set(bands), (box_um(arr, fi, grid), bands, found)
            checked += 1
    assert checked


def assert_mirror_symmetric_about(model, family_id, y_mid):
    """Mirrored dissolved faces (bounding box reflected in ``y = y_mid``) carry the same band."""
    by_box = {box: band for box, band, *_ in band_rows(model, family_id)}
    mirrored = 0
    for (x0, y0, x1, y1), band in by_box.items():
        mirror = (x0, round(2 * y_mid - y1, 6), x1, round(2 * y_mid - y0, 6))
        if mirror in by_box:
            mirrored += 1
            assert by_box[mirror] == band, ((x0, y0, x1, y1), mirror)
    assert mirrored


def band_areas_without_sheets(model, family_id) -> dict[int, float]:
    """Area (um^2) of every band of a family over the cells no sheet (the port) covers."""
    areas: dict[int, float] = {}
    for _, band, sheet, area in band_rows(model, family_id):
        if sheet is None:
            areas[band] = areas.get(band, 0.0) + area
    return areas


def sa(request_id="sa", z_um=0, margins=MARGINS):
    """Bands on the substrate-to-vacuum interface at ``z_um``."""
    return InsetRequest(
        request_id, SolutionInterface("substrate", "vacuum", z_um=z_um), margins
    )


def test_two_sheets_a_gap_of_exactly_twice_the_margin_apart_never_reach_the_upper_band_in_the_gap():
    """Sheets at x = -1 and x = 1, margin 1.0 um: every gap point is within 1.0 um of a sheet."""
    sheets = [(-10.0, 0.0, -1.0, 10.0), (1.0, 0.0, 10.0, 10.0)]
    model = build(
        slab_model("A", [("left", "N1", sheets[0]), ("right", "N2", sheets[1])]),
        "A",
        sa(),
    )
    # the two offsets at 1.0 um coincide on x = 0 with band 1 on both sides: no seam there, so the band-1
    # face runs across the axis (one ring round both sheets), and no face ends on x = 0
    rows = band_rows(model, "sa")
    assert not [box for box, *_ in rows if 0 in (box[0], box[2])]
    (across,) = [row for row in rows if row[0][0] < 0 < row[0][2] and row[1] != 2]
    within = [
        90 + 2 * r * 19 + math.pi * r * r for r in MARGINS
    ]  # a 9 x 10 um sheet grown by r
    assert across[1] == 1
    assert across[3] == pytest.approx(2 * (within[1] - within[0]), rel=1e-6)
    assert_labels_match_the_distance_to(model, "sa", sheets)
    assert_mirror_symmetric_about(model, "sa", y_mid=5.0)


def test_the_die_edge_is_not_a_distance_origin_of_a_solution_interface():
    """A sheet 0.2 um from the die edge: the strip beside the edge is band 0 from the sheet, the far edge is band 2."""
    sheets = [(-39.8, -10.0, -10.0, 10.0)]
    model = build(slab_model("A", [("left", "N1", sheets[0])]), "A", sa())
    assert_labels_match_the_distance_to(model, "sa", sheets)
    corner = [
        band
        for box, band, *_ in band_rows(model, "sa")
        if box[0] == -40.0 and box[3] == 40.0
    ]
    assert corner == [2]


def test_a_metal_edge_on_the_die_edge_is_not_an_origin():
    """A sheet flush with the die edge x = -40: bands run along its other three edges only."""
    sheets = [(-40.0, -10.0, -10.0, 10.0)]
    model = build(slab_model("A", [("left", "N1", sheets[0])]), "A", sa())
    support = next(f.support for f in model.labelled.catalogue.families)
    assert not all(support.is_origin(0, i) for i in range(len(support.rings[0])))
    free = support.free_boundary
    assert free  # the die edge, with the sheet's flush edge on it
    assert_labels_match_the_distance_to(
        model, "sa", [(-80.0, -10.0, -10.0, 10.0)]
    )  # a sheet reaching past the die gives the same distances inside it


def test_a_support_with_no_gap_edge_fails_closed():
    """A ground plane over the whole die has no opening: no gap edge, no bands."""
    for request in (InsetRequest("g", ConductorSurface("pad", "sheet"), MARGINS),):
        with pytest.raises(InputError, match="no gap edge"):
            build(
                slab_model("A", [("pad", "N1", (-40.0, -40.0, 40.0, 40.0))]),
                "A",
                request,
            )


def dilation_area(r: float) -> float:
    """Area of a 4 x 4 um square grown by ``r`` um: straight strips and four quarter discs."""
    return (4 + 2 * r) ** 2 - (4 - math.pi) * r * r


def test_route_a_thin_film_pair_solution_bands_ripple_out_of_the_two_sheets():
    """Sheets x 0..4 and 6..10, y 0..4 on a die x -3..13, y -1..5; the 2 um gap is twice the margin 1.0 um."""
    model = build_input(
        thin_film_pair(),
        "A",
        InsetRequest("band_left", ConductorSurface("chip/left", "sheet"), MARGINS),
        InsetRequest("band_right", ConductorSurface("chip/right", "sheet"), MARGINS),
        InsetRequest("band_sa", SA, MARGINS),
    )
    sheets = [(0.0, 0.0, 4.0, 4.0), (6.0, 0.0, 10.0, 4.0)]
    assert_labels_match_the_distance_to(model, "band_sa", sheets)
    assert_mirror_symmetric_about(model, "band_sa", y_mid=2.0)
    # d < 0.5 and d < 1 around both squares, the die area outside them, and the port footprint (x 4..6,
    # y 1..3) that the groups leave to the port: d = min(x - 4, 6 - x) there.
    within = [2 * (dilation_area(r) - 16) for r in MARGINS]
    die = 16 * 6 - 2 * 16
    port = (2.0, 2.0, 0.0)
    expected = {
        0: within[0] - port[0],
        1: within[1] - within[0] - port[1],
        2: die - within[1] - port[2],
    }
    assert band_areas_without_sheets(model, "band_sa") == pytest.approx(
        expected, rel=1e-5
    )
    gap = {
        box: band
        for box, band, *_ in band_rows(model, "band_sa")
        if 4 <= box[0] and box[2] <= 6
    }
    assert gap[(4.0, -1.0, 6.0, 0.0)] == gap[(4.0, 4.0, 6.0, 5.0)] == 2
    # the seams at x = 4.5, 5.0 and 5.5 stop at the port boundary: the port is one face, left out of the bands
    assert [box for box in gap if box[1] >= 1.0 and box[3] <= 3.0] == [
        (4.0, 1.0, 6.0, 3.0)
    ]


def test_pad_in_ground_bands_run_along_the_cutout_edge_only(tmp_path):
    """Routes B and C: the ground's bands stop at the 40 um cutout; its outer edge on the die edge is no origin."""
    effective = gds_pad_in_ground(tmp_path)
    for route in "BC":
        sides = ("top", "bottom")
        insets = [
            InsetRequest(f"{entity}_{side}", ConductorSurface(entity, side), MARGINS)
            for entity in ("ground_0000", "pad")
            for side in sides
        ]
        insets.append(
            InsetRequest(
                "sa", SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0), MARGINS
            )
        )
        model = build_input(effective, route, *insets)
        for side in sides:
            ground = next(
                f.support
                for f in model.labelled.catalogue.families
                if f.family_id == f"ground_0000_{side}"
            )
            origins = [p for p in ground.origins]
            xs = [c for p in origins for c in (p.p[0], p.q[0])]
            assert (
                min(xs) >= 30_000 and max(xs) <= 70_000
            )  # the cutout, not the 0..100 um die edge
            assert [len(ring) for ring in ground.rings] == [5, 6]
            assert ground.free_boundary == ground.rings[0]
            pad = next(
                f.support
                for f in model.labelled.catalogue.families
                if f.family_id == f"pad_{side}"
            )
            assert pad.free_boundary == ()
        sa_support = next(
            f.support for f in model.labelled.catalogue.families if f.family_id == "sa"
        )
        assert sa_support.free_boundary == ()


# --------------------------------------------------------------------------- band seams stop at a port


def all_interface_insets(effective, route, sa_pair):
    """Bands on every supported surface of every conductor and on the SA interface at the port plane."""
    sides = ("sheet",) if route == "A" else ("top", "bottom")
    insets = [
        InsetRequest(
            f"band_{entity.semantic_id}_{side}".replace("/", "_"),
            ConductorSurface(entity.semantic_id, side),
            MARGINS,
        )
        for entity in effective.build_input.entities
        if entity.material_kind == "conductor"
        for side in sides
    ]
    insets.append(
        InsetRequest("band_sa", SolutionInterface(*sa_pair, z_um=0.0), MARGINS)
    )
    return tuple(insets)


PORT_CASES = {  # route -> (input, SA pair, port group, port area um^2, SA band areas um^2, mirror axis y in um)
    "A": (
        thin_film_pair,
        ("substrate", "upper_air"),
        "PORT::port_sheet",
        4.0,
        (15.571, 18.712, 25.717),
        2.0,
    ),
    "B": (
        gds_pad_in_ground,
        ("substrate", "VACUUM_REGION"),
        "PORT::PORT_SHEET__P1__0000",
        40.0,
        (115.785, 115.356, 928.858),
        50.0,
    ),
}
PORT_CASES["C"] = PORT_CASES["B"]


@pytest.fixture(params=sorted(PORT_CASES))
def banded_port_model(request, tmp_path):
    """Every surface and the SA interface banded (0.5, 1.0 um) in a model with a lumped port, on each route."""
    route = request.param
    make_input, sa_pair, *rest = PORT_CASES[route]
    effective = make_input(tmp_path) if route != "A" else make_input()
    insets = all_interface_insets(effective, route, sa_pair)
    model = build_conformal_model(
        effective.build_input, effective.representation, route, insets=insets
    )
    return route, effective, insets, model, rest


def corner_count(model: ConformalModel, face) -> int:
    """Corners (direction changes) of a face's outer loop; edges split where a neighbour's seam ends are no corner."""
    points = []
    for edge_id, sign in face.loops[0]:
        edge = model.topology.edges[edge_id]
        vertex = model.topology.vertices[edge.v0 if sign > 0 else edge.v1]
        points.append((vertex.x, vertex.y))
    count = len(points)
    return sum(
        (points[i][0] - points[i - 1][0]) * (points[(i + 1) % count][1] - points[i][1])
        != (points[i][1] - points[i - 1][1])
        * (points[(i + 1) % count][0] - points[i][0])
        for i in range(count)
    )


def test_inset_seams_stop_at_the_port_so_it_is_one_rectangle(banded_port_model):
    """The port group is one rectangular face whose area is the footprint; the SA bands keep their areas."""
    route, effective, insets, model, (group, area, sa_areas, y_mid) = banded_port_model
    assert not model.report.failures
    (face_id,) = model.routed.groups[group].ids
    face = model.topology.faces[face_id]
    assert len(face.loops) == 1 and corner_count(model, face) == 4
    assert (
        face.twice_xy_area * model.labelled.catalogue.grid.size_um** 2 / 2
        == pytest.approx(area)
    )
    bands = band_areas_without_sheets(model, "band_sa")
    assert [bands[k] for k in sorted(bands)] == pytest.approx(list(sa_areas), abs=1e-3)
    assert_mirror_symmetric_about(model, "band_sa", y_mid)


def test_the_files_of_a_banded_port_load_and_mesh(banded_port_model, tmp_path):
    """The written model verifies against its sidecar, keeps one port face and meshes."""
    route, effective, insets, _, (group, area, sa_areas, _) = banded_port_model
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        insets=insets,
        output_dir=tmp_path / "out",
    )
    sidecar = read_sidecar(artifacts.sidecar)
    assert len(views.group_members(sidecar)[group]) == 1
    assert sidecar["group"][group]["area_um2"] == pytest.approx(area)
    sa = sidecar["group"]
    found = [
        sa[n]["area_um2"]
        for n in sorted(sa)
        if sa[n]["interface_type"] == "SA" and sa[n]["band"] is not None
    ]
    assert found == pytest.approx(list(sa_areas), abs=1e-3)
    report = verify_xao(artifacts.xao, sidecar)
    assert report.ok, report.problems
    with gmsh_session(artifacts.xao):
        apply_sidecar_embeds(sidecar)
        mesh_coarsely(60.0)


def test_a_refinement_curve_across_the_port_is_a_user_cut_and_stays_in_the_port():
    """Explicit user curves are never clipped: the port holds the pieces of a curve crossing it."""
    effective = thin_film_pair()
    inset = InsetRequest("band_sa", SA, MARGINS)
    across = RefinementRequest("across", 0.0, ((5.0, 0.5), (5.0, 3.5)))
    model = build_conformal_model(
        effective.build_input,
        effective.representation,
        "A",
        insets=(inset,),
        refinements=(across,),
    )
    (face_id,) = model.routed.groups["PORT::port_sheet"].ids
    port = model.topology.faces[face_id]
    assert (
        port.refinement_edges
    )  # the curve is embedded in the port: the seams' clip never touches it
    assert corner_count(model, port) == 4
    assert not model.report.failures
