"""Different-Net contact is an error in every route (brief section 14); same-Net contact is normal."""

from __future__ import annotations

import pytest

from sgbproto.errors import ContactError
from sgbproto.pipeline import build
from sgbproto.records import Footprint, Grid, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import _chip_prisms, rect_um, to_grid


def model(route: str, rects: list[tuple[str, str, tuple], ...], grid: Grid = Grid(), z=None) -> ModelSpec:
    """Chip with conductor rectangles ``(entity, net, (x0, y0, x1, y1))`` (um) as sheets (A) or 0.2 um metal (B)."""
    z0, z1 = (z_value(0), z_value(0)) if route == "A" else (z_value(0), z_value("0.2"))
    prisms = _chip_prisms(grid, 40.0)
    for entity, net, box in rects:
        fp = Footprint.from_rings(to_grid(grid, rect_um(*box)))
        prisms.append(PrismRecord(entity, fp, z0, z1, "Al", "conductor", net_id=net))
    return ModelSpec(grid, tuple(prisms), name="contact", meta={"route": route})


@pytest.mark.parametrize("route", ["A", "B"])
def test_two_nets_sharing_an_edge_fail_naming_both_entities_and_the_location(route):
    spec = model(route, [("padL", "N1", (0, 0, 10, 10)), ("padR", "N2", (10, 0, 20, 10))])
    with pytest.raises(ContactError) as err:
        build(spec)
    text = str(err.value)
    assert "padL" in text and "padR" in text and "N1" in text and "N2" in text
    assert "10.0000" in text  # x of the shared edge's end points


@pytest.mark.parametrize("route", ["A", "B"])
def test_two_nets_sharing_only_a_vertex_fail(route):
    spec = model(route, [("padL", "N1", (0, 0, 10, 10)), ("padR", "N2", (10, 10, 20, 20))])
    with pytest.raises(ContactError, match=r"padL.*padR|padR.*padL"):
        build(spec)
    with pytest.raises(ContactError, match=r"\(10\.0000, 10\.0000\)"):
        build(spec)


def test_a_source_gap_closed_by_a_coarse_grid_fails_but_the_fine_grid_passes():
    rects = [("padL", "N1", (0, 0, 10, 10)), ("padR", "N2", (10.004, 0, 20, 10))]  # a 4 nm gap
    assert build(model("A", rects, Grid(0.001))).report is not None  # 1 nm grid keeps the gap
    with pytest.raises(ContactError, match=r"grid g = 0.01"):
        build(model("A", rects, Grid(0.01)))  # 10 nm grid: both edges snap to x = 10


def test_a_post_of_another_net_standing_on_metal_is_a_contact():
    grid = Grid()
    spec = model("B", [("ground", "G", (-30, -30, 30, 30))])
    post = PrismRecord("post", Footprint.from_rings(to_grid(grid, rect_um(-5, -5, 5, 5))), z_value("0.2"), z_value(3),
                       "Al", "conductor", net_id="X")
    with pytest.raises(ContactError, match=r"ground.*post|post.*ground"):
        build(ModelSpec(grid, spec.prisms + (post,), name="stack", meta={"route": "B"}))


@pytest.mark.parametrize("route", ["A", "B"])
def test_same_net_contact_passes(route):
    spec = model(route, [("padL", "N1", (0, 0, 10, 10)), ("padR", "N1", (10, 0, 20, 10))])  # shared edge: one cell
    result = build(spec)
    assert not result.report.failures


def test_same_net_vertex_contact_passes_for_sheets():
    """Two Route-A sheets of one Net meeting at a corner are two faces sharing a vertex: legal."""
    result = build(model("A", [("padL", "N1", (0, 0, 10, 10)), ("padR", "N1", (10, 10, 20, 20))]))
    assert not result.report.failures


def test_an_unknown_edge_policy_is_rejected():
    from sgbproto.records import CurveFamilyRecord

    with pytest.raises(ValueError, match="unknown edge policy"):
        CurveFamilyRecord("f", "seam", (), (z_value(0),), "dissolve")  # tile seams are not a family


def test_a_same_net_vertex_contact_of_finite_metal_is_a_dedicated_nonmanifold_error():
    from sgbproto.errors import NonManifoldError

    spec = model("B", [("padL", "N1", (0, 0, 10, 10)), ("padR", "N1", (10, 10, 20, 20))])
    with pytest.raises(NonManifoldError) as err:
        build(spec)
    text = str(err.value)
    assert "N1" in text and "padL" in text and "padR" in text and "(10.0000, 10.0000)" in text and "vertical edge" in text


def test_a_hole_touching_its_outer_boundary_is_a_nonmanifold_pinch_in_route_b_but_fine_for_a_sheet():
    from sgbproto.errors import NonManifoldError

    grid = Grid()

    def spec(route):
        z0, z1 = (z_value(0), z_value(0)) if route == "A" else (z_value(0), z_value("0.2"))
        fp = Footprint.from_rings(to_grid(grid, rect_um(-10, -10, 10, 10)), [to_grid(grid, [(10.0, 0.0), (5.0, -5.0), (0.0, 0.0), (5.0, 5.0)])])
        prisms = _chip_prisms(grid, 40.0) + [PrismRecord("ring", fp, z0, z1, "Al", "conductor", net_id="N")]
        return ModelSpec(grid, tuple(prisms), name="ring", meta={"route": route})

    with pytest.raises(NonManifoldError, match=r"ring.*\(10\.0000, 0\.0000\)"):
        build(spec("B"))
    assert not build(spec("A")).report.failures
