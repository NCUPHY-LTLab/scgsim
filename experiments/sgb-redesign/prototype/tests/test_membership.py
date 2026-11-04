"""Area-weighted membership survives dissolution (brief section 14), on the "tab" model."""

from __future__ import annotations

import pytest

from sgbproto.pipeline import build
from sgbproto.records import Footprint, Grid, ModelSpec, PrismRecord, z_value
from sgbproto.scenes import _chip_prisms, rect_um, to_grid


def tab_model(route: str) -> ModelSpec:
    """A 20x20 um pad with an overlapping same-Net 10x6 um tab (it sticks out 5 um on the right)."""
    grid = Grid()
    z0, z1 = (z_value(0), z_value(0)) if route == "A" else (z_value(0), z_value("0.2"))
    prisms = _chip_prisms(grid, 40.0)
    for name, box in (("pad", (-10, -10, 10, 10)), ("pad_tab", (5, -3, 15, 3))):
        prisms.append(PrismRecord(name, Footprint.from_rings(to_grid(grid, rect_um(*box))), z0, z1, "Al", "conductor", net_id="P"))
    return ModelSpec(grid, tuple(prisms), name=f"tab_{route}", meta={"route": route})


@pytest.mark.parametrize("route", ["A", "B"])
def test_the_merged_top_face_keeps_the_overlapping_tab_in_its_membership(route):
    topo = build(tab_model(route)).topology
    union = 400 + 60 - 30  # pad + tab - overlap, um^2
    top = [f for f in topo.faces.values() if f.kind == "horizontal" and "pad" in f.membership
           and (f.group.startswith(("PEC::P", "MA::P")) if route == "B" else f.group == "PEC::P")]
    assert len(top) == 1  # the overlap dissolved: one face
    share = top[0].membership
    assert share["pad"] == pytest.approx(400 / union, abs=1e-9)
    assert share["pad_tab"] == pytest.approx(60 / union, abs=1e-9)
    assert share["vacuum"] == pytest.approx(1.0) and sum(share.values()) > 1.0  # non-exclusive


def test_slab_cells_carry_the_union_of_their_raw_cells():
    result = build(tab_model("B"))
    slab = next(s for s in result.slabs if s.plane == "slab[0,1/5]")
    cell = max(range(1, len(slab.arr.faces)), key=lambda fi: slab.arr.face_twice_area(fi) if slab.labels[fi] and slab.labels[fi].kind == "conductor" else -1)
    assert slab.membership[cell]["pad_tab"] == pytest.approx(60 / 430, abs=1e-9)
    assert slab.membership[cell]["pad"] == pytest.approx(400 / 430, abs=1e-9)
