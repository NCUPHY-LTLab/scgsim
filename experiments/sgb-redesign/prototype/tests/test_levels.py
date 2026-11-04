"""Levels, slabs and active sets (brief section 2)."""

from __future__ import annotations

from fractions import Fraction

from sgbproto.levels import build_levels
from sgbproto.scenes import airbridge, pad_in_ground


def test_route_b_levels_include_the_port_mid_plane_and_sheets_sit_on_levels():
    spec = pad_in_ground("B")
    lv = build_levels(spec.prisms)
    assert lv.levels == (Fraction(-100), Fraction(0), Fraction(1, 10), Fraction(1, 5), Fraction(100))
    names = {k: spec.prisms[k].entity_id for k in range(len(spec.prisms))}
    port_level = lv.levels.index(Fraction(1, 10))
    assert [names[k] for k in lv.level_sheets[port_level]] == ["p1"]
    # ground is active in both slabs of the metal layer, vacuum in all three above the substrate
    assert {names[k] for k in lv.slab_active[1]} == {"vacuum", "ground", "pad"}
    assert {names[k] for k in lv.slab_active[2]} == {"vacuum", "ground", "pad"}
    assert {names[k] for k in lv.slab_active[0]} == {"substrate"}


def test_route_a_sheets_are_never_active_in_a_slab():
    spec = pad_in_ground("A")
    lv = build_levels(spec.prisms)
    assert lv.levels == (Fraction(-100), Fraction(0), Fraction(100))
    assert all(not spec.prisms[k].is_sheet for slab in lv.slab_active for k in slab)
    assert len(lv.level_sheets[1]) == 3  # ground, pad, port


def test_airbridge_has_five_slabs():
    lv = build_levels(airbridge().prisms)
    assert lv.levels == tuple(Fraction(x) for x in ("-100", "0", "0.2", "3", "3.3", "100"))
