"""AEDT live read-back accepts a junction end piece that overlaps its own arm, as the shared partition does.

A landing end piece (the drawn sheet from its end to the cut, used where SGB rebuilds the arm
with an exact arc) overlaps its arm instead of sharing an edge with it.  These cases run the
read-back's end-contact stage on supplied native footprints; no AEDT application starts.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from scgsim.aedt.runtime.native.junction_readback import JunctionReadback
from scgsim.aedt.runtime.native.junction_topology import native_boundary_contact
from scgsim.aedt.specs.epr import PlanarJunction

# A round pad drawn as a polygon, near the sheet's end: its apex at (100, 0), chords to y = +-2.
PAD_ARM = ((99.9, -2.0), (100.0, 0.0), (99.9, 2.0), (90.0, 2.0), (90.0, -2.0))
# The sheet from its start, inside the pad, to the cut: it overlaps the pad between the chords.
LANDING_END = ((99.95, -1.0), (100.005, -1.0), (100.005, 1.0), (99.95, 1.0))
# A straight pad edge at x = 100 and the free end piece beyond it, sharing that edge.
STRAIGHT_ARM = ((90.0, -1.0), (100.0, -1.0), (100.0, 1.0), (90.0, 1.0))
STRAIGHT_END = ((100.0, -1.0), (100.005, -1.0), (100.005, 1.0), (100.0, 1.0))
# Ends that meet the round pad only at its apex: exactly, or across 1 nm^2-scale CAD round-off.
POINT_END = ((100.0, -1.0), (100.005, -1.0), (100.005, 1.0), (100.0, 1.0))
ROUND_OFF_END = (
    (99.999999, -1.0),
    (100.005, -1.0),
    (100.005, 1.0),
    (99.999999, 1.0),
)


def _footprint(ring):
    return [(tuple(tuple(Fraction(str(v)) for v in point) for point in ring),)]


def _end_readback(arm, end):
    """The read-back state the end-contact stage reads: one arm and one derived end of net pad."""
    readback = JunctionReadback(
        None, {}, PlanarJunction("jj", "sheet", "pad", "ground", (1, 0), 2, 1e-8, 0)
    )
    readback.record = {
        "ends": [{"polygon_id": "end", "source_entity_id": "chip/pad", "net_id": "pad"}]
    }
    readback.source = {"junction_partitions": {"jj": readback.record}}
    readback.native_by_polygon = {
        ("chip/pad", "arm"): _footprint(arm),
        ("chip/pad", "end"): _footprint(end),
    }
    readback.native_z_by_polygon = {
        key: ((0.0, 0.2), True) for key in readback.native_by_polygon
    }
    readback.entity_nets = {"chip/pad": "pad"}
    return readback


@pytest.mark.parametrize(
    ("arm", "end"),
    [(PAD_ARM, LANDING_END), (STRAIGHT_ARM, STRAIGHT_END)],
    ids=["landing-overlaps-its-arm", "straight-shares-an-edge"],
)
def test_an_end_that_overlaps_or_shares_an_edge_with_its_arm_is_accepted(arm, end):
    """A landing end's area overlap and a free end's shared edge both prove contact."""
    # The landing end shares no boundary span with its arm: only its overlap proves contact.
    shares_an_edge = native_boundary_contact(_footprint(end), _footprint(arm))
    assert shares_an_edge == (arm is STRAIGHT_ARM)
    _end_readback(arm, end)._verify_end_contacts()  # noqa: SLF001 -- the stage under test, without an SDK application.


@pytest.mark.parametrize(
    "end", [POINT_END, ROUND_OFF_END], ids=["at-a-point", "below-readback-resolution"]
)
def test_an_end_that_only_touches_its_arm_at_a_point_is_refused(end):
    """Neither a point contact nor an overlap narrower than the native tolerance proves contact."""
    with pytest.raises(
        RuntimeError,
        match="native derived end lacks positive-length or positive-area source-arm contact",
    ):
        _end_readback(PAD_ARM, end)._verify_end_contacts()  # noqa: SLF001 -- the stage under test, without an SDK application.


def test_a_landing_end_touching_another_net_is_still_refused():
    """Contact with another net is refused whatever the contact with the own arm."""
    readback = _end_readback(PAD_ARM, LANDING_END)
    readback.native_by_polygon["chip/ground", "plane"] = _footprint(
        ((100.005, -1.0), (101.0, -1.0), (101.0, 1.0), (100.005, 1.0))
    )
    readback.native_z_by_polygon["chip/ground", "plane"] = ((0.0, 0.2), True)
    readback.entity_nets["chip/ground"] = "ground"
    with pytest.raises(RuntimeError, match="native derived end contacts another net"):
        readback._verify_end_contacts()  # noqa: SLF001 -- the stage under test, without an SDK application.
