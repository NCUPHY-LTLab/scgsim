"""Keep live and saved junction terminal roles independent of polygon ring order."""

from fractions import Fraction

import pytest

from scgsim.aedt.runtime.native.junction_readback import JunctionReadback
from scgsim.aedt.runtime.native.junction_topology import native_junction_corners
from scgsim.aedt.specs.epr import PlanarJunction

CANONICAL = ((100.0, -1.0), (120.0, -1.0), (120.0, 1.0), (100.0, 1.0))
RINGS = [
    ring[index:] + ring[:index]
    for ring in (CANONICAL, CANONICAL[::-1])
    for index in range(4)
]


def _rational(point):
    return tuple(Fraction(str(float(value))) for value in point)


def _junction(direction=(1.0, 0.0)):
    return PlanarJunction("jj", "port", "pad", "ground", direction, 2.0, 1e-8, 0.0)


@pytest.mark.parametrize("ring", RINGS)
@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (-1, 0), (0, -1), (0.6, 0.8)])
def test_source_frame_preserves_actual_native_coordinates(ring, direction):
    """Order each legal ring in its axial frame without replacing readback values."""
    dx, dy = direction

    def transform(point):
        x, y = point
        return (x * dx - y * dy, x * dy + y * dx)

    polygon = {"exterior": [transform(point) for point in ring], "holes": []}
    offset = (Fraction(1, 10_000_000), Fraction(-1, 10_000_000))
    matches = {
        _rational(point): tuple(
            a + b for a, b in zip(_rational(point), offset, strict=True)
        )
        for point in polygon["exterior"]
    }
    expected = [matches[_rational(transform(point))] for point in CANONICAL]
    actual = native_junction_corners(polygon, direction, matches)
    assert actual == expected
    assert all(a is b for a, b in zip(actual, expected, strict=True))


def _live_readback(ring, *, partial_pad=False):
    """Supply exact footprints to the existing live contact-verification stages."""
    readback = JunctionReadback(None, {}, _junction())
    readback.expected = {"exterior": ring, "holes": []}
    readback.central_matches = {_rational(point): _rational(point) for point in ring}
    pad = (
        (90, -1),
        (100, -1),
        (100, 0 if partial_pad else 1),
        (90, 0 if partial_pad else 1),
    )
    ground = ((120, -1), (130, -1), (130, 1), (120, 1))
    readback.native_by_net = {
        "pad": [(tuple(map(_rational, pad)),)],
        "ground": [(tuple(map(_rational, ground)),)],
    }
    readback.potential_by_net = dict(readback.native_by_net)
    return readback


@pytest.mark.parametrize("ring", RINGS)
def test_live_terminal_and_open_side_roles_follow_direction(ring):
    """Accept full end contact while keeping both long sides open for every ring."""
    readback = _live_readback(ring)
    readback._verify_central_interior()  # noqa: SLF001 -- exercise the native contact stages without starting an SDK application.
    readback._verify_edge_contacts()  # noqa: SLF001 -- exercise the native contact stages without starting an SDK application.
    assert readback.edge_evidence == [
        {"start_um": [100.0, 1.0], "end_um": [100.0, -1.0], "terminal_net": "pad"},
        {"start_um": [120.0, -1.0], "end_um": [120.0, 1.0], "terminal_net": "ground"},
        {"start_um": [100.0, -1.0], "end_um": [120.0, -1.0], "terminal_net": None},
        {"start_um": [120.0, 1.0], "end_um": [100.0, 1.0], "terminal_net": None},
    ]


@pytest.mark.parametrize("ring", RINGS)
def test_live_missing_terminal_contact_still_fails(ring):
    """Reject a pad covering only half its designated end edge after reordering."""
    readback = _live_readback(ring, partial_pad=True)
    readback._verify_central_interior()  # noqa: SLF001 -- isolate the live contact guard from SDK object construction.
    with pytest.raises(
        RuntimeError, match="designated junction end edge is not wholly covered"
    ):
        readback._verify_edge_contacts()  # noqa: SLF001 -- the existing strict missing-contact guard is the tested contract.


@pytest.mark.parametrize("ring", RINGS)
def test_live_wrong_net_corner_contact_still_fails(ring):
    """Reject a foreign net touching the pad corner of an otherwise valid sheet."""
    readback = _live_readback(ring)
    foreign = ((99, -2), (100, -2), (100, -1), (99, -1))
    readback.potential_by_net["foreign"] = [(tuple(map(_rational, foreign)),)]
    readback._verify_central_interior()  # noqa: SLF001 -- isolate the live corner guard from SDK object construction.
    with pytest.raises(RuntimeError, match="forbidden side/wrong-net corner contact"):
        readback._verify_edge_contacts()  # noqa: SLF001 -- preserve checks against every potentially contacting net.
