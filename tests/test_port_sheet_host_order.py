"""A port sheet's hosts in the order its authored direction meets them: start host, then end host (issue #282)."""

from __future__ import annotations

import pytest

from scgsim.geometry import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.models.regions import hosts_along_direction


def rectangle(x0, x1, y0=0.0, y1=2.0):
    """One counter-clockwise axis-aligned ring."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def sheet(direction, landings):
    """A sheet from x = 0 to 10 um; ``landings`` are (host, x0, x1) overlaps, in the order they were found."""
    overlaps = tuple(
        PortSheetOverlapRecord(
            f"overlap_{k}", "sheet", "sheet", host, f"{host}_polygon", rectangle(x0, x1)
        )
        for k, (host, x0, x1) in enumerate(landings)
    )
    return PortSheetRegionRecord(
        "sheet",
        "port",
        "sheet",
        rectangle(0, 10),
        overlaps=overlaps,
        metadata={"direction": direction},
    )


GROUND = ("ground", 0, 1)
PAD = ("pad", 9, 10)


@pytest.mark.parametrize("found", [(PAD, GROUND), (GROUND, PAD)])
@pytest.mark.parametrize(
    ("direction", "expected"),
    [((1.0, 0.0, 0.0), ("ground", "pad")), ((-1.0, 0.0, 0.0), ("pad", "ground"))],
)
def test_the_start_host_comes_first_whatever_order_the_overlaps_were_found_in(
    found, direction, expected
):
    """The ground lands at x = 0..1 and the pad at 9..10: +x starts at the ground, -x at the pad."""
    assert hosts_along_direction(sheet(direction, found)) == expected


def test_a_host_landing_twice_is_one_host_at_the_mean_of_its_landings():
    """A pad found twice, at 3..4 and 9..10 (mean 6.5), lies after a ground at 5..6 along +x."""
    region = sheet((1.0, 0.0, 0.0), [("pad", 3, 4), ("ground", 5, 6), ("pad", 9, 10)])
    assert hosts_along_direction(region) == ("ground", "pad")
