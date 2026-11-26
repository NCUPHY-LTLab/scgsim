"""The frequencies of a Driven sweep: the ones Palace solves at, how close two may be, and where full solves go.

``palace_frequencies_ghz`` merges the ranges' ``Samples`` entries as Palace 0.18.1's ``DrivenSolverData`` does, and
``DrivenSweep`` refuses two frequencies a Palace build may merge or not.  The rounding facts are from stock Palace
0.18.1 dry runs (issue 307): its build fuses the multiply and the add of ``ConstructLinearRange``.  No Palace runs
here.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from scgsim.palace import Linear, Log, Points
from scgsim.palace.config.driven_sweep import DrivenSweep, palace_frequencies_ghz

# The band 10 MHz apart and a window 10 kHz apart around 7.32 GHz: the example of the S-parameter guide.
BAND_AND_WINDOW = [Linear(4e9, 9e9, points=501), Linear(7.30e9, 7.34e9, points=4001)]


# ---- The frequencies Palace solves at ----------------------------------------------------------------------


def test_a_linear_range_is_evenly_spaced_with_both_ends():
    """``NSample`` frequencies from ``MinFreq`` to ``MaxFreq``, as Palace's ``ConstructLinearRange`` spaces them."""
    entry = Linear(2e9, 10e9, points=5).samples_entry()
    assert palace_frequencies_ghz([entry]) == (2.0, 4.0, 6.0, 8.0, 10.0)


def test_a_log_range_is_evenly_spaced_on_a_log_scale():
    """1 to 10 GHz in 5 points is 10^(k/4) GHz: 1, 10^(1/4), sqrt(10), 10^(3/4) and 10 GHz."""
    entry = Log(1e9, 1e10, points=5).samples_entry()
    assert palace_frequencies_ghz([entry]) == pytest.approx(
        (1.0, 1.77827941003892, 3.16227766016838, 5.62341325190349, 10.0), rel=1e-14
    )
    # A range of two decades in 3 points has the decade between.
    entry = Log(1e8, 1e10, points=3).samples_entry()
    assert palace_frequencies_ghz([entry]) == pytest.approx((0.1, 1.0, 10.0), rel=1e-15)


def test_the_ranges_merge_sorted_and_each_shared_frequency_once():
    """The band's 501 and the window's 4,001 less the five they share: 7.30, 7.31, 7.32, 7.33 and 7.34 GHz.

    Palace sorts every range's frequencies together, whatever the order the ranges are given in.
    """
    band, window = BAND_AND_WINDOW
    merged = palace_frequencies_ghz([window.samples_entry(), band.samples_entry()])
    assert len(merged) == 501 + 4001 - 5
    assert (merged[0], merged[-1]) == (4.0, 9.0)
    assert all(high > low for low, high in pairwise(merged))
    # 7.29 GHz is the band's last below the window, 10 MHz away; the window then steps 10 kHz.
    around = [f for f in merged if 7.28 < f < 7.36]
    assert len(around) == 4001 + 2
    assert around[:3] == pytest.approx([7.29, 7.30, 7.30001], rel=1e-15)
    assert around[-2:] == pytest.approx([7.34, 7.35], rel=1e-15)


# The hanger resonator's Eigenmode mode, from which its study placed a window and a full solve.
HANGER_F0_HZ = 7356647803.532
HANGER_Q = 13947.620963028376


def hanger_sweep():
    """The hanger study's sweep: the 4-9 GHz band, +-20 linewidths of 2,001 around f0, and a full solve at f0."""
    linewidth_hz = HANGER_F0_HZ / HANGER_Q
    window = Linear(
        HANGER_F0_HZ - 20.0 * linewidth_hz,
        HANGER_F0_HZ + 20.0 * linewidth_hz,
        points=2001,
    )
    return {
        "ranges": [
            Linear(4e9, 9e9, points=501),
            window,
            Points([HANGER_F0_HZ], full_solve=True),
        ],
        "adaptive_tolerance": 1e-3,
    }


@pytest.mark.parametrize(
    ("sweep", "frequencies"),
    [
        pytest.param(
            {"ranges": [Linear(4e9, 6e9, points=201), Linear(6e9, 9e9, points=301)]},
            501,
            id="a shared end",
        ),
        pytest.param(hanger_sweep(), 2502, id="the hanger study's"),
        pytest.param(
            {"ranges": [Points([7.3e9, 7.3e9 + 10.5])]},
            2,
            id="one printed unit and more",
        ),
        pytest.param(
            {"ranges": [Points([0.5e9, 0.5e9 + 1.5])]}, 2, id="1.5 Hz apart below 1 GHz"
        ),
    ],
)
def test_two_frequencies_one_to_rounding_or_apart_in_every_palace_build_are_accepted(
    sweep, frequencies
):
    """Neighbours are one frequency to rounding, or farther apart than Palace's 1 Hz merge and one printed unit.

    A frequency two ranges share is the same to rounding, so every build merges it: the end both bands give,
    and the hanger study's resonance, the centre of its window and its full solve.  Apart, 10.5 Hz is more than
    the 10 Hz Palace prints between 1 and 10 GHz, and 1.5 Hz more than both the 1 Hz it prints and the 1 Hz it
    merges below 1 GHz.
    """
    assert DrivenSweep(**sweep).frequency_count == frequencies


@pytest.mark.parametrize(
    ("ranges", "match"),
    [
        pytest.param(
            [Linear(4e9, 9e9, points=501), Points([4_439_999_999.0], full_solve=True)],
            r"4\.439999999 GHz and 4\.44 GHz are 1 Hz apart, and two frequencies must be the "
            r"same or more than 10 Hz apart there",
            id="1 Hz from a band's frequency, a full solve",
        ),
        pytest.param(
            [Linear(4e9, 9e9, points=501), Points([4_439_999_999.0])],
            r"4\.439999999 GHz and 4\.44 GHz are 1 Hz apart",
            id="1 Hz from a band's frequency",
        ),
        pytest.param(
            [Linear(4e9, 9e9, points=501), Points([7.3e9, 7.3e9 + 0.5])],
            r"7\.3 GHz and 7\.3000000005 GHz are 0\.5 Hz apart",
            id="0.5 Hz apart",
        ),
        pytest.param(
            [Linear(4e9, 9e9, points=501), Points([0.05e9, 0.05e9 + 0.5])],
            r"0\.05 GHz and 0\.0500000005 GHz are 0\.5 Hz apart, and two frequencies must be the same "
            r"or more than 1 Hz apart there",
            id="0.5 Hz apart at 50 MHz, printed apart",
        ),
        pytest.param(
            [Points([1e12 + k * 4 * math.ulp(1e12) for k in range(3001)])],
            r"1000 GHz and 1000 GHz are 0\.00102 Hz apart, and two frequencies must be the same or more "
            r"than 10000 Hz apart there",
            id="a chain of neighbours within rounding, 1.46 Hz long at 1 THz",
        ),
    ],
)
def test_two_frequencies_a_palace_build_may_merge_or_not_are_refused(ranges, match):
    """Closer than Palace's merge distance and rounding, but not the same, a build may merge them or not.

    The band's 4.44 GHz is 4.4399999999999995 in unfused arithmetic and 4.44 in the stock build's fused one:
    0.9999992 Hz and 1.0000001 Hz above 4.439999999 GHz, so SCGSim would merge the two and that build keep both,
    and the run would return a row more than SCGSim reads back.  At 50 MHz Palace prints a frequency to 0.1 Hz,
    so its 1 Hz merge sets the spacing, uniform or adaptive alike.  Palace measures each frequency from the last
    one it kept, so a chain whose neighbours are each within rounding is refused once it leaves the rounding of
    the first: 3,001 points 4 units in the last place apart at 1 THz span 1.46 Hz, and Palace would keep two of
    them that print the same.
    """
    for adaptive_tolerance in (None, 1e-3):
        if adaptive_tolerance is None and any(item.full_solve for item in ranges):
            continue
        with pytest.raises(ValueError, match=match):
            DrivenSweep(ranges=ranges, adaptive_tolerance=adaptive_tolerance)


def test_a_full_solve_on_another_range_s_frequency_is_that_frequency_once():
    """A full solve on a frequency of the band is that frequency once, and one more solve for the cap.

    Within rounding Palace's own check decides: unfused arithmetic puts the band's 4.44 GHz at
    4.4399999999999995, and the stock build, which fuses it, at 4.44 and accepts a full solve there; it loses one
    at 4.73 GHz, where the two arithmetics swap, and stops as it starts (stock Palace 0.18.1 dry runs).  The lowest
    and the highest frequency are solved in full anyway, so a full solve there adds none.
    """
    band = Linear(4e9, 9e9, points=501)
    grid = palace_frequencies_ghz([band.samples_entry()])
    assert grid[45] == 4.45 and grid[44] == 4.4399999999999995

    def sweep(full_solve_hz, **options):
        ranges = [band, Points([full_solve_hz], full_solve=True)]
        return DrivenSweep(ranges=ranges, adaptive_tolerance=1e-3, **options)

    for on_the_band in (4.45e9, 4.44e9, 4.73e9):
        assert sweep(on_the_band).frequency_count == 501
    with pytest.raises(ValueError, match="adaptive_max_samples must be at least 3"):
        sweep(4.45e9, adaptive_max_samples=2)
    assert sweep(4e9, adaptive_max_samples=2).frequency_count == 501
    assert sweep(9e9, adaptive_max_samples=2).frequency_count == 501


def test_frequencies_within_1_hz_of_the_last_kept_are_one():
    """Palace's ``std::unique`` keeps a frequency 1 Hz or more above the last one kept, not the one before.

    0.6 Hz above 7.3 GHz goes; 1.2 Hz above stays, though it is only 0.6 Hz above the one that went.
    """
    entries = [{"Type": "Point", "Freq": [7.3 + 1.2e-9, 7.3, 2.0, 7.3 + 0.6e-9]}]
    assert palace_frequencies_ghz(entries) == (2.0, 7.3, 7.3 + 1.2e-9)
