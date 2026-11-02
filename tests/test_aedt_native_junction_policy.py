"""Check the absolute micrometre policy used by native junction readback."""

import math

import pytest

from scgsim.aedt.runtime.native.junction_topology import (
    NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
    native_coordinates_close,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((0.0,), (1e-5,), True),
        ((0.0,), (-1e-5,), True),
        ((0.0,), (math.nextafter(1e-5, math.inf),), False),
        ((0.0,), (math.nextafter(-1e-5, -math.inf),), False),
        ((1e6,), (1e6 + 1e-4,), False),
        ((0.0, 0.0), (1e-5, -1e-5), True),
        ((0.0,), (0.0, 0.0), False),
        ((math.nan,), (0.0,), False),
        ((math.inf,), (math.inf,), False),
        ((0.0,), (-math.inf,), False),
    ],
)
def test_native_coordinate_policy_is_absolute(left, right, expected):
    """Retain inclusive 1e-5 um bounds without relative scaling or nonfinite matches."""
    assert NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM == 1e-5
    assert native_coordinates_close(left, right) is expected
