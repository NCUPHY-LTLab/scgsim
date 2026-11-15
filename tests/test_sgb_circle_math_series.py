"""The decimal circle arithmetic stops its series early without changing a digit, and ignores the caller's context.

``circle_math`` sums its Taylor series only until no later term can change the sum.  These tests compare
the results with the full 80-term sums on a deterministic set of angles and ratios that includes the
reduction boundaries, and check that a build's witness points and identity digest do not depend on
``decimal.getcontext()``.
"""

from __future__ import annotations

import decimal
from decimal import Context, Decimal
from fractions import Fraction

import sgb_supported_cells as disk_qubit
from sgb_supported_kit import catalogue, type_bands

from scgsim.geometry.exact import circle_math
from scgsim.geometry.exact.carriers import (  # private-import: each build must compute its witnesses afresh, under its own context
    _witness,
)
from scgsim.geometry.sgb.model import build_model_from_catalogue

FULL_TERMS = 80
CONTEXT = Context(prec=circle_math.PRECISION)


def full_sin_cos(angle: Decimal) -> tuple[Decimal, Decimal]:
    """Sine and cosine with every one of the 80 terms summed (the series before it stopped early)."""
    reduced = CONTEXT.remainder_near(angle, circle_math.TWO_PI)
    square = CONTEXT.multiply(reduced, reduced)
    sin_term, cos_term = reduced, circle_math.ONE
    sin_sum, cos_sum = reduced, circle_math.ONE
    for n in range(1, FULL_TERMS):
        cos_term = CONTEXT.minus(
            CONTEXT.divide(
                CONTEXT.multiply(cos_term, square), Decimal((2 * n - 1) * (2 * n))
            )
        )
        sin_term = CONTEXT.minus(
            CONTEXT.divide(
                CONTEXT.multiply(sin_term, square), Decimal((2 * n) * (2 * n + 1))
            )
        )
        cos_sum = CONTEXT.add(cos_sum, cos_term)
        sin_sum = CONTEXT.add(sin_sum, sin_term)
    return CONTEXT.plus(sin_sum), CONTEXT.plus(cos_sum)


def full_atan(value: Decimal) -> Decimal:
    """Arctangent with every one of the 80 terms summed, after the same three halvings."""
    for _ in range(3):
        value = CONTEXT.divide(
            value,
            CONTEXT.add(
                circle_math.ONE,
                CONTEXT.sqrt(
                    CONTEXT.add(circle_math.ONE, CONTEXT.multiply(value, value))
                ),
            ),
        )
    square = CONTEXT.multiply(value, value)
    term, total = value, value
    for n in range(1, FULL_TERMS):
        term = CONTEXT.minus(CONTEXT.multiply(term, square))
        total = CONTEXT.add(total, CONTEXT.divide(term, Decimal(2 * n + 1)))
    return CONTEXT.multiply(total, Decimal(8))


def angles() -> list[Decimal]:
    """Angles across seven turns, the reduction boundaries at odd multiples of pi, and tiny angles."""
    pi = circle_math.PI
    grid = [circle_math.to_decimal(Fraction(k, 97)) for k in range(-700, 701)]
    boundaries = [
        CONTEXT.add(CONTEXT.multiply(pi, Decimal(k)), offset)
        for k in range(-7, 8)
        for offset in (
            Decimal(0),
            Decimal("1e-40"),
            Decimal("-1e-40"),
            Decimal("1e-12"),
        )
    ]
    quarters = [CONTEXT.multiply(circle_math.HALF_PI, Decimal(k)) for k in range(-8, 9)]
    tiny = [
        Decimal(f"{sign}1e-{e}")
        for sign in "+-"
        for e in (5, 10, 20, 30, 31, 59, 60, 61, 90)
    ]
    return [*grid, *boundaries, *quarters, *tiny, Decimal(0)]


def ratios() -> list[Decimal]:
    """Ratios y / x of every size, both signs, around 1 where the halvings change most, and zero."""
    grid = [circle_math.to_decimal(Fraction(k, 37)) for k in range(-500, 501)]
    near_one = [
        CONTEXT.add(Decimal(1), Decimal(f"{d}e-{e}"))
        for d in (-1, 1)
        for e in (3, 20, 50)
    ]
    sizes = [
        Decimal(f"{sign}1e{e}") for sign in "+-" for e in (-61, -30, -8, 8, 30, 61)
    ]
    return [*grid, *near_one, *sizes, *(CONTEXT.minus(r) for r in near_one), Decimal(0)]


def test_the_early_stopped_sine_and_cosine_equal_the_full_series():
    """Every angle, boundaries and tiny angles included, gives the same digits as all 80 terms."""
    for angle in angles():
        assert circle_math.sin_cos(angle) == full_sin_cos(angle), angle


def test_the_early_stopped_arctangent_equals_the_full_series():
    """Every ratio gives the same digits as all 80 terms (``atan2(r, 1)`` is the series of ``r``)."""
    for ratio in ratios():
        assert circle_math.atan2(ratio, circle_math.ONE) == full_atan(ratio), ratio


def witnesses_and_digest(precision: int) -> tuple[list, str]:
    """The witness points of every arc and the identity digest of the D1 die, built under ``precision``."""
    _witness.cache_clear()
    with decimal.localcontext() as context:
        context.prec = precision
        model = build_model_from_catalogue(
            catalogue(disk_qubit.SCENES["d1_die_defaults"]()), "B", insets=type_bands()
        )
        witnesses = sorted(
            (arc.id, arc.witness_points()) for arc in model.labelled.arcs.values()
        )
    return witnesses, model.identity_digest()


def test_a_banded_arc_scene_is_the_same_under_every_caller_context():
    """The D1 die (arcs, cut pieces and inset corners) gives the same witnesses and digest at 8, 28 and 60 digits."""
    default = witnesses_and_digest(decimal.getcontext().prec)
    assert default[0]  # the scene has arcs, so the comparison is not vacuous
    assert witnesses_and_digest(8) == default
    assert witnesses_and_digest(60) == default
