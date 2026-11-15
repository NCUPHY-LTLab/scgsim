"""Platform-independent circle arithmetic in fixed-precision decimals.

Owns every trigonometric step that decides *topology*: the witness chords of an arc
(which lattice points the polyline passes through) and the cut points of an arc into
quarter pieces.  Those lattice points feed snap rounding and enter content-addressed
ids, so they must not depend on the C library: ``math.sin`` and friends may differ in the
last bit between platforms and library versions, which flips a rounding tie.

Everything here uses :mod:`decimal` with a private context of ``PRECISION`` digits
(correctly rounded ``+ - * / sqrt``, identical on every platform) and series written out
in plain arithmetic.  The results are accurate to far more digits than the lattice
needs; only their determinism matters.  Every operation names that context, so no
result depends on the caller's ``decimal.getcontext()``; callers that do their own
decimal steps on these values run them under :func:`working_precision`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from decimal import ROUND_FLOOR, Context, Decimal, localcontext
from fractions import Fraction

from scgsim.refusal import internal

PRECISION = 60
_CONTEXT = Context(prec=PRECISION)
ZERO = Decimal(0)
ONE = Decimal(1)
HALF = Decimal("0.5")
PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
)
TWO_PI = _CONTEXT.multiply(PI, 2)
HALF_PI = _CONTEXT.divide(PI, 2)
SERIES_TERMS = 80
"""The most terms a series sums; it stops earlier once no later term can change the sum."""


def working_precision() -> AbstractContextManager[Context]:
    """The module's private context as the current one, for decimal steps written with operators.

    ``a + b`` or ``-a`` round to the *current* context; inside this block that is the same
    ``PRECISION``-digit context every function here uses, whatever the caller has set.
    """
    return localcontext(_CONTEXT)


def to_decimal(value: Fraction | int) -> Decimal:
    """A rational as a decimal at the working precision (correctly rounded)."""
    fraction = Fraction(value)
    return _CONTEXT.divide(Decimal(fraction.numerator), Decimal(fraction.denominator))


def sqrt(value: Decimal) -> Decimal:
    """Correctly rounded square root at the working precision."""
    return _CONTEXT.sqrt(value)


def sin_cos(angle: Decimal) -> tuple[Decimal, Decimal]:
    """Sine and cosine of an angle in radians (Taylor series after reduction to [-pi, pi])."""
    reduced = _CONTEXT.remainder_near(angle, TWO_PI)
    square = _CONTEXT.multiply(reduced, reduced)
    sin_term, cos_term = reduced, ONE
    sin_sum, cos_sum = reduced, ONE
    for n in range(1, SERIES_TERMS):
        cos_term = _CONTEXT.minus(
            _CONTEXT.divide(
                _CONTEXT.multiply(cos_term, square), Decimal((2 * n - 1) * (2 * n))
            )
        )
        sin_term = _CONTEXT.minus(
            _CONTEXT.divide(
                _CONTEXT.multiply(sin_term, square), Decimal((2 * n) * (2 * n + 1))
            )
        )
        # From n^2 > x^2 on, each term is under a quarter of the one before, so once both
        # terms are too small to change their sums, every later term is too.
        if (
            n * n > square
            and _absorbed(cos_term, cos_sum)
            and _absorbed(sin_term, sin_sum)
        ):
            break
        cos_sum = _CONTEXT.add(cos_sum, cos_term)
        sin_sum = _CONTEXT.add(sin_sum, sin_term)
    return _CONTEXT.plus(sin_sum), _CONTEXT.plus(cos_sum)


def _absorbed(term: Decimal, total: Decimal) -> bool:
    """Whether adding ``term``, or any term smaller than it, leaves ``total`` unchanged.

    A term below half the spacing of the decimals next to ``total`` rounds away.  Just below a
    power of ten that spacing is a tenth of the one above, so the bound uses the smaller one:
    ``|term| < 10^(e - PRECISION - 1)`` for a total of exponent ``e``.
    """
    if term.is_zero():
        return True
    return not total.is_zero() and term.adjusted() < total.adjusted() - PRECISION - 1


def atan2(y: Decimal, x: Decimal) -> Decimal:
    """Angle of the vector (x, y) in (-pi, pi]."""
    if x == 0 and y == 0:
        raise ValueError(internal("atan2 of the zero vector"))
    if x == 0:
        return HALF_PI if y > 0 else _CONTEXT.minus(HALF_PI)
    angle = _atan(_CONTEXT.divide(y, x))
    if x > 0:
        return angle
    return _CONTEXT.add(angle, PI) if y >= 0 else _CONTEXT.subtract(angle, PI)


def _atan(value: Decimal) -> Decimal:
    """Arctangent by three argument halvings and the Taylor series."""
    halvings = 3
    for _ in range(halvings):
        value = _CONTEXT.divide(
            value,
            _CONTEXT.add(ONE, sqrt(_CONTEXT.add(ONE, _CONTEXT.multiply(value, value)))),
        )
    square = _CONTEXT.multiply(value, value)
    term, total = value, value
    for n in range(1, SERIES_TERMS):
        term = _CONTEXT.minus(_CONTEXT.multiply(term, square))
        increment = _CONTEXT.divide(term, Decimal(2 * n + 1))
        # After the halvings |value| < tan(pi/16) < 0.2, so each increment is under a 25th of
        # the one before: once one is too small to change the sum, every later one is too.
        if _absorbed(increment, total):
            break
        total = _CONTEXT.add(total, increment)
    return _CONTEXT.multiply(total, Decimal(2**halvings))


def acos(value: Decimal) -> Decimal:
    """Arc cosine of a value in [-1, 1]."""
    return atan2(sqrt(_CONTEXT.subtract(ONE, _CONTEXT.multiply(value, value))), value)


def ceil_div(numerator: Decimal, denominator: Decimal) -> int:
    """The smallest integer not below ``numerator / denominator`` (both positive)."""
    quotient = _CONTEXT.divide(numerator, denominator)
    floor = int(quotient.to_integral_value(rounding=ROUND_FLOOR))
    return floor if quotient == floor else floor + 1


def round_half_up(value: Decimal) -> int:
    """Nearest integer, a tie rounding up (the lattice rounding of the whole pipeline)."""
    return int(_CONTEXT.add(value, HALF).to_integral_value(rounding=ROUND_FLOOR))


def vector_angle(
    centre: tuple[Fraction, Fraction], point: tuple[Fraction, Fraction]
) -> Decimal:
    """Polar angle of ``point`` around ``centre`` in (-pi, pi]."""
    return atan2(to_decimal(point[1] - centre[1]), to_decimal(point[0] - centre[0]))


def sweep_magnitude(start_angle: Decimal, end_angle: Decimal, *, ccw: bool) -> Decimal:
    """Angle travelled from ``start_angle`` to ``end_angle`` in the given direction, in [0, 2 pi)."""
    return _wrap(
        _CONTEXT.subtract(end_angle, start_angle)
        if ccw
        else _CONTEXT.subtract(start_angle, end_angle)
    )


def _wrap(delta: Decimal) -> Decimal:
    wrapped = _CONTEXT.remainder(delta, TWO_PI)
    return _CONTEXT.add(wrapped, TWO_PI) if wrapped < 0 else wrapped


class Rotation:
    """Rotates a vector by a fixed angle, step after step, in decimals."""

    def __init__(self, step: Decimal, *, ccw: bool) -> None:
        sine, cosine = sin_cos(step)
        self.cos = cosine
        self.sin = sine if ccw else _CONTEXT.minus(sine)

    def apply(self, x: Decimal, y: Decimal) -> tuple[Decimal, Decimal]:
        """The vector turned by one step."""
        return (
            _CONTEXT.subtract(
                _CONTEXT.multiply(x, self.cos), _CONTEXT.multiply(y, self.sin)
            ),
            _CONTEXT.add(
                _CONTEXT.multiply(x, self.sin), _CONTEXT.multiply(y, self.cos)
            ),
        )
