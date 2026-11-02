"""Shared grid inputs and assertions for geometry grid contract tests."""

from decimal import Decimal

from scgsim.geometry.models.grid import GeometryGridContext

GRID = GeometryGridContext(0.001)


def rectangle(x0, y0, x1, y1):
    """Return a counter-clockwise rectangular ring for the given bounds."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def assert_lattice(points):
    """Assert that every XY coordinate lies on the 0.001 um lattice."""
    for point in points:
        for value in point[:2]:
            scaled = Decimal(str(value)) / Decimal("0.001")
            assert scaled == scaled.to_integral_value()


def ring_area(ring):
    """Return the absolute shoelace area of a closed coordinate ring."""
    return (
        abs(
            sum(
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
            )
        )
        / 2
    )
