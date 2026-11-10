"""Smooth curves, their biarc fit and the chain of circular arcs SGB can carry (issue 109, option 1).

Nothing here is production code.  It answers whether a smooth boundary curve, converted *before* SGB sees it into
straight edges and circular arcs, builds and bands in SGB as SGB is today.

* **Curves** (micrometres, double precision): the gdsfactory Euler bend (``p = 0.5``, effective radius 100 um, as
  cells 14 and 15 of ``tests/sgb_supported_pdk_cells`` sample it), its exact offsets (the CPW edges), a circle, and
  the two spline strips of cell 21 (the PDK's piecewise cubic Hermite ``y(x)`` and its clamped cubic B-spline, both
  translated by +-5 um in y as the PDK draws them).
* **Biarc fit** (exact arithmetic of reals, here doubles): between two points of the curve with their tangents, two
  circular arcs that meet with a common tangent (a *tangent joint*).  An interval is halved until the biarc lies
  within the fitting bound of the curve.
* **Lattice chain**: SGB keeps every end point on the 1 nm lattice, so the joints must be lattice points.  Two ways
  to get there are compared:

  - ``naive``: round each joint to the nearest lattice point and pass each arc through its rounded ends and its
    old middle point.  The arcs no longer share a tangent: every joint gets a small kink.
  - ``tangent``: keep the joints tangent *exactly*.  Each arc starts with the end tangent of the arc before
    (a rational direction), so its centre lies on the line through the joint and the previous centre; the
    lattice point of each joint is chosen among its neighbours so that the arc stays closest to the curve.
    The last joint of a run closes on the fixed end tangent; it is the one joint with a (very small) kink.

Lengths in the chain are grid units (integers for points, ``Fraction`` for centres).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.interpolate import BSpline

GRID_UM = 0.001
Vec = tuple[float, float]


# --------------------------------------------------------------------------- curves


@dataclass(frozen=True)
class Curve:
    """A smooth plane curve on ``[t0, t1]``: point and unit tangent (um); ``breaks`` must be fit joints."""

    point: Callable[[np.ndarray], np.ndarray]  # (n,) -> (n, 2)
    tangent: Callable[[np.ndarray], np.ndarray]  # (n,) -> (n, 2), unit
    t0: float
    t1: float
    breaks: tuple[float, ...] = ()

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Point and unit tangent at one parameter."""
        return self.point(np.array([t]))[0], self.tangent(np.array([t]))[0]


def offset(curve: Curve, d: float) -> Curve:
    """The curve moved ``d`` um to its left along the normal (a CPW edge of a centreline)."""

    def point(t: np.ndarray) -> np.ndarray:
        tan = curve.tangent(t)
        return curve.point(t) + d * np.column_stack((-tan[:, 1], tan[:, 0]))

    return Curve(point, curve.tangent, curve.t0, curve.t1, curve.breaks)


def translate(curve: Curve, dx: float, dy: float) -> Curve:
    """The curve moved by ``(dx, dy)`` um (the PDK's spline strip edges)."""
    return Curve(
        lambda t: curve.point(t) + (dx, dy),
        curve.tangent,
        curve.t0,
        curve.t1,
        curve.breaks,
    )


def reverse(curve: Curve) -> Curve:
    """The same curve travelled the other way."""
    a, b = curve.t0, curve.t1
    return Curve(
        lambda t: curve.point(a + b - t),
        lambda t: -curve.tangent(a + b - t),
        a,
        b,
        tuple(sorted(a + b - x for x in curve.breaks)),
    )


def euler_bend(angle_deg: float, p: float = 0.5) -> Curve:
    """gdsfactory's Euler bend (``use_eff=True``): clothoid, circular arc, clothoid; from (0, 0) heading +x.

    Scaled so that the 90-degree bend ends at (100, 100) and the 180-degree bend at (0, 200) um, as
    ``EULER_90`` and ``EULER_180`` of the test scenes.  Parameter: arc length.
    """
    alpha = math.radians(angle_deg)
    sp = math.sqrt(p * alpha)  # clothoid length at R0 = 1 (curvature equals arc length)
    rp = 1 / sp  # radius of the middle arc
    total = 2 * sp + rp * alpha * (1 - p)

    def theta(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        first = s**2 / 2
        middle = p * alpha / 2 + (s - sp) / rp
        last = alpha - (total - s) ** 2 / 2
        return np.where(s < sp, first, np.where(s <= total - sp, middle, last))

    # cumulative position on a fine grid, then Gauss-Legendre from the nearest node
    nodes = np.linspace(0.0, total, 4097)
    gx, gw = np.polynomial.legendre.leggauss(12)

    def integrate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        half = (b - a) / 2
        mid = (b + a) / 2
        s = mid[:, None] + half[:, None] * gx[None, :]
        th = theta(s)
        return np.column_stack(
            ((np.cos(th) * gw).sum(1) * half, (np.sin(th) * gw).sum(1) * half)
        )

    steps = integrate(nodes[:-1], nodes[1:])
    cumulative = np.vstack(([0.0, 0.0], np.cumsum(steps, axis=0)))
    end = cumulative[-1]
    scale = (100.0 if angle_deg < 179 else 200.0) / end[1]

    def raw_point(s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, total)
        index = np.clip(np.searchsorted(nodes, s) - 1, 0, len(nodes) - 2)
        return cumulative[index] + integrate(nodes[index], s)

    def point(t: np.ndarray) -> np.ndarray:
        return raw_point(np.asarray(t) / scale) * scale

    def tangent(t: np.ndarray) -> np.ndarray:
        th = theta(np.asarray(t) / scale)
        return np.column_stack((np.cos(th), np.sin(th)))

    return Curve(point, tangent, 0.0, total * scale, (sp * scale, (total - sp) * scale))


def circle_bend(centre: Vec, r: float, start_deg: float, sweep_deg: float) -> Curve:
    """A circular arc (cell 13's centreline), parameter angle in radians."""
    a0 = math.radians(start_deg)
    sign = 1.0 if sweep_deg > 0 else -1.0

    def point(t: np.ndarray) -> np.ndarray:
        return np.column_stack((centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)))

    def tangent(t: np.ndarray) -> np.ndarray:
        return sign * np.column_stack((-np.sin(t), np.cos(t)))

    a1 = a0 + math.radians(sweep_deg)
    if sign < 0:  # parameter increases along travel: use the angle reversed
        return reverse(Curve(point, lambda t: -tangent(t), a1, a0))
    return Curve(point, tangent, a0, a1)


HERMITE_KNOTS = (
    (-200.0, 0.0, 0.0),
    (-100.0, 60.0, -0.2),
    (0.0, -40.0, 0.0),
    (100.0, 60.0, 0.2),
    (200.0, 0.0, 0.0),
)


def hermite_strip_centreline() -> Curve:
    """The PDK's ``interpolation_spline_strip`` law: piecewise cubic Hermite y(x), parameter x (um)."""
    xs = np.array([k[0] for k in HERMITE_KNOTS])

    def span(x: np.ndarray) -> tuple[np.ndarray, ...]:
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
        x0, x1 = xs[i], xs[i + 1]
        y0 = np.array([k[1] for k in HERMITE_KNOTS])[i]
        y1 = np.array([k[1] for k in HERMITE_KNOTS])[i + 1]
        m0 = np.array([k[2] for k in HERMITE_KNOTS])[i]
        m1 = np.array([k[2] for k in HERMITE_KNOTS])[i + 1]
        h = x1 - x0
        return (x - x0) / h, h, y0, y1, m0, m1

    def point(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        t, h, y0, y1, m0, m1 = span(x)
        y = (2 * t**3 - 3 * t**2 + 1) * y0 + (t**3 - 2 * t**2 + t) * h * m0
        y += (-2 * t**3 + 3 * t**2) * y1 + (t**3 - t**2) * h * m1
        return np.column_stack((x, y))

    def tangent(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        t, h, y0, y1, m0, m1 = span(x)
        dy = ((6 * t**2 - 6 * t) * y0 + (3 * t**2 - 4 * t + 1) * h * m0) / h
        dy += ((-6 * t**2 + 6 * t) * y1 + (3 * t**2 - 2 * t) * h * m1) / h
        n = np.hypot(1.0, dy)
        return np.column_stack((1.0 / n, dy / n))

    return Curve(point, tangent, -200.0, 200.0, (-100.0, 0.0, 100.0))


def bspline_strip_centreline() -> Curve:
    """The PDK's ``bspline_strip``: clamped cubic B-spline, knots (0, 0.5, 1), unit weights, parameter u."""
    control = np.array(
        [(-200.0, 0.0), (-100.0, 0.0), (0.0, 150.0), (100.0, -30.0), (200.0, -30.0)]
    )
    spline = BSpline(np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1.0]), control, 3)
    derivative = spline.derivative()

    def tangent(u: np.ndarray) -> np.ndarray:
        d = derivative(np.asarray(u, dtype=float))
        return d / np.hypot(d[:, 0], d[:, 1])[:, None]

    return Curve(
        lambda u: spline(np.asarray(u, dtype=float)), tangent, 0.0, 1.0, (0.5,)
    )


# --------------------------------------------------------------------------- biarc fit (reals)


@dataclass(frozen=True)
class RealArc:
    """An arc from ``a`` with unit tangent ``ta`` to ``b`` (um); ``centre`` is None for a straight piece."""

    a: np.ndarray
    ta: np.ndarray
    b: np.ndarray


def _arc_centre(
    a: np.ndarray, ta: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, float] | None:
    n = np.array([-ta[1], ta[0]])
    c = b - a
    den = 2 * float(c @ n)
    if abs(den) < 1e-14 * float(c @ c):
        return None
    lam = float(c @ c) / den
    return a + lam * n, lam


def distance_to_arc(
    points: np.ndarray, a: np.ndarray, ta: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Distance of each point to the arc ``a -> b`` with start tangent ``ta`` (a segment when straight)."""
    found = _arc_centre(a, ta, b)
    ends = np.minimum(np.hypot(*(points - a).T), np.hypot(*(points - b).T))
    if found is None:
        d = b - a
        s = np.clip(((points - a) @ d) / (d @ d), 0, 1)
        return np.hypot(*(points - (a + s[:, None] * d)).T)
    centre, lam = found
    r = abs(lam)
    radial = np.abs(np.hypot(*(points - centre).T) - r)
    # inside the arc's angular range: the chord side test against the arc's sweep
    va, vb = a - centre, b - centre
    ccw = lam > 0
    ang = lambda v: np.arctan2(v[..., 1], v[..., 0])  # noqa: E731
    sweep = (
        (ang(vb) - ang(va)) % (2 * np.pi) if ccw else (ang(va) - ang(vb)) % (2 * np.pi)
    )
    rel = (
        (ang(points - centre) - ang(va)) % (2 * np.pi)
        if ccw
        else (ang(va) - ang(points - centre)) % (2 * np.pi)
    )
    return np.where(rel <= sweep, radial, ends)


def biarc(
    p0: np.ndarray, t0: np.ndarray, p1: np.ndarray, t1: np.ndarray
) -> tuple[RealArc, RealArc]:
    """The equal-tangent-length biarc from ``p0`` (tangent ``t0``) to ``p1`` (tangent ``t1``)."""
    v = p1 - p0
    t = t0 + t1
    denom = 2 * (1 - float(t0 @ t1))
    if abs(denom) < 1e-12:
        d = float(v @ v) / (4 * float(v @ t1))
    else:
        vt = float(v @ t)
        d = (-vt + math.sqrt(vt * vt + denom * float(v @ v))) / denom
    joint = (p0 + d * t0 + p1 - d * t1) / 2
    first = RealArc(p0, t0, joint)
    tj = _end_tangent(first)
    return first, RealArc(joint, tj, p1)


def _end_tangent(arc: RealArc) -> np.ndarray:
    c = arc.b - arc.a
    return 2 * float(arc.ta @ c) / float(c @ c) * c - arc.ta


def biarc_deviation(
    curve: Curve, ta: float, tb: float, samples: int = 64
) -> tuple[float, RealArc, RealArc]:
    """Largest distance (um) of the curve on ``[ta, tb]`` from the biarc between its ends, and the biarc."""
    pa, va = curve.at(ta)
    pb, vb = curve.at(tb)
    first, second = biarc(pa, va, pb, vb)
    pts = curve.point(np.linspace(ta, tb, samples))
    dev = np.minimum(
        distance_to_arc(pts, first.a, first.ta, first.b),
        distance_to_arc(pts, second.a, second.ta, second.b),
    ).max()
    return float(dev), first, second


def fit(curve: Curve, bound_um: float) -> list[tuple[RealArc, float, float]]:
    """Biarcs within ``bound_um`` of the curve: (arc, t_start, t_end) in order.

    Greedy: from each start, the longest interval whose biarc fits (bisection on its end parameter).  The
    curve's ``breaks`` are always joints.
    """
    cuts = [curve.t0, *curve.breaks, curve.t1]
    out: list[tuple[RealArc, float, float]] = []
    for lo, hi in zip(cuts, cuts[1:], strict=False):
        ends = _greedy(curve, lo, hi, bound_um)
        # the same count with every piece at the smallest common deviation: no short remainder at the end
        tight, loose = 0.0, bound_um
        for _ in range(12):
            mid = (tight + loose) / 2
            trial = _greedy(curve, lo, hi, mid, limit=len(ends))
            if trial is not None and len(trial) == len(ends):
                loose, ends = mid, trial
            else:
                tight = mid
        ta = lo
        for tb in ends:
            _, first, second = biarc_deviation(curve, ta, tb)
            out.append((first, ta, _nearest_parameter(curve, ta, tb, first.b)))
            out.append((second, out[-1][2], tb))
            ta = tb
    return out


def _greedy(
    curve: Curve, lo: float, hi: float, bound_um: float, limit: int | None = None
) -> list[float] | None:
    """End parameters of the longest-first biarc intervals within ``bound_um``; None past ``limit`` pieces."""
    ends: list[float] = []
    ta = lo
    while ta < hi:
        if limit is not None and len(ends) >= limit:
            return None
        if biarc_deviation(curve, ta, hi)[0] <= bound_um:
            ends.append(hi)
            break
        good, bad = ta, hi
        for _ in range(24):
            mid = (good + bad) / 2
            if biarc_deviation(curve, ta, mid)[0] <= bound_um:
                good = mid
            else:
                bad = mid
        ends.append(good)
        ta = good
    return ends


def _nearest_parameter(curve: Curve, ta: float, tb: float, point: np.ndarray) -> float:
    """Parameter of the curve point nearest to ``point`` on ``[ta, tb]`` (dense search, then a finer one)."""
    fine = np.linspace(ta, tb, 2001)
    i = int(np.argmin(np.hypot(*(curve.point(fine) - point).T)))
    finer = np.linspace(fine[max(i - 1, 0)], fine[min(i + 1, 2000)], 2001)
    return float(finer[np.argmin(np.hypot(*(curve.point(finer) - point).T))])


# --------------------------------------------------------------------------- the lattice chain


@dataclass(frozen=True)
class LatticeArc:
    """An arc between lattice points (grid units) with an exact rational centre, or a straight piece."""

    start: tuple[int, int]
    end: tuple[int, int]
    centre: tuple[Fraction, Fraction] | None  # None: straight
    ccw: bool = True

    def through(self) -> tuple[Fraction, Fraction]:
        """A rational point exactly on the circle, half-way along the arc (the start reflected about the
        line from the centre at a quarter of the sweep)."""
        assert self.centre is not None
        cx, cy = self.centre
        vx, vy = self.start[0] - cx, self.start[1] - cy
        a0 = math.atan2(float(vy), float(vx))
        a1 = math.atan2(float(self.end[1] - cy), float(self.end[0] - cx))
        sweep = (a1 - a0) % (2 * math.pi) if self.ccw else -((a0 - a1) % (2 * math.pi))
        q = a0 + sweep / 4
        ux, uy = Fraction(math.cos(q)), Fraction(math.sin(q))
        k = 2 * (vx * ux + vy * uy) / (ux * ux + uy * uy)
        return (cx + k * ux - vx, cy + k * uy - vy)


def _limit(direction: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    """The same direction with small denominators (error below 1e-15 rad)."""
    m = max(abs(direction[0]), abs(direction[1]))
    return (
        (direction[0] / m).limit_denominator(10**16),
        (direction[1] / m).limit_denominator(10**16),
    )


def _arc_from(
    start: tuple[int, int], tangent: tuple[Fraction, Fraction], end: tuple[int, int]
) -> tuple[LatticeArc, tuple[Fraction, Fraction]]:
    """The arc leaving ``start`` along ``tangent`` that ends at ``end``, and its exact end tangent."""
    tx, ty = tangent
    cx, cy = end[0] - start[0], end[1] - start[1]
    cc = cx * cx + cy * cy
    cn = -ty * cx + tx * cy
    tc = tx * cx + ty * cy
    end_tangent = (2 * tc * cx / cc - tx, 2 * tc * cy / cc - ty)
    if cn == 0:
        return LatticeArc(start, end, None), (tx, ty)
    lam = Fraction(cc) / (2 * cn)
    centre = (start[0] - lam * ty, start[1] + lam * tx)
    return LatticeArc(start, end, centre, lam > 0), end_tangent


def _arc_into(
    start: tuple[int, int], end: tuple[int, int], tangent: tuple[Fraction, Fraction]
) -> tuple[LatticeArc, tuple[Fraction, Fraction]]:
    """The arc from ``start`` that arrives at ``end`` along ``tangent``, and its start tangent."""
    back, start_tangent = _arc_from(end, (-tangent[0], -tangent[1]), start)
    arc = LatticeArc(start, end, back.centre, not back.ccw)
    return arc, (-start_tangent[0], -start_tangent[1])


def _angle(
    u: tuple[Fraction, Fraction] | np.ndarray, v: tuple[Fraction, Fraction] | np.ndarray
) -> float:
    ux, uy = float(u[0]), float(u[1])
    vx, vy = float(v[0]), float(v[1])
    return abs(math.atan2(ux * vy - uy * vx, ux * vx + uy * vy))


def lattice_point(p_um: np.ndarray) -> tuple[int, int]:
    """Nearest lattice point (grid units) of a point in um."""
    return (round(float(p_um[0]) / GRID_UM), round(float(p_um[1]) / GRID_UM))


def _deviation(arc: LatticeArc, pts_grid: np.ndarray) -> float:
    a = np.array(arc.start, dtype=float)
    b = np.array(arc.end, dtype=float)
    if arc.centre is None:
        return float(
            distance_to_arc(pts_grid, a, (b - a) / np.hypot(*(b - a)), b).max()
        )
    c = np.array([float(arc.centre[0]), float(arc.centre[1])])
    r = math.hypot(*(a - c))
    v = a - c
    ta = np.array([-v[1], v[0]]) / r * (1 if arc.ccw else -1)
    return float(distance_to_arc(pts_grid, a, ta, b).max())


@dataclass(frozen=True)
class Chain:
    """The lattice arcs of one fitted curve, its largest deviation (nm) and the joint kinks (rad)."""

    arcs: tuple[LatticeArc, ...]
    max_deviation_nm: float
    kinks_rad: tuple[float, ...]


def tangent_chain(
    curve: Curve,
    bound_um: float,
    start_tangent: tuple[Fraction, Fraction],
    end_tangent: tuple[Fraction, Fraction] | None,
    search: int = 2,
    closing_reach: int = 12,
) -> Chain:
    """Lattice arcs with exactly shared tangents, steered to the curve, closing on ``end_tangent``.

    ``start_tangent`` and ``end_tangent`` are the exact directions of the straight edges the curve joins: the
    first arc leaves along the first, the last arrives along the second.  An end point and an end tangent
    cannot both be met by lattice arcs that all share their tangents, so the joint before the last arc keeps
    a kink; it is chosen among ``(2 closing_reach + 1)^2`` lattice points for the smallest kink that keeps the
    two arcs within two grid steps of the curve.  ``end_tangent=None`` (a curve that ends on the die edge, with
    no straight edge after it) lets the last arc arrive along whatever tangent it has: no kink at all.
    """
    pieces = fit(curve, bound_um)
    joints = [curve.at(pieces[0][1])[0]] + [arc.b for arc, _, _ in pieces]
    samples = [curve.point(np.linspace(ta, tb, 24)) / GRID_UM for _, ta, tb in pieces]
    current = lattice_point(joints[0])
    tangent = start_tangent
    arcs: list[LatticeArc] = []
    kinks: list[float] = []
    last = len(pieces) - 1
    for k in range(len(pieces)):
        target = lattice_point(joints[k + 1])
        if k == last:
            if end_tangent is None:
                arc, _ = _arc_from(current, tangent, target)
                kinks.append(0.0)
            else:
                arc, arriving = _arc_into(current, target, end_tangent)
                kinks.append(_angle(tangent, arriving))
            arcs.append(arc)
            break
        best: (
            tuple[float, LatticeArc, tuple[Fraction, Fraction], tuple[int, int]] | None
        ) = None
        closing_joint = k + 1 == last and end_tangent is not None
        reach = closing_reach if closing_joint else search
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                candidate = (target[0] + dx, target[1] + dy)
                if candidate == current:
                    continue
                arc, leaving = _arc_from(current, tangent, candidate)
                cost = _deviation(arc, samples[k])
                if closing_joint:
                    # the next arc closes on the end tangent: the smallest kink here within two grid steps
                    assert end_tangent is not None
                    closing, arriving = _arc_into(
                        candidate, lattice_point(joints[-1]), end_tangent
                    )
                    deviation = max(cost, _deviation(closing, samples[k + 1]))
                    kink = _angle(leaving, arriving)
                    cost = kink if deviation <= 2 else 1 + deviation
                else:
                    # look one arc ahead: the next arc leaves along this one's end tangent
                    following, _ = _arc_from(
                        candidate, leaving, lattice_point(joints[k + 2])
                    )
                    cost = max(cost, _deviation(following, samples[k + 1]))
                if best is None or cost < best[0]:
                    best = (cost, arc, leaving, candidate)
        assert best is not None
        _, arc, leaving, current = best
        arcs.append(arc)
        tangent = _limit(leaving)
        kinks.append(0.0)
    deviation = max(_deviation(a, s) for a, s in zip(arcs, samples, strict=True))
    return Chain(tuple(arcs), deviation * GRID_UM * 1000, tuple(kinks))


def naive_chain(curve: Curve, bound_um: float) -> Chain:
    """Each biarc joint rounded to the lattice, each arc through its rounded ends and its old middle point."""
    pieces = fit(curve, bound_um)
    arcs: list[LatticeArc] = []
    samples = []
    for arc, ta, tb in pieces:
        start, end = lattice_point(arc.a), lattice_point(arc.b)
        found = _arc_centre(arc.a / GRID_UM, arc.ta, arc.b / GRID_UM)
        samples.append(curve.point(np.linspace(ta, tb, 24)) / GRID_UM)
        if found is None:
            arcs.append(LatticeArc(start, end, None))
            continue
        centre, lam = found
        # the arc's middle point (unrounded), then the circle through the rounded ends and it
        va = arc.a / GRID_UM - centre
        vb = arc.b / GRID_UM - centre
        mid = centre + abs(lam) * (va + vb) / np.hypot(*(va + vb))
        ax, ay = map(Fraction, start)
        bx, by = map(Fraction, end)
        mx, my = Fraction(float(mid[0])), Fraction(float(mid[1]))
        det = 2 * (ax * (by - my) + bx * (my - ay) + mx * (ay - by))
        a2, b2, m2 = ax * ax + ay * ay, bx * bx + by * by, mx * mx + my * my
        ux = (a2 * (by - my) + b2 * (my - ay) + m2 * (ay - by)) / det
        uy = (a2 * (mx - bx) + b2 * (ax - mx) + m2 * (bx - ax)) / det
        arcs.append(LatticeArc(start, end, (ux, uy), lam > 0))
    kinks = []
    for first, second in zip(arcs, arcs[1:], strict=False):
        kinks.append(
            _angle(_tangent_at(first, at_end=True), _tangent_at(second, at_end=False))
        )
    deviation = max(_deviation(a, s) for a, s in zip(arcs, samples, strict=True))
    return Chain(tuple(arcs), deviation * GRID_UM * 1000, tuple(kinks))


def _tangent_at(arc: LatticeArc, *, at_end: bool) -> np.ndarray:
    p = np.array(arc.end if at_end else arc.start, dtype=float)
    if arc.centre is None:
        d = np.array(arc.end, dtype=float) - np.array(arc.start, dtype=float)
        return d / np.hypot(*d)
    v = p - np.array([float(arc.centre[0]), float(arc.centre[1])])
    t = np.array([-v[1], v[0]]) / np.hypot(*v)
    return t if arc.ccw else -t


def polygon_deviation_nm(curve: Curve, polygon_um: list[Vec]) -> float:
    """Largest distance (nm) from the curve to a polyline drawn for it (the PDK's sampled polygon)."""
    pts = curve.point(np.linspace(curve.t0, curve.t1, 20001))
    best = np.full(len(pts), np.inf)
    for a, b in zip(polygon_um, polygon_um[1:], strict=False):
        a, b = np.asarray(a), np.asarray(b)
        d = b - a
        s = np.clip(((pts - a) @ d) / (d @ d), 0, 1)
        best = np.minimum(best, np.hypot(*(pts - (a + s[:, None] * d)).T))
    return float(best.max() * 1000)
