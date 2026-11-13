"""The scenes of the spline-curve study: the polygon scenes of the test suite and their arc-chain versions.

Polygon scenes come unchanged from ``tests/sgb_supported_pdk_cells`` (cells 14, 15 and 21, the latter both resampled
to about 3 degrees per vertex and as the PDK draws it with 258 points).  An arc-chain scene draws the same
conductors with the exact curve converted, before SGB, into a chain of circular arcs (``arcchain``) between the
same straight edges.  The arcs enter SGB through ``arc_through_points``, the function the authored ``circular_arc``
boundary curves use today: no SGB code changes.

Scene names: ``<cell>_arcs`` (exactly tangent chain, fitting bound = the grid's snap bound, sqrt(2)/2 nm),
``<cell>_arcs_naive`` (joints rounded, small kinks), ``<cell>_arcs_<n>nm`` (tangent chain, bound n nm).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

import arcchain as ac
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import (
    GRID,
    METAL_TOP,
    Conductor,
    Scene,
    catalogue,
    die_footprint,
)

from scgsim.geometry.sgb.exact.carriers import ArcCarrier, Lineage, LineEdge
from scgsim.geometry.sgb.exact.grid import exact_z
from scgsim.geometry.sgb.exact.prisms import Footprint, PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.exact.source_arcs import arc_through_points

SNAP_BOUND_UM = (
    math.sqrt(2) / 2 * GRID.size_um
)  # the grid's snap bound: half a grid diagonal
Um = tuple[float, float]
Dir = tuple[Fraction, Fraction]


# --------------------------------------------------------------------------- paths of lines and curves


@dataclass(frozen=True)
class CurvePiece:
    """A smooth curve between two straight edges: it leaves along ``start`` and arrives along ``end``."""

    curve: ac.Curve
    start: Dir
    end: (
        Dir | None
    )  # None: the curve ends on the die edge, with no straight edge to join

    def reversed(self) -> CurvePiece:
        """The same piece travelled the other way (both ends join straight edges)."""
        assert self.end is not None
        return CurvePiece(
            ac.reverse(self.curve),
            (-self.end[0], -self.end[1]),
            (-self.start[0], -self.start[1]),
        )


Piece = tuple[Um, Um] | CurvePiece  # a straight edge (two points) or a curve


def reverse_path(path: list[Piece]) -> list[Piece]:
    """A path travelled the other way."""
    return [
        p.reversed() if isinstance(p, CurvePiece) else (p[1], p[0])
        for p in reversed(path)
    ]


@dataclass
class Fitting:
    """How curves become arcs: the bound (um), the chain kind, and what the conversion measured."""

    bound_um: float = SNAP_BOUND_UM
    kind: str = "tangent"
    deviations_nm: list[float] = field(default_factory=list)
    kinks_rad: list[float] = field(default_factory=list)
    arcs: int = 0


def ring_primitives(
    entity: str, ring: int, path: list[Piece], fitting: Fitting
) -> list[LineEdge | ArcCarrier]:
    """The ring of straight edges and arc carriers that a closed path of pieces becomes (lattice units)."""
    out: list[LineEdge | ArcCarrier] = []

    def lineage() -> Lineage:
        return Lineage("polygon_edge", entity, 0, ring, len(out))

    for piece in path:
        if not isinstance(piece, CurvePiece):
            p, q = ac.lattice_point(piece[0]), ac.lattice_point(piece[1])
            if p != q:
                out.append(LineEdge(p, q, len(out)))
            continue
        if fitting.kind == "tangent":
            chain = ac.tangent_chain(
                piece.curve, fitting.bound_um, piece.start, piece.end
            )
        else:
            chain = ac.naive_chain(piece.curve, fitting.bound_um, *_leads(piece))
        fitting.deviations_nm.append(chain.max_deviation_nm)
        fitting.kinks_rad.extend(chain.kinks_rad)
        fitting.arcs += len(chain.arcs)
        for arc in chain.arcs:
            if arc.centre is None:
                out.append(LineEdge(arc.start, arc.end, len(out)))
                continue
            what = f"{entity} ring {ring} arc {len(out)}"
            out.extend(
                arc_through_points(
                    f"{entity}/ring{ring}/arc{len(out)}",
                    arc.start,
                    arc.through(),
                    arc.end,
                    lineage(),
                    what,
                    grid=GRID,
                )
            )
    _check_closed(out)
    return out


def _leads(piece: CurvePiece) -> tuple[Dir | None, Dir | None]:
    """The directions of the straight leads a curve joins at its start and its end, ``None`` where it joins none.

    A CPW curve (cells 13 to 15) joins a straight lead at both ends.  A curve that ends on the die edge
    (cell 21's strip edges) meets an end cap at a corner at both ends: its ``start`` is only the direction its
    chain leaves along, not a lead it continues.
    """
    if piece.end is None:
        return None, None
    return piece.start, piece.end


def _check_closed(ring: list[LineEdge | ArcCarrier]) -> None:
    def ends(p: LineEdge | ArcCarrier) -> tuple[tuple[int, int], tuple[int, int]]:
        if isinstance(p, LineEdge):
            return p.p, p.q
        return (int(p.start[0]), int(p.start[1])), (int(p.end[0]), int(p.end[1]))

    for a, b in zip(ring, [*ring[1:], ring[0]], strict=True):
        assert ends(a)[1] == ends(b)[0], (ends(a), ends(b))


def oriented(
    ring: list[LineEdge | ArcCarrier], ccw: bool
) -> list[LineEdge | ArcCarrier]:
    """The ring in the orientation SGB requires (outer counter-clockwise)."""
    points: list[tuple[int, int]] = []
    for p in ring:
        points.extend([p.p] if isinstance(p, LineEdge) else p.witness_points()[:-1])
    twice = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, [*points[1:], points[0]], strict=True)
    )
    return ring if (twice > 0) == ccw else [p.reversed() for p in reversed(ring)]


# --------------------------------------------------------------------------- arc-chain scenes


@dataclass
class ArcConductor:
    """A conductor whose outline is a ring of straight edges and arcs."""

    entity: str
    net: str
    outer: list[LineEdge | ArcCarrier]


@dataclass
class ArcScene:
    """A sample layout drawn with arc chains; ``fitting`` records the conversion."""

    name: str
    die: tuple[float, float, float, float]
    conductors: list[ArcConductor]
    fitting: Fitting
    conversion_seconds: float = 0.0


def catalogue_of(scene: Scene | ArcScene) -> PrismCatalogue:
    """Route B prisms of a polygon or an arc-chain scene (substrate, vacuum, 0.2 um metal)."""
    if isinstance(scene, Scene):
        return catalogue(scene)
    die = die_footprint(scene)  # type: ignore[arg-type] - only .die is read
    prisms = [
        PrismRecord(
            "substrate", die, exact_z(-100), exact_z(0), "substrate", "dielectric"
        ),
        PrismRecord("vacuum", die, exact_z(0), exact_z(100), "vacuum", "vacuum"),
    ]
    for c in scene.conductors:
        footprint, arcs = Footprint.from_primitives(c.outer)
        prisms.append(
            PrismRecord(
                c.entity,
                footprint,
                exact_z(0),
                exact_z(METAL_TOP),
                "Al",
                "conductor",
                net_id=c.net,
                arcs=arcs,
            )
        )
    return PrismCatalogue(GRID, tuple(prisms), name=scene.name)


def boundary_pieces(scene: Scene | ArcScene) -> dict[str, object]:
    """Pieces of the conductor outlines as SGB receives them, and what the conversion measured."""
    if isinstance(scene, Scene):
        return {
            c.entity: {
                "straight_edges": len(c.outer) + sum(len(h) for h in c.holes),
                "arc_pieces": 0,
            }
            for c in scene.conductors
        }
    out: dict[str, object] = {
        c.entity: {
            "straight_edges": sum(isinstance(p, LineEdge) for p in c.outer),
            "arc_pieces": sum(isinstance(p, ArcCarrier) for p in c.outer),
            "witness_chords": sum(
                len(p.witness_points()) - 1
                for p in c.outer
                if isinstance(p, ArcCarrier)
            ),
        }
        for c in scene.conductors
    }
    f = scene.fitting
    out["fitting"] = {
        "kind": f.kind,
        "bound_nm": round(f.bound_um * 1000, 4),
        "fitted_arcs": f.arcs,
        "max_deviation_nm": round(max(f.deviations_nm), 3),
        "kinks_over_1e-9_rad": sum(k > 1e-9 for k in f.kinks_rad),
        "largest_kink_rad": max(f.kinks_rad, default=0.0),
        "conversion_seconds": round(scene.conversion_seconds, 2),
    }
    return out


X: Dir = (Fraction(1), Fraction(0))
Y: Dir = (Fraction(0), Fraction(1))
MINUS_X: Dir = (Fraction(-1), Fraction(0))


def _cpw_path(
    centre: ac.Curve, end_point: Um, end_dir: Dir, tail: Um, d: float
) -> list[Piece]:
    """Straight lead-in from x = -20, the curve's offset by ``d``, straight lead-out (``tail`` from its end)."""
    tx, ty = float(end_dir[0]), float(end_dir[1])
    end = (end_point[0] - d * ty, end_point[1] + d * tx)
    return [
        ((-20.0, d), (0.0, d)),
        CurvePiece(ac.offset(centre, d), X, end_dir),
        (end, (end[0] + tail[0], end[1] + tail[1])),
    ]


def _cpw(  # noqa: PLR0913 - one CPW scene's parts
    name: str,
    centre: ac.Curve,
    end: tuple[Um, Dir, Um],
    w: float,
    die: tuple[float, float, float, float],
    corners: tuple[list[Um], list[Um]],
    fitting: Fitting,
) -> ArcScene:
    end_point, end_dir, tail = end

    def path(d: float) -> list[Piece]:
        return _cpw_path(centre, end_point, end_dir, tail, d)

    def ring(entity: str, pieces: list[Piece]) -> list[LineEdge | ArcCarrier]:
        return oriented(ring_primitives(entity, 0, pieces, fitting), ccw=True)

    gap = w / 2 + pdk.CPW_GAP
    upper, lower = path(w / 2), reverse_path(path(-w / 2))
    signal = [
        *upper,
        (_last(upper[-1]), _first(lower[0])),
        *lower,
        (_last(lower[-1]), _first(upper[0])),
    ]
    left = _through(path(gap), corners[0])
    right = _through(path(-gap), corners[1])
    return ArcScene(
        name,
        die,
        [
            ArcConductor("signal", "signal", ring("signal", signal)),
            ArcConductor("ground_left", "Ground", ring("ground_left", left)),
            ArcConductor("ground_right", "Ground", ring("ground_right", right)),
        ],
        fitting,
    )


def _first(p: Piece) -> Um:
    return tuple(p.curve.at(p.curve.t0)[0]) if isinstance(p, CurvePiece) else p[0]  # type: ignore[return-value]


def _last(p: Piece) -> Um:
    return tuple(p.curve.at(p.curve.t1)[0]) if isinstance(p, CurvePiece) else p[1]  # type: ignore[return-value]


def _through(path: list[Piece], corners: list[Um]) -> list[Piece]:
    """A path closed through die corners back to its start."""
    points = [_last(path[-1]), *corners, _first(path[0])]
    return [*path, *zip(points, points[1:], strict=False)]


def cell13_arcs(fitting: Fitting) -> ArcScene:
    """Cell 13's circular bend (R 100, w 7 / g 6) drawn as exact arcs: the reference of a circle."""
    return _cpw(
        "cell13_arcs",
        ac.circle_bend((0.0, 100.0), 100.0, -90.0, 90.0),
        ((100.0, 100.0), Y, (0.0, 20.0)),
        7.0,
        (-20.0, -30.0, 130.0, 120.0),
        ([(-20.0, 120.0)], [(130.0, 120.0), (130.0, -30.0), (-20.0, -30.0)]),
        fitting,
    )


def cell14_arcs(fitting: Fitting) -> ArcScene:
    """Cell 14's 90-degree Euler CPW bend (w 7 / g 6), the exact Euler edges as arc chains."""
    return _cpw(
        "cell14_arcs",
        ac.euler_bend(90.0),
        ((100.0, 100.0), Y, (0.0, 20.0)),
        7.0,
        (-20.0, -30.0, 130.0, 120.0),
        ([(-20.0, 120.0)], [(130.0, 120.0), (130.0, -30.0), (-20.0, -30.0)]),
        fitting,
    )


def cell15_arcs(fitting: Fitting) -> ArcScene:
    """Cell 15's 180-degree Euler U-turn (w 10 / g 6), the exact Euler edges as arc chains."""
    return _cpw(
        "cell15_arcs",
        ac.euler_bend(180.0),
        ((0.0, 200.0), MINUS_X, (-20.0, 0.0)),
        10.0,
        (-20.0, -30.0, 190.0, 230.0),
        ([], [(-20.0, 230.0), (190.0, 230.0), (190.0, -30.0), (-20.0, -30.0)]),
        fitting,
    )


def cell21_arcs(kind: str, fitting: Fitting) -> ArcScene:
    """Cell 21's 10 um spline strip: both edges (the centreline +-5 um in y) as arc chains, straight end caps."""
    centre = (
        ac.hermite_strip_centreline()
        if kind == "interpolation"
        else ac.bspline_strip_centreline()
    )
    half = pdk.SPLINE_HALF_WIDTH
    # both ends lie on the die edge: the chains leave along the curve's exact (horizontal) end tangent and
    # arrive along their own, so no joint keeps a kink
    upper = CurvePiece(ac.translate(centre, 0.0, half), X, None)
    lower = CurvePiece(ac.reverse(ac.translate(centre, 0.0, -half)), MINUS_X, None)
    path: list[Piece] = [
        upper,
        (_last(upper), _first(lower)),
        lower,
        (_last(lower), _first(upper)),
    ]
    ring = oriented(ring_primitives("strip", 0, path, fitting), ccw=True)
    return ArcScene(
        f"cell21_{kind}_arcs",
        (-200.0, -100.0, 200.0, 150.0),
        [ArcConductor("strip", "strip", ring)],
        fitting,
    )


def _timed(build: Callable[[Fitting], ArcScene], fitting: Fitting) -> ArcScene:
    import time  # noqa: PLC0415 - the conversion time is part of the record

    started = time.perf_counter()
    scene = build(fitting)
    scene.conversion_seconds = time.perf_counter() - started
    return scene


ARC_BUILDERS: dict[str, Callable[[Fitting], ArcScene]] = {
    "cell13": cell13_arcs,
    "cell14": cell14_arcs,
    "cell15": cell15_arcs,
    "cell21_interpolation": lambda f: cell21_arcs("interpolation", f),
    "cell21_bspline": lambda f: cell21_arcs("bspline", f),
}

SCENES: dict[str, Callable[[], Scene | ArcScene]] = {
    name: pdk.SCENES[name]
    for name in (
        "cell13_circular_bend",
        "cell14_euler_bend",
        "cell15_euler_u_turn",
        "cell21_interpolation_resampled",
        "cell21_bspline_resampled",
    )
}
SCENES.update(pdk.DRAWN_SPLINES)
for _cell, _build in ARC_BUILDERS.items():
    SCENES[f"{_cell}_arcs"] = lambda b=_build: _timed(b, Fitting())
    SCENES[f"{_cell}_arcs_naive"] = lambda b=_build: _timed(b, Fitting(kind="naive"))
    for _nm in (5, 50):
        SCENES[f"{_cell}_arcs_{_nm}nm"] = lambda b=_build, n=_nm: _timed(
            b, Fitting(bound_um=n / 1000)
        )


# --------------------------------------------------------------------------- band reference of an arc scene


def _witness_ring(ring: list[LineEdge | ArcCarrier]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for p in ring:
        points.extend([p.p] if isinstance(p, LineEdge) else p.witness_points()[:-1])
    return points


def witness_scene(scene: ArcScene) -> Scene:
    """The polygon scene of the arcs' witness points: the reference ``sgb_supported_kit.reference`` erodes."""
    return Scene(
        scene.name,
        (),
        scene.die,
        [Conductor(c.entity, c.net, _witness_ring(c.outer)) for c in scene.conductors],
    )


def witness_allowance_um2(scene: Scene | ArcScene) -> float:
    """What the witness polygon may add to a band's difference: two contours of the outline's length, g/4 each."""
    if isinstance(scene, Scene):
        return 0.0
    g = GRID.size_um
    length = 0.0
    for c in scene.conductors:
        ring = _witness_ring(c.outer)
        length += (
            sum(
                math.dist(a, b) for a, b in zip(ring, [*ring[1:], ring[0]], strict=True)
            )
            * g
        )
    return 2 * length * g / 4
