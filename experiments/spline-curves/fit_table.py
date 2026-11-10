"""Arc counts and deviations of the curves of the study, against the polygons the layouts draw today.

For one edge of each scene (cell 14 and 15: the inner signal edge, the Euler centreline offset by w / 2; cell 21:
the upper strip edge), it prints

* the polygon the test scenes draw (as the PDK samples it, or resampled to about 3 degrees per vertex): its chords
  and its largest distance from the exact curve;
* for each fitting bound: the arcs of the biarc fit and of the lattice chain, and the chain's largest distance
  from the exact curve.

Usage (repository root)::

    uv run --locked --no-sync python experiments/spline-curves/fit_table.py
"""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "tests"), str(HERE)]

import arcchain as ac  # noqa: E402
import sgb_supported_pdk_cells as pdk  # noqa: E402
from sgb_supported_kit import offset_path  # noqa: E402

BOUNDS_NM = (math.sqrt(2) / 2, 5.0, 50.0, 200.0)
X, Y, MINUS_X = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
    (Fraction(-1), Fraction(0)),
)


def polygons() -> dict[str, tuple[ac.Curve, tuple, dict[str, list]]]:
    """Each edge: its exact curve, its end tangents, and the polygons drawn for it today."""
    half = pdk.SPLINE_HALF_WIDTH
    hermite = ac.translate(ac.hermite_strip_centreline(), 0, half)
    bspline = ac.translate(ac.bspline_strip_centreline(), 0, half)

    def shifted(points: list) -> list:
        return [(x, y + half) for x, y in points]

    return {
        "cell14 inner edge (Euler 90, +3.5 um)": (
            ac.offset(ac.euler_bend(90), 3.5),
            (X, Y),
            {"PDK polygon (14 chords)": offset_path(pdk.EULER_90, 3.5)},
        ),
        "cell15 inner edge (Euler 180, +5 um)": (
            ac.offset(ac.euler_bend(180), 5.0),
            (X, MINUS_X),
            {"PDK polygon (30 chords)": offset_path(pdk.EULER_180, 5.0)},
        ),
        "cell21 interpolation strip, upper edge": (
            hermite,
            (X, None),  # the strip ends on the die edge: no closing tangent
            {
                "PDK polygon": shifted(pdk.hermite_centreline()),
                "resampled to 3 deg": shifted(
                    pdk.resample_by_turn(pdk.hermite_centreline(1024), 3.0)
                ),
            },
        ),
        "cell21 B-spline strip, upper edge": (
            bspline,
            (X, None),
            {
                "PDK polygon": shifted(pdk.bspline_centreline()),
                "resampled to 3 deg": shifted(
                    pdk.resample_by_turn(pdk.bspline_centreline(1024), 3.0)
                ),
            },
        ),
    }


def main() -> int:
    """Print one JSON record per edge."""
    for name, (curve, (start, end), drawn) in polygons().items():
        record: dict[str, object] = {"edge": name, "polygons": {}, "fits": []}
        for label, points in drawn.items():
            record["polygons"][label] = {  # type: ignore[index]
                "chords": len(points) - 1,
                "max_deviation_nm": round(ac.polygon_deviation_nm(curve, points), 1),
            }
        for bound in BOUNDS_NM:
            fitted = ac.fit(curve, bound / 1000)
            chain = ac.tangent_chain(curve, bound / 1000, start, end)
            record["fits"].append(  # type: ignore[union-attr]
                {
                    "bound_nm": round(bound, 3),
                    "arcs": len(fitted),
                    "chain_max_deviation_nm": round(chain.max_deviation_nm, 2),
                    "smallest_radius_um": round(min(_radius(a) for a in chain.arcs), 2),
                    "shortest_arc_um": round(
                        min(math.dist(a.start, a.end) for a in chain.arcs) / 1000, 3
                    ),
                }
            )
        print(json.dumps(record), flush=True)
    return 0


def _radius(arc: ac.LatticeArc) -> float:
    if arc.centre is None:
        return math.inf
    return math.dist(arc.start, (float(arc.centre[0]), float(arc.centre[1]))) / 1000


if __name__ == "__main__":
    raise SystemExit(main())
