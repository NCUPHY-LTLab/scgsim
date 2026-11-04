"""Independent erosion oracle for the inset bands (KLayout), used by examples and tests only.

Owns the cross-check demanded for brief section 6: erode the same support region with KLayout's integer
region engine (DBU = 1 grid unit = 1 nm), by the *polygonal* route -- dilate the region's complement by a regular
512-gon and subtract -- and compare band areas with the exact-carrier result of the pipeline (chord polygon plus
analytic circular segments of the realised arcs).

Error sources of the oracle, reported with the table: chords of the polygonised source arcs (sagitta ~0.01 unit),
the 512-gon disk (relative area error ~2.5e-5 of the disk), and integer rounding of every coordinate (+-0.5 unit,
which for a 100-unit margin is the dominant term).
"""

from __future__ import annotations

import math

import klayout.db as db

from sgbproto.arcs import face_exact_area
from sgbproto.inset import SupportRegion
from sgbproto.pipeline import BuildResult
from sgbproto.records import ArcCarrier, LineEdge


def region_polygons(region: SupportRegion, chord_sagitta: float = 0.01) -> db.Region:
    """The support region as a KLayout region (arcs polygonised finely)."""
    hull, holes = None, []
    for ri, ring in enumerate(region.rings):
        pts: list[db.Point] = []
        for prim in ring:
            if isinstance(prim, LineEdge):
                pts.append(db.Point(*prim.p))
                continue
            r = float(prim.radius)
            t0, sweep = prim.sweep()
            n = max(2, math.ceil(abs(sweep) / (2 * math.acos(1 - chord_sagitta / r))))
            cx, cy = float(prim.centre[0]), float(prim.centre[1])
            for k in range(n):
                t = t0 + sweep * k / n
                pts.append(db.Point(round(cx + r * math.cos(t)), round(cy + r * math.sin(t))))
        if ri == 0:
            hull = pts
        else:
            holes.append(pts)
    polygon = db.Polygon(hull)
    for h in holes:
        polygon.insert_hole(h)
    return db.Region(polygon)


def disk_polygon(radius: int, sides: int = 512) -> db.Polygon:
    return db.Polygon([db.Point(round(radius * math.cos(2 * math.pi * i / sides)),
                                round(radius * math.sin(2 * math.pi * i / sides))) for i in range(sides)])


def eroded_klayout(region: db.Region, margin: int, sides: int = 512) -> db.Region:
    """``region (-) disk(margin)``: subtract the complement dilated by a ``sides``-gon."""
    if margin == 0:
        return region.merged()
    box = region.bbox().enlarged(2 * margin + 10, 2 * margin + 10)
    complement = db.Region(box) - region
    return (region - complement.minkowski_sum(disk_polygon(margin, sides))).merged()


def exact_band_areas(result: BuildResult, family_id: str, level_index: int) -> dict[int, float]:
    """Area (grid^2) of every band of a family from the level arrangement, arcs realised exactly."""
    layer = result.level_layers[level_index]
    areas: dict[int, float] = {}
    for fi in range(1, len(layer.arr.faces)):
        for fam, k in layer.labels[fi].bands:
            if fam == family_id:
                areas[k] = areas.get(k, 0.0) + face_exact_area(layer.arr, layer.edge_curve, fi)
    return areas


def comparison_rows(result: BuildResult, region: SupportRegion, family_id: str, level_index: int) -> list[dict]:
    """Per margin: area of ``R_m`` (union of bands >= k) exact vs KLayout, plus the oracle's own noise estimate."""
    kl_region = region_polygons(region)
    bands = exact_band_areas(result, family_id, level_index)
    rows = []
    for k, m in enumerate((0, *region.margins)):
        exact = sum(a for band, a in bands.items() if band >= k)
        eroded = eroded_klayout(kl_region, m)
        polygon_area = float(eroded.area())
        perimeter = sum(poly.perimeter() for poly in eroded.each())
        rows.append({
            "region": family_id, "margin_units": m, "components_klayout": eroded.count(),
            "exact_um2": exact * 1e-6, "klayout_um2": polygon_area * 1e-6,
            "difference_um2": (exact - polygon_area) * 1e-6,
            "relative": (exact - polygon_area) / exact if exact else 0.0,
            "oracle_rounding_bound_um2": perimeter * 0.5 * 1e-6,  # +-0.5 unit on every boundary point
        })
    return rows
