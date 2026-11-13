"""Independent erosion oracle for the inset bands (KLayout), used by tests only.

Erodes the same support region with KLayout's integer region engine (DBU = 1 grid unit = 1 nm) by the
*polygonal* route -- dilate the complement by a regular 512-gon and subtract -- and compares band areas
with the exact-carrier result of the pipeline (chord polygon plus the analytic circular segments of the
realised arcs).

Error sources of the oracle: the chords of the polygonised source arcs (sagitta about 0.01 unit), the
512-gon disk (relative area error about 2.5e-5 of the disk) and the integer rounding of every coordinate
(+-0.5 unit, which dominates for small margins).  The rows report that rounding bound beside the difference.
"""

from __future__ import annotations

import math

import klayout.db as db
from sgb_scenes import exact_band_areas

from scgsim.geometry.exact.carriers import LineEdge
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.support import SupportRegion

DISK_SIDES = 512
CHORD_SAGITTA = 0.01


def region_polygons(region: SupportRegion) -> db.Region:
    """The support region as a KLayout region (arcs polygonised finely)."""
    hull, holes = None, []
    for ri, ring in enumerate(region.rings):
        points: list[db.Point] = []
        for prim in ring:
            if isinstance(prim, LineEdge):
                points.append(db.Point(*prim.p))
                continue
            r = float(prim.radius)
            t0, sweep = (float(v) for v in prim.sweep())
            n = max(2, math.ceil(abs(sweep) / (2 * math.acos(1 - CHORD_SAGITTA / r))))
            cx, cy = float(prim.centre[0]), float(prim.centre[1])
            for k in range(n):
                t = t0 + sweep * k / n
                points.append(
                    db.Point(round(cx + r * math.cos(t)), round(cy + r * math.sin(t)))
                )
        if ri == 0:
            hull = points
        else:
            holes.append(points)
    polygon = db.Polygon(hull)
    for hole in holes:
        polygon.insert_hole(hole)
    return db.Region(polygon)


def region_of_rings(region: SupportRegion) -> db.Region:
    """The support region of a multi-component face: every ring as an even-odd polygon, merged."""
    result = db.Region()
    for ring in region.rings:
        points = []
        for prim in ring:
            if isinstance(prim, LineEdge):
                points.append(db.Point(*prim.p))
            else:
                r = float(prim.radius)
                t0, sweep = (float(v) for v in prim.sweep())
                n = max(
                    2, math.ceil(abs(sweep) / (2 * math.acos(1 - CHORD_SAGITTA / r)))
                )
                cx, cy = float(prim.centre[0]), float(prim.centre[1])
                points.extend(
                    db.Point(
                        round(cx + r * math.cos(t0 + sweep * k / n)),
                        round(cy + r * math.sin(t0 + sweep * k / n)),
                    )
                    for k in range(n)
                )
        result ^= db.Region(db.Polygon(points))
    return result.merged()


def port_region(model: ConformalModel, family_id: str) -> db.Region:
    """The footprints of the lumped ports on the level of a family (a port face is no band's surface)."""
    level = next(
        f.target_levels[0]
        for f in model.labelled.catalogue.families
        if f.family_id == family_id
    )
    result = db.Region()
    for prism in model.labelled.catalogue.prisms:
        if prism.material_kind == "port" and prism.z0 == level:
            for ring in prism.footprint.rings():
                result ^= db.Region(db.Polygon([db.Point(*p) for p in ring]))
    return result.merged()


def disk_polygon(radius: int, sides: int = DISK_SIDES) -> db.Polygon:
    """A regular polygon approximating a disk (KLayout has no round shapes in its integer engine)."""
    return db.Polygon(
        [
            db.Point(
                round(radius * math.cos(2 * math.pi * i / sides)),
                round(radius * math.sin(2 * math.pi * i / sides)),
            )
            for i in range(sides)
        ]
    )


def eroded_klayout(region: db.Region, margin: int) -> db.Region:
    """``region (-) disk(margin)``: subtract the complement dilated by a polygonal disk."""
    if margin == 0:
        return region.merged()
    box = region.bbox().enlarged(2 * margin + 10, 2 * margin + 10)
    complement = db.Region(box) - region
    return (region - complement.minkowski_sum(disk_polygon(margin))).merged()


def comparison_rows(model: ConformalModel, family_id: str) -> list[dict]:
    """Per margin: area of ``R_m`` (union of the bands >= k) outside the ports, exact against KLayout, plus the oracle's own noise estimate."""
    family = next(
        f for f in model.labelled.catalogue.families if f.family_id == family_id
    )
    support = family.support
    assert support is not None
    ports = port_region(model, family_id)
    kl_region = region_of_rings(support)
    bands = exact_band_areas(model, family_id, include_ports=False)
    rows = []
    for k, margin in enumerate((0, *support.margins)):
        exact = sum(area for band, area in bands.items() if band >= k)
        eroded = (
            eroded_klayout(kl_region, margin) - ports
        )  # the band faces leave the port to the port
        polygon_area = float(eroded.area())
        perimeter = sum(poly.perimeter() for poly in eroded.each())
        rows.append(
            {
                "family": family_id,
                "margin_units": margin,
                "components_klayout": eroded.count(),
                "exact_um2": exact * 1e-6,
                "klayout_um2": polygon_area * 1e-6,
                "difference_um2": (exact - polygon_area) * 1e-6,
                "relative": (exact - polygon_area) / exact if exact else 0.0,
                "oracle_rounding_bound_um2": perimeter * 0.5 * 1e-6,
            }
        )
    return rows
