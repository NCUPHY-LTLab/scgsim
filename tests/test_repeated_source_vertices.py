"""A source polygon that lists a vertex twice in a row models as the shape it draws (Issue #107).

gdsfactory also stores a circle's closing point, so its 32-gon arrives as 33 points. KLayout
starts the stored walk at the lowest point, so the repeat sits mid-walk: (r, 0) is listed twice.
The repeated vertex is a zero-length edge: SCGSim drops it when it reads the polygon, with no
tolerance.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import pytest

from scgsim.geometry import build_gds_stack_geometry_input
from scgsim.geometry.source.occurrence import read_polygons_by_layer

gdstk = pytest.importorskip("gdstk")

# A 32-gon of radius 10 um on the 1 nm lattice, walked from its lowest vertex (0, -10) as KLayout stores it.
_ANGLES = (*range(24, 32), *range(24))
CLEAN_32_GON = tuple(
    (
        round(10 * math.cos(2 * math.pi * k / 32), 3),
        round(10 * math.sin(2 * math.pi * k / 32), 3),
    )
    for k in _ANGLES
)
# The gdsfactory spelling: the closing point (10, 0) is listed again right after itself.
GDSFACTORY_32_GON = (*CLEAN_32_GON[:9], CLEAN_32_GON[8], *CLEAN_32_GON[9:])
# The same 32-gon with its first vertex repeated as an explicit closing point.
CLOSED_32_GON = (*CLEAN_32_GON, CLEAN_32_GON[0])


def write_source(tmp_path, name, disk, *, fused):
    """A GDS holding the disk pad on layer 1 and its stack; ``fused`` reads it as a fused selector component."""
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("REPEATED_VERTEX")
    cell.add(gdstk.Polygon(disk, layer=1))
    gds = tmp_path / f"{name}.gds"
    library.write_gds(gds, timestamp=datetime(2000, 1, 1))
    geometry = {"z_um": 0, "thickness_um": 1, "route_ab_fused_selector_mode": fused}
    stack = {
        "layers": [
            {
                "semantic_id": "pad",
                "role": "conductor",
                "material_id": "PEC",
                "layer": 1,
                "datatype": 0,
                "geometry": geometry,
            }
        ],
        "solution_regions": {},
        "materials": {"PEC": {"kind": "conductor"}},
        "metadata": {"top_cell_name": cell.name},
    }
    stack_path = tmp_path / f"{name}.json"
    stack_path.write_text(json.dumps(stack), encoding="utf-8")
    return gds, stack_path


def test_the_repeated_vertex_survives_a_gds_round_trip(tmp_path):
    """The case is real: gdstk writes and reads the repeated point back unchanged."""
    gds, _ = write_source(tmp_path, "gdsfactory", GDSFACTORY_32_GON, fused=False)
    (polygon,) = (
        gdstk.read_gds(os.fspath(gds)).cells[0].get_polygons(layer=1, datatype=0)
    )
    assert len(polygon.points) == 33


@pytest.mark.parametrize(
    "disk", [GDSFACTORY_32_GON, CLOSED_32_GON], ids=["mid_walk", "closing"]
)
def test_reading_drops_the_repeated_vertex_and_keeps_clean_polygons(disk):
    """The read disk has the 32 distinct vertices in their order; a polygon without a repeat is not rebuilt."""
    cell = gdstk.Cell("READ")
    cell.add(gdstk.Polygon(disk, layer=1))
    clean = gdstk.Polygon(CLEAN_32_GON, layer=3)
    cell.add(clean)
    polygons = read_polygons_by_layer(cell)
    (read_disk,) = polygons[(1, 0)]
    assert [tuple(point) for point in read_disk.points.tolist()] == list(CLEAN_32_GON)
    assert polygons[(3, 0)][0].points.tolist() == clean.points.tolist()


def test_reading_keeps_near_but_unequal_neighbours():
    """Only exactly equal neighbours are dropped: a vertex 1 nm from the next one stays."""
    near = ((0.0, 0.0), (10.0, 0.0), (10.001, 0.0), (10.0, 10.0), (0.0, 10.0))
    cell = gdstk.Cell("NEAR")
    cell.add(gdstk.Polygon(near, layer=1))
    (polygon,) = read_polygons_by_layer(cell)[(1, 0)]
    assert len(polygon.points) == len(near)


@pytest.mark.parametrize(
    "fused", [False, True], ids=["layer_polygon", "fused_selector"]
)
@pytest.mark.parametrize(
    "disk", [GDSFACTORY_32_GON, CLOSED_32_GON], ids=["mid_walk", "closing"]
)
def test_a_32_gon_with_a_repeated_vertex_models_like_the_clean_32_gon(
    tmp_path, disk, fused
):
    """The whole source read gives the same polygons and Entities for either spelling of the disk."""
    clean, repeated = (
        build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
        for gds, stack in (
            write_source(tmp_path, "clean", CLEAN_32_GON, fused=fused),
            write_source(tmp_path, "repeated", disk, fused=fused),
        )
    )
    assert repeated.polygons == clean.polygons
    assert repeated.entities == clean.entities
    (pad,) = clean.polygons
    assert len(pad.exterior) == 32
