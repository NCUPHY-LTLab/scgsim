"""Public, normalized geometry cases used by the SGB method figures.

These are compiler inputs, not GDSFactory cells or device designs. Coordinates
are micrometres. The complex case extends the existing public junction fixture
with its documented square hole; no geometric distance bands are requested.
"""

from dataclasses import replace
from math import sqrt

from scgsim.geometry import (
    ApproximationProfile,
    BoundaryCurveChainSpec,
    GeometryBuildInput,
    SemanticEntitySpec,
    SourceCurveSpec,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.source.grid import apply_geometry_grid
from test_rectangular_junction_geometry import physical_fixture


HOLE = ((-1.5, 0.5), (-0.5, 0.5), (-0.5, 1.5), (-1.5, 1.5))


def junction_hole_case():
    source, stack, profile = physical_fixture()
    source = replace(
        source,
        polygons=tuple(
            replace(polygon, holes=(HOLE,))
            if polygon.polygon_id == "left" else polygon
            for polygon in source.polygons
        ),
        metadata={**source.metadata, "source_dbu_um": 0.001},
    )
    return apply_geometry_grid(source, GeometryGridContext(0.001)), stack, profile


def two_volume_case():
    """A silicon region and vacuum share the horizontal z=0 face."""
    outer = ((0, 0), (2, 0), (2, 2), (0, 2))
    entities = []
    regions = {}
    for name, kind, material, lo, hi in (
        ("substrate", "dielectric", "Si", -1, 0),
        ("upper_air", "vacuum", "vacuum", 0, 1),
    ):
        geometry = {
            "outer_loop": outer, "hole_loops": (),
            "z_min_um": lo, "z_max_um": hi, "thickness_um": hi - lo,
            "domain_bounds_um": {
                "x_min_um": 0, "x_max_um": 2, "y_min_um": 0,
                "y_max_um": 2, "z_min_um": lo, "z_max_um": hi,
            },
        }
        entities.append(SemanticEntitySpec(
            name, "solution_region", material, kind, 0, "polygon",
            geometry=geometry,
        ))
        regions[name] = {"semantic_id": name, "geometry": geometry, "metadata": {}}
    source = GeometryBuildInput((), tuple(entities), regions,
                                {"source_dbu_um": 0.001})
    stack = {
        "layers": [], "solution_regions": regions,
        "materials": {
            "Si": {"kind": "dielectric", "permittivity": 4, "loss_tangent": 0},
            "vacuum": {"kind": "vacuum", "permittivity": 1, "loss_tangent": 0},
        }, "metadata": {},
    }
    return apply_geometry_grid(source, GeometryGridContext(0.001)), stack, ApproximationProfile("solid", "v1", {})


def circular_hole_case():
    """An explicitly authored circular hole; distinct from the square-hole case."""
    source, stack, profile = junction_hole_case()
    q = 0.5 / sqrt(2)
    points = ((-0.5, 1), (-1, 1.5), (-1.5, 1), (-1, 0.5))
    middle = ((-1 + q, 1 + q), (-1 - q, 1 + q),
              (-1 - q, 1 - q), (-1 + q, 1 - q))
    curves = tuple(SourceCurveSpec(
        "circular_arc", (points[i], middle[i], points[(i + 1) % 4])
    ) for i in range(4))
    chain = BoundaryCurveChainSpec(
        "circular-hole", "chip/left", (-1.75, 1), curves,
        role="hole", hole_index=0, source_occurrence_path="chip",
    )
    return replace(source, boundary_curves=(chain,)), stack, profile
