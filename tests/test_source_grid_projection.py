"""Keep source-grid projection payloads and failure ordering stable."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
from test_rectangular_junction_geometry import fixture

from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import GeometryBuildInput, LayoutPolygonSpec
from scgsim.geometry.source.grid_projection import apply_geometry_grid


def _payload_sha256(value: GeometryBuildInput) -> str:
    encoded = json.dumps(
        asdict(value), sort_keys=True, separators=(",", ":"), default=list
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_projection_preserves_complete_public_payload_hash():
    """Lock the full conductor, port, context, and metadata result."""
    build, _ = fixture()
    build = replace(build, metadata={**build.metadata, "source_dbu_um": 0.001})
    projected = apply_geometry_grid(build, GeometryGridContext(0.001))

    # The typed Route B fixture uses cutout_boundary_shell.  Its polygons are built by
    # hand, so their drawn_rings stay empty.
    assert _payload_sha256(projected) == (
        "0630af5ea6e2ed878094543d184d1cae1b1a9e37ca64b68063583bcc1ec4e038"
    )


def test_projection_keeps_port_contact_failure_and_message():
    """Retain contact validation for an unselected third conductor."""
    build, _ = fixture()
    polygon = LayoutPolygonSpec(
        "other", "metal", ((4, 2.0002), (5, 2.0002), (5, 3), (4, 3))
    )
    other_entity = replace(
        build.entities[0], semantic_id="chip/other", polygon_ids=("other",)
    )
    build = replace(
        build,
        polygons=(*build.polygons, polygon),
        entities=(*build.entities, other_entity),
    )

    with pytest.raises(ValueError) as error:
        apply_geometry_grid(build, GeometryGridContext(0.001))

    assert str(error.value) == (
        "port contact changed at grid: sheet, chip/other, other on layers junction, metal "
        "(grid g = 0.001 um): they touch along an edge after snapping at "
        "(4.000, 2.000)..(5.000, 2.000) um; the narrowest drawn gap between them is "
        "0.2 nm = 0.2 grid steps. Fix the layout: draw the gap or contact between them "
        "on the grid, widening or closing it."
    )


def test_bound_context_mismatch_fails_before_projection():
    """Reject a conflicting policy before doing source geometry work."""
    build, _ = fixture()
    build = replace(
        build,
        metadata={
            **build.metadata,
            "geometry_grid": GeometryGridContext(0.001).to_payload(),
        },
    )

    with pytest.raises(
        ValueError, match="internal: an input on grid .* was projected on grid"
    ):
        apply_geometry_grid(build, GeometryGridContext(0.002))


def test_json_source_audit_preserves_hole_list_in_python_payload():
    """Keep raw JSON audit rings as lists in projected source metadata."""
    exterior = [[0, 0], [5, 0], [5, 5], [0, 5]]
    holes = [[[1, 1], [2, 1], [2, 2], [1, 2]]]
    build = GeometryBuildInput(
        polygons=(
            LayoutPolygonSpec(
                "p",
                "metal",
                ((0, 0), (5, 0), (5, 5), (0, 5)),
                metadata={
                    "geometry_grid_source_identity": {
                        "source_entity_id": "e",
                        "source_polygon_id": "raw",
                    }
                },
            ),
        ),
        entities=(),
        metadata={
            "geometry_grid_source_audits": [
                {
                    "source_entity_id": "e",
                    "requested_polygons": [
                        {
                            "source_polygon_id": "raw",
                            "exterior": exterior,
                            "holes": holes,
                        }
                    ],
                }
            ]
        },
    )

    projected = apply_geometry_grid(build, GeometryGridContext(0.001))
    raw_geometry = projected.polygons[0].metadata["unquantized_source_geometry"]
    payload_geometry = asdict(projected)["polygons"][0]["metadata"][
        "unquantized_source_geometry"
    ]

    assert raw_geometry["exterior"] is exterior
    assert raw_geometry["holes"] is holes
    assert payload_geometry["exterior"] == exterior
    assert payload_geometry["holes"] == holes
    assert isinstance(payload_geometry["holes"], list)
