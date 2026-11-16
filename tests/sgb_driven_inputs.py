"""The source and SGB model of a two-port HFSS Driven Terminal request: two pads, a ground strip, two port sheets."""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sgb_inputs import JUNCTION_GRID_UM, rectangle, solution_region

from scgsim.aedt.preparation.sgb_model import PreparedSgbModel, prepare_sgb_model
from scgsim.aedt.specs.common import LumpedTerminalPort
from scgsim.geometry import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    PhysicalLayerDeclaration,
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
    SemanticEntitySpec,
)
from scgsim.geometry.models.grid import GeometryGridContext

TWO_PORT_PADS = (
    ("pad_1", "P1", ((0, 0), (5, 0), (5, 3), (0, 3))),
    ("pad_2", "P2", ((10, 0), (15, 0), (15, 3), (10, 3))),
    ("ground", "G", ((-2, -6), (17, -6), (17, -3), (-2, -3))),
)
# Each sheet: its id, port index, pad and x extent; it overlaps the strip and its pad by 1 um.
TWO_PORT_SHEETS = (("port_1", 1, "pad_1", 1, 4), ("port_2", 2, "pad_2", 11, 14))


def _two_port_sheet(name: str, index: int, pad: str, x0: float, x1: float):
    overlaps = tuple(
        PortSheetOverlapRecord(
            f"{name}_{host}",
            name,
            name,
            f"chip/{host}",
            host,
            rectangle(x0, x1, y0, y1),
        )
        for host, y0, y1 in (("ground", -4, -3), (pad, 0, 1))
    )
    return PortSheetRegionRecord(
        name,
        "port",
        name,
        rectangle(x0, x1, -4, 1),
        overlaps=overlaps,
        metadata={
            "source": "palace_lumped_port_sheet",
            "source_name": f"chip/{name}",
            "source_occurrence_path": "chip",
            "target_layer": "metal",
            "port_index": index,
            "direction": (0, 1, 0),
            "direction_raw": (0, 1, 0),
            "direction_sign_convention": "gdsfactory_port_orientation_outward",
        },
    )


def lumped_two_port_source(
    vacuum: str = "VACUUM_REGION", *, sheets=TWO_PORT_SHEETS
) -> tuple[GeometryBuildInput, dict[str, Any]]:
    """Two pads beside one ground strip, and a lumped-port sheet from the strip up to each pad.

    The pads ``chip/pad_1`` (Net ``P1``) and ``chip/pad_2`` (``P2``) and the strip ``chip/ground`` (``G``) are
    0.25 um face metals on a 2 um substrate.  Port sheet ``port_<k>`` spans the 3 um gap from the strip to
    pad ``k`` along +y: the source of a two-port HFSS Driven Terminal model.  ``sheets`` are the port sheets
    as :data:`TWO_PORT_SHEETS` gives them.
    """
    declaration = PhysicalLayerDeclaration(
        "film",
        "metal",
        {"level": "metal", "gds_pair": [1, 0]},
        (0, 0.25),
        "metal",
        "stack_film",
        {"normal_thickness_um": 0.25},
        {"layer_attachment_face": "lower"},
    )
    metals = tuple(
        SemanticEntitySpec(
            f"chip/{name}",
            "conductor",
            "metal",
            "conductor",
            1,
            "polygon",
            part_role="face_metal",
            net_id=net,
            polygon_ids=(name,),
            host_void_semantic_id=vacuum,
            geometry={
                "geometry_source": "gds_polygon",
                "z_um": 0,
                "thickness_um": 0.25,
                "outer_loop": loop,
            },
            metadata={
                "logical_layer_id": "metal",
                "physical_layer": declaration.to_payload(),
                "source_occurrence_path": "chip",
            },
        )
        for name, net, loop in TWO_PORT_PADS
    )
    sheets = tuple(_two_port_sheet(*sheet) for sheet in sheets)
    substrate = solution_region("substrate", "dielectric", "Si", (-4, 19, -8, 5, -2, 0))
    upper = solution_region(
        vacuum,
        "vacuum",
        "air",
        (-4, 19, -8, 5, -2 if vacuum == "VACUUM_REGION" else 0, 4.25),
    )
    regions = {
        item.semantic_id: {
            "semantic_id": item.semantic_id,
            "geometry": dict(item.geometry),
            "metadata": {},
        }
        for item in (substrate, upper)
    }
    stack = {
        "layers": [
            {
                "semantic_id": m.semantic_id,
                "geometry": dict(m.geometry),
                "metadata": dict(m.metadata),
            }
            for m in metals
        ],
        "solution_regions": regions,
        "materials": {
            "Si": {"kind": "dielectric", "permittivity": 4, "loss_tangent": 0},
            "air": {"kind": "vacuum", "permittivity": 1, "loss_tangent": 0},
            "metal": {"kind": "conductor"},
        },
        "metadata": {},
    }
    source = GeometryBuildInput(
        tuple(
            LayoutPolygonSpec(name, "metal", loop) for name, _, loop in TWO_PORT_PADS
        ),
        (*metals, substrate, upper),
        solution_regions=regions,
        metadata={
            "geometry_grid": GeometryGridContext(JUNCTION_GRID_UM).to_payload(),
            "source_dbu_um": JUNCTION_GRID_UM,
        },
        port_sheet_regions=sheets,
    )
    return source, stack


DRIVEN_PORTS = (
    LumpedTerminalPort(1, "p1", "port_1", ("chip/pad_1",), ("chip/ground",), 50.0),
    LumpedTerminalPort(2, "p2", "port_2", ("chip/pad_2",), ("chip/ground",), 50.0),
)


@functools.cache
def driven_model(modeling: str = "solid") -> PreparedSgbModel:
    """The SGB model of :func:`lumped_two_port_source` with :data:`DRIVEN_PORTS`, built once per session.

    ``solid`` is Route C (conductor solids), ``thin_film`` Route A (sheets on the substrate).
    """
    source, stack = lumped_two_port_source()
    folder = Path(tempfile.mkdtemp(prefix="scgsim-test-driven-model-"))
    atexit.register(shutil.rmtree, folder, True)
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling=modeling,
        output_dir=folder,
        ports=DRIVEN_PORTS,
    )
