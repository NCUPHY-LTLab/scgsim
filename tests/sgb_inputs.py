"""Real effective inputs for the SGB stage-S2 tests, built with the public source API.

Two routes to an effective ``GeometryBuildInput``:

* ``gds_pad_in_ground`` writes a GDS and a stack file, runs
  ``build_gds_stack_geometry_input`` (source stages S0), applies the automatic vacuum
  region and interprets the physical geometry (S1) with the ``solid`` profile;
* ``thin_film_pair`` composes the source input of two face metals joined by a lumped
  port and interprets it with the ``thin_film`` profile, which turns both metals into
  Route-A sheets.

* ``rectangular_junction_pair`` is the thin-film pair joined by a rectangular junction: the
  lumped-port sheet is the junction's central rectangle between two L-shaped terminals;
* ``geometry_plan_electrodes`` prepares a real ``GeometryPlan`` of two electrodes on a
  substrate and interprets its snapshot (needs gdsfactory).

All return the effective input and the ``RepresentationContext`` S2 consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gdstk

from scgsim.geometry import (
    ApproximationProfile,
    GeometryBuildInput,
    JunctionTerminalBinding,
    LayoutPolygonSpec,
    PhysicalLayerDeclaration,
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
    RectangularJunctionSpec,
    SemanticEntitySpec,
    build_gds_stack_geometry_input,
    interpret_physical_geometry,
    with_rectangular_junctions,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack

GRID_UM = 0.001


@dataclass(frozen=True)
class EffectiveInput:
    """An effective input and the representation context S1 produced for it."""

    build_input: GeometryBuildInput
    representation: RepresentationContext


# --------------------------------------------------------------------------- GDS route


def physical_layer(
    name: str, thickness_um: float, declaration: str | None
) -> dict[str, Any]:
    """The ``physical_layer`` metadata of a metal that stands on z = 0, as the PDK declares it (or leaves it out)."""
    if declaration is None:
        return {}
    film = declaration in {"lower", "upper"}
    return {
        "physical_layer": {
            "physical_layer_id": f"{name}_metal",
            "source_level_id": "metal",
            "source_level_and_gds_pair": {"level": "metal"},
            "source_interval_um": [0.0, thickness_um],
            "material_id": "Al",
            "feature_kind": "stack_film" if film else "finite_volume",
            "physical_dimensions": {"normal_thickness_um": thickness_um}
            if film
            else {"axial_span_um": thickness_um},
            "source_contact_facts": {"layer_attachment_face": declaration}
            if film
            else {},
            "actual_entity_contact_graph": None,
        }
    }


def gds_pad_in_ground(  # noqa: PLR0913 - a test builder with independent switches
    tmp_path: Path,
    *,
    thickness_um: float = 0.2,
    with_port: bool = True,
    pad_polygon: gdstk.Polygon | None = None,
    nets: bool = True,
    extra_metadata: dict[str, Any] | None = None,
    attachment: tuple[str | None, str | None] = ("lower", "lower"),
    die_um: float = 100.0,
) -> EffectiveInput:
    """A ``die_um`` (default 100 um) die with a ground plane, a 20 um pad in a 40 um cutout and a lumped port.

    The cutout, pad and port keep their size and sit in the middle of the die; the substrate is a tenth
    of the die thick and the vacuum padding a fifth of it (never less than the 10 and 20 um of the
    100 um die), so a larger die scales only the distances the benchmark of ``scgsim.meshing`` varies.

    Ground (GDS layer 1) and pad (layer 2) belong to one logical metal layer; the port
    (layer 9) bridges the 10 um gap and lands one micrometre on each host.  The metal is a film
    on the substrate top, as in the PDK: it declares its attachment at the lower face (z = 0), so
    the port lies on the substrate surface.  ``attachment`` is the declaration of (ground, pad):
    ``"lower"`` or ``"upper"`` for a film attached at that face, ``"finite_volume"`` for a body
    declaration without an attachment face, ``None`` for no physical declaration at all.
    """
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("TOP")
    centre = die_um / 2
    substrate_um = max(10.0, die_um / 10)
    die = gdstk.rectangle((0, 0), (die_um, die_um))
    cutout = gdstk.rectangle((centre - 20, centre - 20), (centre + 20, centre + 20))
    ground = gdstk.boolean(die, cutout, "not", precision=1e-9)[0]
    ground.layer = 1
    cell.add(ground)
    pad = pad_polygon or gdstk.rectangle(
        (centre - 10, centre - 10), (centre + 10, centre + 10)
    )
    pad.layer = 2
    cell.add(pad)
    if with_port:
        cell.add(
            gdstk.rectangle(
                (centre - 21, centre - 2), (centre - 9, centre + 2), layer=9
            )
        )
    gds = tmp_path / "pad_in_ground.gds"
    library.write_gds(gds, timestamp=datetime(2000, 1, 1))

    def metal(
        name: str, layer: int, net: str, face: str | None, **geometry: Any
    ) -> dict[str, Any]:
        """A metal layer record; by default it declares the PDK's film on the substrate top (attached at its lower face)."""
        record: dict[str, Any] = {
            "semantic_id": name,
            "role": "conductor",
            "material_id": "Al",
            "layer": layer,
            "datatype": 0,
            "geometry": {"z_um": 0, "thickness_um": thickness_um, **geometry},
            "metadata": {
                "logical_layer_id": "metal",
                **physical_layer(name, thickness_um, face),
            },
        }
        if nets:
            record["net_id"] = net
        return record

    stack: dict[str, Any] = {
        "layers": [
            metal("ground", 1, "G", attachment[0], route_ab_fused_selector_mode=True),
            metal("pad", 2, "P", attachment[1]),
        ],
        "solution_regions": {
            "substrate": {
                "material_id": "Si",
                "geometry": {
                    "z_um": -substrate_um,
                    "thickness_um": substrate_um,
                    "domain_bounds_um": {
                        "x_min_um": 0,
                        "x_max_um": die_um,
                        "y_min_um": 0,
                        "y_max_um": die_um,
                    },
                },
            }
        },
        "materials": {
            "Al": {"kind": "conductor"},
            "Si": {"kind": "dielectric", "permittivity": 11.45},
        },
        "metadata": {
            "top_cell_name": "TOP",
            "port_sheet_source_layers": [
                {
                    "source": "palace_lumped_port_sheet",
                    "name": "P1",
                    "layer": 9,
                    "datatype": 0,
                    "port_index": 1,
                    "target_layer": "metal",
                    "direction": [1, 0, 0],
                }
            ]
            if with_port
            else [],
            **(extra_metadata or {}),
        },
    }
    stack = dict(
        apply_vacuum_region_to_stack(
            stack, VacuumRegionSpec.from_padding(max(20.0, die_um / 5))
        )
    )
    stack_file = tmp_path / "pad_in_ground.json"
    stack_file.write_text(json.dumps(stack, default=list), encoding="utf-8")
    source = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack_file)
    effective, _, plan = interpret_physical_geometry(
        source,
        json.loads(stack_file.read_text()),
        ApproximationProfile("solid", "v1", {}),
    )
    return EffectiveInput(effective, plan.context)


# --------------------------------------------------------------------------- thin-film route


LEFT_EDGE_UM = 4.0
RIGHT_EDGE_UM = 6.0


def _rectangle(x0: float, x1: float, y0: float = 1.0, y1: float = 3.0):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _solution_region(name: str, kind: str, material: str, bounds, holes=()):
    x0, x1, y0, y1, z0, z1 = bounds
    return SemanticEntitySpec(
        name,
        "solution_region",
        material,
        kind,
        0,
        "polygon",
        geometry={
            "outer_loop": _rectangle(x0, x1, y0, y1),
            "hole_loops": holes,
            "z_min_um": z0,
            "z_max_um": z1,
            "thickness_um": z1 - z0,
            "domain_bounds_um": {
                "x_min_um": x0,
                "x_max_um": x1,
                "y_min_um": y0,
                "y_max_um": y1,
                "z_min_um": z0,
                "z_max_um": z1,
            },
        },
    )


def thin_film_pair(
    landing_um: float = 0.5, *, profile: str = "thin_film", bump: bool = False
) -> EffectiveInput:
    """Two 0.25 um face metals joined by a lumped port landing ``landing_um`` on each.

    The selected Thin Film profile collapses the film thickness, so both metals become
    Route-A sheets at z = 0 and the vacuum above them shrinks from 4.25 um to 4 um.  With
    ``profile="solid"`` the declared films stay finite.  With ``bump`` a 2.75 um tall
    ``finite_volume`` body stands on the left film (a bump or post): a finite body that is not a film.
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
    loops = {
        "left": _rectangle(0.0, LEFT_EDGE_UM, 0.0, 4.0),
        "right": _rectangle(RIGHT_EDGE_UM, 10.0, 0.0, 4.0),
    }
    metals = tuple(
        SemanticEntitySpec(
            f"chip/{name}",
            "conductor",
            "metal",
            "conductor",
            1,
            "polygon",
            part_role="face_metal",
            net_id=name,
            polygon_ids=(name,),
            host_void_semantic_id="upper_air",
            geometry={
                "geometry_source": "gds_polygon",
                "z_um": 0,
                "thickness_um": 0.25,
                "outer_loop": loop,
            },
            metadata={
                "logical_layer_id": "metal",
                "physical_layer": declaration.to_payload(),
            },
        )
        for name, loop in loops.items()
    )
    if bump:
        metals = (*metals, _bump_entity())
    port_loop = _rectangle(LEFT_EDGE_UM - landing_um, RIGHT_EDGE_UM + landing_um)
    port = PortSheetRegionRecord(
        "port_sheet",
        "port",
        "port_polygon",
        port_loop,
        overlaps=(
            PortSheetOverlapRecord(
                "left_overlap",
                "port_sheet",
                "port_polygon",
                "chip/left",
                "left",
                _rectangle(LEFT_EDGE_UM - landing_um, LEFT_EDGE_UM),
            ),
            PortSheetOverlapRecord(
                "right_overlap",
                "port_sheet",
                "port_polygon",
                "chip/right",
                "right",
                _rectangle(RIGHT_EDGE_UM, RIGHT_EDGE_UM + landing_um),
            ),
        ),
        metadata={
            "source_name": "chip/o_junction_lumped",
            "target_layer": "metal",
            "port_index": 1,
            "direction": (1, 0, 0),
            "direction_raw": (1, 0, 0),
            "direction_sign_convention": "gdsfactory_port_orientation_outward",
        },
    )
    polygons = (
        *(LayoutPolygonSpec(name, "metal", loop) for name, loop in loops.items()),
        *((LayoutPolygonSpec("bump", "bump", BUMP_LOOP),) if bump else ()),
        LayoutPolygonSpec("port_polygon", "port", port_loop),
    )
    substrate = _solution_region(
        "substrate", "dielectric", "Si", (-3, 13, -1, 5, -2, 0)
    )
    lower = _solution_region(
        "lower_air",
        "vacuum",
        "air",
        (-4, 14, -2, 6, -2, 0),
        (substrate.geometry["outer_loop"],),
    )
    upper = _solution_region("upper_air", "vacuum", "air", (-4, 14, -2, 6, 0, 4.25))
    regions = {
        item.semantic_id: {
            "semantic_id": item.semantic_id,
            "geometry": dict(item.geometry),
            "metadata": {},
        }
        for item in (substrate, lower, upper)
    }
    stack = {
        "layers": [
            {
                "semantic_id": item.semantic_id,
                "geometry": dict(item.geometry),
                "metadata": dict(item.metadata),
            }
            for item in metals
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
        polygons,
        (*metals, substrate, lower, upper),
        port_sheet_regions=(port,),
        solution_regions=regions,
        metadata={"geometry_grid": GeometryGridContext(GRID_UM).to_payload()},
    )
    effective, _, plan = interpret_physical_geometry(
        source, stack, ApproximationProfile(profile, "v1", {})
    )
    return EffectiveInput(effective, plan.context)


BUMP_LOOP = ((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0))


def _bump_entity() -> SemanticEntitySpec:
    """A ``finite_volume`` bump body on the left film: source Z 0.25 .. 3.0 um, one Net with the film."""
    declaration = PhysicalLayerDeclaration(
        "bump_layer",
        "bump",
        {"level": "bump", "gds_pair": [2, 0]},
        (0.25, 3.0),
        "metal",
        "finite_volume",
        {},
        {},
    )
    return SemanticEntitySpec(
        "chip/bump",
        "conductor",
        "metal",
        "conductor",
        2,
        "polygon",
        part_role="bump_body",
        net_id="left",
        polygon_ids=("bump",),
        host_void_semantic_id="upper_air",
        geometry={
            "geometry_source": "gds_polygon",
            "z_um": 0.25,
            "thickness_um": 2.75,
            "outer_loop": BUMP_LOOP,
        },
        metadata={
            "logical_layer_id": "bump",
            "physical_layer": declaration.to_payload(),
        },
    )


# --------------------------------------------------------------------------- GeometryPlan route


def _plan_level(name: str, material: str, interval: tuple[float, float], role: str):
    """One layer-stack level carrying its neutral physical declaration."""
    declaration = PhysicalLayerDeclaration(
        name, name, {"level": name}, interval, material, "finite_volume", {}, {}
    )
    return SimpleNamespace(
        material=material,
        zmin=interval[0],
        thickness=interval[1] - interval[0],
        mesh_order=1,
        info={
            "simulation_role": role,
            "include_in_component_simulation": True,
            "part_role": "signal",
            "physical_layer": declaration.to_payload(),
        },
    )


def electrode_plan(monkeypatch, *, gap_um: float = 10.0, same_net: bool = False):
    """A real, unprepared ``GeometryPlan``: a 20 um pad and a 30 um electrode ``gap_um`` apart.

    The Nets are A and B, or both A with ``same_net``.  A negative ``gap_um`` makes them overlap.  The
    component, layer stack and material records are small public stand-ins for a PDK.
    """
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    from scgsim.geometry import GeometryPlan  # noqa: PLC0415 -- the plan needs the optional layout stack above

    # Restore process-local PDK state after this public generic fixture.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    far = 20.0 + gap_um
    component.add_polygon(((0, 0), (20, 0), (20, 20), (0, 20)), layer=(1, 0))
    component.add_polygon(
        ((far, 0), (far + 30, 0), (far + 30, 20), (far, 20)), layer=(2, 0)
    )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "pad", "level": "metal", "gds_layer": [1, 0]},
            {"semantic_id": "electrode", "level": "metal", "gds_layer": [2, 0]},
        ],
    }
    stack = SimpleNamespace(
        layers={
            "VACUUM_REGION": _plan_level(
                "VACUUM_REGION", "vacuum", (-10, 12), "solution_region"
            ),
            "substrate": _plan_level("substrate", "Si", (-5, 0), "solution_region"),
            "metal": _plan_level("metal", "PEC", (0, 0.2), "conductor"),
        }
    )
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=stack,
        material_records={
            "PEC": {"kind": "conductor"},
            "vacuum": {"kind": "vacuum"},
            "Si": {"kind": "dielectric", "permittivity": 11.45},
        },
        coupon_padding_um=10,
        grid_size_um=GRID_UM,
    )
    if same_net:
        plan.set_nets({"A": ("chip/pad", "chip/electrode")})
    else:
        plan.set_nets({"A": ("chip/pad",), "B": ("chip/electrode",)})
    return plan


def geometry_plan_electrodes(monkeypatch, *, gap_um: float = 10.0) -> EffectiveInput:
    """A real ``GeometryPlan``: a 20 um pad and a 30 um electrode ``gap_um`` apart, on Nets A and B.

    The plan, its snapshot and the physical interpretation are the production code.
    """
    snapshot = electrode_plan(monkeypatch, gap_um=gap_um).prepare()
    effective, _, physical = interpret_physical_geometry(
        snapshot.geometry_input,
        snapshot.stack,
        ApproximationProfile("solid", "v1", {}),
    )
    return EffectiveInput(effective, physical.context)


# --------------------------------------------------------------------------- rectangular junction


JUNCTION_SHEET = ((0, 0), (10, 0), (10, 2), (0, 2))
JUNCTION_LEFT = ((-2, 0), (3, 0), (3, 1), (2, 1), (2, 2), (-2, 2))
JUNCTION_RIGHT = ((7, 0), (12, 0), (12, 2), (8, 2), (8, 1), (7, 1))
JUNCTION_BUMP_LOOP = ((-1.5, 0.5), (0.5, 0.5), (0.5, 1.5), (-1.5, 1.5))
JUNCTION_GRID_UM = 0.5


def rectangular_junction_pair() -> EffectiveInput:
    """Two L-shaped 0.25 um face metals joined by a rectangular junction (the source request of a Josephson element).

    The junction partition replaces the authored sheet by its central rectangle, which touches the
    two terminal-owned end polygons; the thin-film profile makes both terminals Route-A sheets.
    """
    source, stack = rectangular_junction_source()
    effective, _, plan = interpret_physical_geometry(
        source, stack, ApproximationProfile("thin_film", "v1", {})
    )
    return EffectiveInput(effective, plan.context)


def rectangular_junction_source(
    vacuum: str = "upper_air", *, bump: bool = False
) -> tuple[GeometryBuildInput, dict[str, Any]]:
    """The source and stack of :func:`rectangular_junction_pair` before physical interpretation.

    ``vacuum`` names the vacuum region above the metals (``VACUUM_REGION`` for an AEDT model, whose
    Region is that box).  With ``bump`` a 2.75 um tall ``finite_volume`` body of Net ``A`` stands on the
    left film (:data:`JUNCTION_BUMP_LOOP`): a finite body that is not a film.
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
    terminals = (("left", "A", JUNCTION_LEFT), ("right", "B", JUNCTION_RIGHT))
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
        for name, net, loop in terminals
    )
    if bump:
        body = _bump_entity()
        metals = (
            *metals,
            replace(
                body,
                net_id="A",
                host_void_semantic_id=vacuum,
                geometry={**body.geometry, "outer_loop": JUNCTION_BUMP_LOOP},
                metadata={**body.metadata, "source_occurrence_path": "chip"},
            ),
        )
    region = PortSheetRegionRecord(
        "sheet",
        "junction",
        "sheet_polygon",
        JUNCTION_SHEET,
        (),
        (),
        {
            "source": "palace_lumped_port_sheet",
            "source_name": "chip/jj/port",
            "source_occurrence_path": "chip/jj",
            "target_layer": "metal",
            "port_index": 1,
            "direction": (1, 0, 0),
            "direction_raw": (1, 0, 0),
            "direction_sign_convention": "gdsfactory_port_orientation_outward",
        },
    )
    substrate = _solution_region(
        "substrate", "dielectric", "Si", (-4, 14, -2, 4, -2, 0)
    )
    upper = _solution_region(
        vacuum,
        "vacuum",
        "air",
        (-4, 14, -2, 4, -2 if vacuum == "VACUUM_REGION" else 0, 4.25),
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
        (
            *(LayoutPolygonSpec(name, "metal", loop) for name, _, loop in terminals),
            *((LayoutPolygonSpec("bump", "bump", JUNCTION_BUMP_LOOP),) if bump else ()),
        ),
        (*metals, substrate, upper),
        solution_regions=regions,
        metadata={
            "geometry_grid": GeometryGridContext(JUNCTION_GRID_UM).to_payload(),
            "source_dbu_um": JUNCTION_GRID_UM,
        },
        port_sheet_regions=(region,),
    )
    request = RectangularJunctionSpec(
        "junction",
        "sheet",
        "sheet_polygon",
        "chip/jj",
        JunctionTerminalBinding("chip/left", "chip", "A"),
        JunctionTerminalBinding("chip/right", "chip", "B"),
    )
    return with_rectangular_junctions(source, (request,)), stack
