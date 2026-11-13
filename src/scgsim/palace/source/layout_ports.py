"""The geometry request of a Palace simulation: its lumped ports resolved and its geometry input prepared.

``layout_geometry_request`` is everything before the SGB build.  For a problem whose layout sheets are
lumped ports (Eigenmode) it binds each requested port to an authored sheet of the plan, then prepares the
geometry input with exactly the sheets requested; for a problem with no lumped ports (Electrostatic) it
prepares the plan's geometry input with its sheets as authored.  The simulation core builds its model from
it in ``mesh()`` and in ``simulation_drc``, and ``input_summary()`` summarises the same prepared input, so
the summary shows the vacuum, ports and conductors the model is built from.  The request holds no
simulation state: the simulation passes its settings in, and ``mesh()`` keeps the materials the
preparation returns.  A port's circuit is carried through unread; the geometry reads only the sheet, its
target layer and its direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.refusals import Declared
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.refusal import found_text

from .ports import LayoutPortBinding, RequestedPort
from .workflow import GeometrySettings, prepare_mesh_input

_WHERE_ADD_PORT = "the call to add_port"

_WHERE_PLAN_PORTS = "the GeometryPlan's port sheets"


@dataclass(frozen=True)
class GeometryBuildRequest:
    """What the SGB geometry build of one run reads (``mesh.build.build_route_geometry``), as prepared here.

    ``stack`` is the prepared stack and ``gds_bytes`` the plan's GDS it is paired with; ``geometry_input``
    is the prepared geometry input and ``representation`` its representation context.  ``route`` is the
    Surface-EPR representation and ``surface_epr_specs`` the normalized specs as the build reads them, a
    mapping by interface type, None when the simulation set none.
    """

    stack: Mapping[str, Any]
    route: str
    gds_bytes: bytes
    geometry_input: GeometryBuildInput
    representation: RepresentationContext
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None


@dataclass(frozen=True)
class LayoutGeometryRequest:
    """The resolved ports, the request of the SGB geometry build, and the materials the preparation found.

    ``mesh()`` and the modeling check both build from ``source``, and ``input_summary()`` summarises its
    prepared ``geometry_input`` and ``stack``; ``materials`` is None when the preparation left the
    simulation's materials as they were.
    """

    ports: list[LayoutPortBinding]
    source: GeometryBuildRequest
    materials: dict[str, Mapping[str, Any]] | None


def layout_geometry_request(
    *,
    plan: GeometryPlanSnapshot,
    ports: list[RequestedPort] | None,
    settings: GeometrySettings,
) -> LayoutGeometryRequest:
    """Resolve ``ports`` to the plan's authored sheets and prepare the plan's own geometry input and stack.

    ``ports`` is None for a problem with no lumped ports: no sheet is selected or cut as a port.
    """
    resolved, source_records = (
        ([], None) if ports is None else _resolve_plan_ports(plan, ports)
    )
    source_stack = plan.stack
    if "VACUUM_REGION" in source_stack["solution_regions"]:
        settings = replace(settings, vacuum_region=None)  # the plan's own vacuum
    prepared = prepare_mesh_input(
        stack=source_stack,
        build_input=plan.geometry_input,
        settings=settings,
        port_sheet_source_layers=source_records,
    )
    specs = settings.surface_epr_specs
    return LayoutGeometryRequest(
        resolved,
        GeometryBuildRequest(
            stack=prepared.stack,
            route=settings.route,
            gds_bytes=plan.gds_bytes,
            geometry_input=prepared.geometry_input,
            representation=prepared.physical_plan.context,
            surface_epr_specs=None if specs is None else specs.to_payload(),
        ),
        prepared.materials,
    )


def _resolve_plan_ports(
    plan: GeometryPlanSnapshot, ports: list[RequestedPort]
) -> tuple[list[LayoutPortBinding], list[dict[str, Any]]]:
    """Bind each requested port, in request order, to the plan's authored sheet of its name and target layer.

    The request order is the Palace index order: the n-th port requested is ``P<n>``.
    """
    available_records = plan.stack["metadata"].get("port_sheet_source_layers", ())
    available = {record["name"]: record for record in available_records}
    if len(available) != len(available_records):
        raise ValueError(
            Declared(
                "GeometryPlan port names must be unique", _WHERE_PLAN_PORTS
            ).refusal(
                found_text(
                    sorted(
                        name
                        for name in available
                        if [r["name"] for r in available_records].count(name) > 1
                    )
                )
            )
        )
    resolved: list[LayoutPortBinding] = []
    source_records: list[dict[str, Any]] = []
    for requested in ports:
        record = available.get(requested.name)
        if record is None or record["target_layer"] != requested.layer:
            raise ValueError(
                Declared("GeometryPlan", _WHERE_ADD_PORT).refusal(
                    f"has no matching authored port {requested.name!r}"
                )
            )
        resolved.append(
            LayoutPortBinding(
                index=len(resolved) + 1,
                name=requested.name,
                target_layer=requested.layer,
                source_layer=f"{record['layer']}/{record['datatype']}",
                direction=tuple(record["direction"]),
                circuit=requested.circuit,
            )
        )
        source_records.append({**record, "port_index": len(resolved)})
    return resolved, source_records


__all__ = [
    "GeometryBuildRequest",
    "LayoutGeometryRequest",
    "layout_geometry_request",
]
