"""The geometry request of a Palace simulation with lumped ports: its ports resolved and its geometry input prepared.

``layout_geometry_request`` is everything before the SGB build of a simulation whose layout sheets are
lumped ports: it binds each requested port to an authored sheet of the plan, then
prepares the geometry input with exactly the sheets requested.  ``EigenmodeSim`` builds its model from
it in ``mesh()`` and in ``simulation_drc``, and ``input_summary()`` summarises the same prepared input, so
the summary shows the vacuum, ports and conductors the model is built from.  The request holds no
simulation state: the simulation passes its settings in, and ``mesh()`` keeps the materials the
preparation returns.  A port's circuit is carried through unread; the geometry reads only the sheet, its
target layer and its direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.input import VacuumRegionSpec
from scgsim.geometry.models.physical import ApproximationProfile
from scgsim.geometry.source.plan import GeometryPlanSnapshot

from .config import LayoutPortBinding
from .ports import RequestedPort
from .transforms import RouteAThinFilm
from .workflow import prepare_mesh_input


@dataclass(frozen=True)
class GeometrySettings:
    """The simulation settings the geometry preparation reads, as the simulation holds them."""

    route: str
    route_a_thin_film: RouteAThinFilm | None
    approximation_profile: ApproximationProfile | None
    vacuum_region: VacuumRegionSpec | None
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None


@dataclass(frozen=True)
class LayoutGeometryRequest:
    """The resolved ports, the keywords of the SGB geometry build, and the materials the preparation found.

    ``mesh()`` and the modeling check both build from ``source``, and ``input_summary()`` summarises its
    prepared ``source_geometry_input`` and ``stack``; ``materials`` is None when the preparation left the
    simulation's materials as they were.
    """

    ports: list[LayoutPortBinding]
    source: dict[str, Any]
    materials: dict[str, Mapping[str, Any]] | None


def layout_geometry_request(
    *,
    plan: GeometryPlanSnapshot,
    ports: list[RequestedPort],
    settings: GeometrySettings,
) -> LayoutGeometryRequest:
    """Resolve ``ports`` to the plan's authored sheets and prepare the plan's own geometry input and stack."""
    resolved, source_records = _resolve_plan_ports(plan, ports)
    source_stack = plan.stack
    prepared = prepare_mesh_input(
        stack=source_stack,
        route=settings.route,
        route_a_thin_film=settings.route_a_thin_film,
        approximation_profile=settings.approximation_profile,
        vacuum_region=(
            None
            if "VACUUM_REGION" in source_stack["solution_regions"]
            else settings.vacuum_region
        ),
        build_input=plan.geometry_input,
        port_sheet_source_layers=source_records,
    )
    return LayoutGeometryRequest(
        resolved,
        {
            "stack": prepared.stack,
            "route": settings.route,
            "surface_epr_specs": settings.surface_epr_specs,
            "source_gds_bytes": plan.gds_bytes,
            "source_geometry_input": prepared.geometry_input,
            "representation_context": prepared.physical_plan.context,
        },
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
        raise ValueError("GeometryPlan port names must be unique")
    resolved: list[LayoutPortBinding] = []
    source_records: list[dict[str, Any]] = []
    for requested in ports:
        record = available.get(requested.name)
        if record is None or record["target_layer"] != requested.layer:
            raise ValueError(
                f"GeometryPlan has no matching authored port {requested.name!r}"
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
    "GeometrySettings",
    "LayoutGeometryRequest",
    "layout_geometry_request",
]
