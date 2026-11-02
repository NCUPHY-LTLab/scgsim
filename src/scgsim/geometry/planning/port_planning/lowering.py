"""Lower authored port regions in source order with their exact host provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, cast

from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import MMContactRecord, SurfacePlanRecord
from scgsim.geometry.planning.port_planning.geometry import (
    port_sheet_boundary_shares_segment,
    port_sheet_direction,
    port_sheet_raw_direction,
    port_sheet_remainder_geometry,
    port_sheet_string,
    port_sheet_target_layer_matches,
    safe_port_sheet_name,
)
from scgsim.geometry.planning.port_planning.route_a import (
    carve_route_a_port_sheet_from_host_plane,
    route_a_port_sheet_sheet_contract,
    route_a_port_sheet_vacuum_volume_id,
)
from scgsim.geometry.planning.port_planning.route_b import (
    partition_route_b_port_sheet_sidewalls,
)
from scgsim.geometry.planning.representation import planned_conductor_representation
from scgsim.geometry.primitives.entities import component_metadata, unique_ids
from scgsim.geometry.primitives.entity_validation import is_solution_entity
from scgsim.geometry.primitives.spatial import same_coordinate_z


def lower_port_sheet_regions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    mm_contacts: tuple[MMContactRecord, ...],
    planned_surfaces: Sequence[SurfacePlanRecord] = (),
    representation_context: RepresentationContext | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Lower the one accepted Palace port-sheet shape before canonical topology."""
    return _PortSheetLowerer(
        build_input, route, mm_contacts, planned_surfaces, representation_context
    ).build()


@dataclass
class _PortSheetLowerer:
    """Carry source indexes and the current port through ordered lowering stages."""

    build_input: GeometryBuildInput
    route: RouteLiteral
    mm_contacts: tuple[MMContactRecord, ...]
    planned_surfaces: Sequence[SurfacePlanRecord]
    representation_context: RepresentationContext | None
    entities: dict[str, SemanticEntitySpec] = field(init=False)
    polygons: dict[str, LayoutPolygonSpec] = field(init=False)
    component_by_entity: dict[str, str] = field(init=False)
    records: list[SurfacePlanRecord] = field(init=False)
    region: PortSheetRegionRecord = field(init=False)
    metadata: Mapping[str, Any] = field(init=False)
    source_name: str = field(init=False)
    target_layer: str = field(init=False)
    port_index: int = field(init=False)
    direction: tuple[float, float, float] = field(init=False)
    raw_direction: tuple[float, float, float] = field(init=False)
    sign_convention: str = field(init=False)
    host_ids: tuple[str, ...] = field(init=False)
    hosts: tuple[SemanticEntitySpec | None, ...] = field(init=False)
    host_entities: tuple[SemanticEntitySpec, ...] = field(init=False)
    z_values: tuple[float, ...] = field(init=False)
    thickness_values: tuple[float, ...] = field(init=False)
    host_void_ids: tuple[str | None, ...] = field(init=False)
    rectangle_id: str | None = field(init=False)
    partition: Mapping[str, Any] = field(init=False)
    remainder: Mapping[str, Any] = field(init=False)
    contact_loops: tuple[tuple[tuple[float, float], ...], ...] = field(init=False)
    host_polygon_ids: tuple[str, ...] = field(init=False)
    owner_provenance: tuple[dict[str, Any], ...] = field(init=False)
    source_provenance: dict[str, Any] = field(init=False)
    plane_z_um: float = field(init=False)
    boundary_volume_ids: tuple[str, ...] = field(init=False)
    embedded_volume_id: str = field(init=False)
    port_surface: SurfacePlanRecord = field(init=False)

    def build(self) -> tuple[SurfacePlanRecord, ...]:
        """Lower each authored port after building the shared source indexes."""
        self.initialize()
        for region in self.build_input.port_sheet_regions:
            self.region = region
            self.read_declaration()
            self.read_hosts()
            self.read_geometry()
            self.read_provenance()
            self.resolve_embedding()
            self.make_surface()
            self.attach_surface()
        return tuple(self.records)

    def initialize(self) -> None:
        """Index source entities, polygons and components before reading any port."""
        self.entities = {
            entity.semantic_id: entity for entity in self.build_input.entities
        }
        self.polygons = {
            polygon.polygon_id: polygon for polygon in self.build_input.polygons
        }
        self.component_by_entity = {
            entity_id: str(component["conductor_component_id"])
            for component in component_metadata(self.mm_contacts)
            for entity_id in component["members"]
        }
        self.records: list[SurfacePlanRecord] = list(self.planned_surfaces)

    def read_declaration(self) -> None:
        """Read the authored source identity and normalized/raw directions."""
        self.metadata = self.region.metadata
        self.source_name = port_sheet_string(self.metadata, "source_name", self.region)
        self.target_layer = port_sheet_string(
            self.metadata, "target_layer", self.region
        )
        self.port_index = cast(int, self.metadata.get("port_index"))
        if (
            isinstance(self.port_index, bool)
            or not isinstance(self.port_index, int)
            or self.port_index < 1
        ):
            raise ValueError(f"{self.region.port_sheet_id} requires 1-based port_index")
        self.direction = port_sheet_direction(self.region)
        self.raw_direction = port_sheet_raw_direction(self.region)
        self.sign_convention = port_sheet_string(
            self.metadata, "direction_sign_convention", self.region
        )

    def read_hosts(self) -> None:
        """Validate the exact two source hosts and their common physical plane."""
        if len(self.region.overlaps) != 2:
            raise ValueError(
                f"{self.region.port_sheet_id} requires exactly two host overlap records"
            )
        self.host_ids = unique_ids(
            (overlap.host_semantic_id for overlap in self.region.overlaps)
        )
        if len(self.host_ids) != 2:
            raise ValueError(
                f"{self.region.port_sheet_id} requires one overlap per distinct face_metal host"
            )
        self.hosts = tuple((self.entities.get(host_id) for host_id in self.host_ids))
        if any(
            (host is None for host in cast(tuple[SemanticEntitySpec, ...], self.hosts))
        ):
            raise ValueError(
                f"{self.region.port_sheet_id} references an unknown port host"
            )
        self.host_entities = tuple(
            (
                host
                for host in cast(tuple[SemanticEntitySpec, ...], self.hosts)
                if host is not None
            )
        )
        if any(
            (
                host.part_role != "face_metal"
                or not port_sheet_target_layer_matches(host, self.target_layer)
                or planned_conductor_representation(
                    host, route=self.route, context=self.representation_context
                )
                != {"A": "surface_sheet", "B": "cutout_boundary_shell"}[self.route]
                for host in self.host_entities
            )
        ):
            raise ValueError(
                f"{self.region.port_sheet_id} hosts must be active A/B face_metal on {self.target_layer}"
            )
        self.z_values = tuple(
            (
                float(host.geometry.get("z_um", float("nan")))
                for host in self.host_entities
            )
        )
        if not all(
            (isfinite(value) for value in self.z_values)
        ) or not same_coordinate_z(self.z_values[0], self.z_values[1]):
            raise ValueError(
                f"{self.region.port_sheet_id} hosts require the same finite z_um"
            )
        self.thickness_values: tuple[float, ...] = ()
        if self.route == "B":
            self.thickness_values = tuple(
                (
                    float(host.geometry.get("thickness_um", float("nan")))
                    for host in self.host_entities
                )
            )
            if (
                not all((isfinite(value) for value in self.thickness_values))
                or any((value <= 0.0 for value in self.thickness_values))
                or (
                    not same_coordinate_z(
                        self.thickness_values[0], self.thickness_values[1]
                    )
                )
            ):
                raise ValueError(
                    f"{self.region.port_sheet_id} Route B hosts require the same finite positive thickness_um"
                )
        self.host_void_ids = tuple(
            (host.host_void_semantic_id for host in self.host_entities)
        )
        if self.route == "B" and (
            any((not host_void_id for host_void_id in self.host_void_ids))
            or len(set(self.host_void_ids)) != 1
            or self.host_void_ids[0] not in self.entities
            or (not is_solution_entity(self.entities[str(self.host_void_ids[0])]))
        ):
            raise ValueError(
                f"{self.region.port_sheet_id} hosts require one common host_void solution"
            )
        if any(
            (
                overlap.host_polygon_id not in self.polygons
                for overlap in self.region.overlaps
            )
        ):
            raise ValueError(
                f"{self.region.port_sheet_id} references an unknown host polygon"
            )

    def read_geometry(self) -> None:
        """Resolve the active remainder and each finite host contact."""
        self.rectangle_id = self.region.metadata.get("rectangular_junction_partition")
        if self.rectangle_id is None:
            self.remainder = port_sheet_remainder_geometry(self.region)
            self.contact_loops = tuple(
                (overlap.overlap_loop for overlap in self.region.overlaps)
            )
        else:
            self.partition = self.build_input.metadata["junction_partitions"][
                self.rectangle_id
            ]
            central = next(
                (
                    polygon
                    for polygon in self.build_input.polygons
                    if polygon.polygon_id == self.partition["central_polygon_id"]
                )
            )
            if (
                self.region.exterior != central.exterior
                or self.region.holes != central.holes
            ):
                raise ValueError(
                    "rectangular port differs from its derived central rectangle"
                )
            self.remainder = {
                "outer_loop": central.exterior,
                "hole_loops": central.holes,
            }
            self.contact_loops = tuple(
                (
                    loop
                    for host in cast(tuple[SemanticEntitySpec, ...], self.hosts)
                    for loop in (
                        host.geometry["outer_loop"],
                        *host.geometry.get("hole_loops", ()),
                    )
                )
            )
        if self.rectangle_id is None:
            contacting = all(
                (
                    port_sheet_boundary_shares_segment(self.remainder, loop)
                    for loop in self.contact_loops
                )
            )
        else:
            contacting = all(
                (
                    any(
                        (
                            port_sheet_boundary_shares_segment(self.remainder, loop)
                            for loop in (
                                host.geometry["outer_loop"],
                                *host.geometry.get("hole_loops", ()),
                            )
                        )
                    )
                    for host in cast(tuple[SemanticEntitySpec, ...], self.hosts)
                )
            )
        if not contacting:
            raise ValueError(
                f"{self.region.port_sheet_id} active sheet must share a finite segment with each host"
            )

    def read_provenance(self) -> None:
        """Detach original host, polygon and overlap provenance in source order."""
        self.host_polygon_ids = tuple(
            (overlap.host_polygon_id for overlap in self.region.overlaps)
        )
        self.owner_provenance = tuple(
            (
                {
                    "semantic_id": host.semantic_id,
                    "net_id": host.net_id,
                    "equipotential_id": host.metadata.get("equipotential_id"),
                    "conductor_component_id": self.component_by_entity.get(
                        host.semantic_id, f"COMP__{host.semantic_id}"
                    ),
                }
                for host in self.host_entities
            )
        )
        self.source_provenance = {
            "source_name": self.source_name,
            "port_index": self.port_index,
            "source_layer": self.region.source_layer,
            "source_polygon_id": self.region.source_polygon_id,
            "overlap_ids": tuple(
                (overlap.overlap_id for overlap in self.region.overlaps)
            ),
            "overlaps": tuple(
                (
                    {
                        "overlap_id": overlap.overlap_id,
                        "host_semantic_id": overlap.host_semantic_id,
                        "host_polygon_id": overlap.host_polygon_id,
                        "overlap_loop": overlap.overlap_loop,
                    }
                    for overlap in self.region.overlaps
                )
            ),
            "host_polygon_ids": self.host_polygon_ids,
            "route": self.route,
            "target_layer": self.target_layer,
            "direction": self.direction,
            "direction_raw": self.raw_direction,
            "direction_sign_convention": self.sign_convention,
        }

    def resolve_embedding(self) -> None:
        """Bind the route-specific sheet plane, volume owners and junction provenance."""
        if self.route == "A":
            self.plane_z_um, self.boundary_volume_ids = (
                route_a_port_sheet_sheet_contract(
                    self.host_ids, self.planned_surfaces, self.region.overlaps
                )
            )
            self.embedded_volume_id = route_a_port_sheet_vacuum_volume_id(
                self.boundary_volume_ids, self.entities
            )
        else:
            self.plane_z_um = self.z_values[0] + self.thickness_values[0] / 2.0
            self.boundary_volume_ids = (cast(str, self.host_void_ids[0]),)
            self.embedded_volume_id = str(self.host_void_ids[0])
        if self.rectangle_id is not None:
            self.source_provenance = {
                **self.source_provenance,
                "rectangular_junction_binding": self.region.metadata[
                    "rectangular_junction_binding"
                ],
                "junction_partition": self.partition,
            }

    def make_surface(self) -> None:
        """Build one source-bound port sheet surface record."""
        self.port_surface = SurfacePlanRecord(
            surface_id=f"SURF__LUMPED_PORT__{safe_port_sheet_name(self.source_name)}",
            owner_semantic_id=self.host_ids[0],
            surface_role="lumped_port",
            geometry_ref={
                "outer_loop": self.remainder["outer_loop"],
                "hole_loops": self.remainder["hole_loops"],
                "plane": {"axis": "z", "value_um": self.plane_z_um},
                "representation": "lumped_port_sheet",
            },
            valid_routes=(self.route,),
            metadata={
                "route": self.route,
                "representation": "lumped_port_sheet",
                "interface_type": "lumped_port",
                "face_kind": "sheet",
                "owner_semantic_ids": self.host_ids,
                "physical_owner_semantic_ids": self.host_ids,
                "boundary_volume_ids": self.boundary_volume_ids,
                "route_a_boundary_port": self.route == "A",
                "embedded_surface": True,
                "embedded_volume_id": self.embedded_volume_id,
                "physical_name": f"LUMPED_PORT__{safe_port_sheet_name(self.source_name)}",
                "source_provenance": self.source_provenance,
                "physical_attribute": {
                    "port_index": self.port_index,
                    "port_name": self.source_name,
                    "source_layer": self.region.source_layer,
                    "target_layer": self.target_layer,
                    "embedded_volume_id": self.embedded_volume_id,
                    "direction": self.direction,
                    "owner_semantic_ids": self.host_ids,
                    "owner_provenance": self.owner_provenance,
                },
            },
        )

    def attach_surface(self) -> None:
        """Partition the host surfaces and append the port in source order."""
        if self.route == "A":
            self.records = carve_route_a_port_sheet_from_host_plane(
                self.records, port_surface=self.port_surface
            )
        elif self.records:
            self.records = partition_route_b_port_sheet_sidewalls(
                self.records,
                port_surface=self.port_surface,
                remainder=self.remainder,
                overlaps_by_host={
                    overlap.host_semantic_id: overlap
                    for overlap in self.region.overlaps
                },
            )
        self.records.append(self.port_surface)
