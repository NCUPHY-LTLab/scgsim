"""Resolve same-net conductor components and their contact provenance."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from scgsim.geometry.grid.planar_loops import clean_loop, polygon_area
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.models.topology import InterfacePlanRecord, MMContactRecord
from scgsim.geometry.planning.contact_planning.geometry import source_layer_provenance
from scgsim.geometry.planning.contact_planning.overlap import (
    validate_volumetric_conductor_contacts,
)
from scgsim.geometry.planning.domain_planning.occupancy import (
    route_conductor_contact_footprints,
)


def plan_mm_contact_records(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    representation_context: RepresentationContext | None = None,
) -> tuple[tuple[InterfacePlanRecord, ...], tuple[MMContactRecord, ...]]:
    """Turn finite-area same-net contact into hidden component provenance.

    Gmsh/OCC must never infer this graph from coincident bodies.  The contact
    record remains present for both Route A and Route B while its internal face
    is deliberately absent from solver physical groups.
    """
    enriched, records, _ = plan_mm_contact_records_with_components(
        build_input,
        route=route,
        interfaces=interfaces,
        representation_context=representation_context,
    )
    return (enriched, records)


def plan_mm_contact_records_with_components(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
    representation_context: RepresentationContext | None = None,
) -> tuple[
    tuple[InterfacePlanRecord, ...],
    tuple[MMContactRecord, ...],
    tuple[dict[str, Any], ...],
]:
    """Return MM contacts and their single source-owned component ledger."""
    return _ConductorMembershipPlanner(
        build_input, route, interfaces, representation_context
    ).build()


@dataclass
class _ConductorMembershipPlanner:
    """Carry live membership, union state and records through ordered planning."""

    build_input: GeometryBuildInput
    route: RouteLiteral | None
    interfaces: tuple[InterfacePlanRecord, ...]
    representation_context: RepresentationContext | None
    entities: dict[str, SemanticEntitySpec] = field(init=False)
    contacts: list[
        tuple[InterfacePlanRecord, SemanticEntitySpec, SemanticEntitySpec]
    ] = field(init=False)
    normalizations: tuple[
        tuple[
            SemanticEntitySpec,
            SemanticEntitySpec,
            tuple[tuple[tuple[float, float], ...], ...],
        ],
        ...,
    ] = field(init=False)
    parent: dict[str, str] = field(init=False)
    component_ids: dict[str, str] = field(init=False)
    component_values: dict[str, tuple[str | None, str | None]] = field(init=False)
    records: list[MMContactRecord] = field(init=False)
    enriched: list[InterfacePlanRecord] = field(init=False)

    def build(self):
        """Resolve contact membership and emit its source-owned records in order."""
        self.collect_contacts()
        self.normalizations = validate_volumetric_conductor_contacts(
            self.build_input,
            route=self.route,
            entities=self.entities,
            representation_context=self.representation_context,
        )
        self.join_components()
        self.resolve_component_values()
        self.emit_contacts()
        self.emit_normalizations()
        return self.assemble()

    def collect_contacts(self) -> None:
        """Collect live conductor contacts and validate their resolved nets."""
        live_ids = {
            entity.semantic_id
            for entity in route_conductor_contact_footprints(
                self.build_input,
                self.route,
                representation_context=self.representation_context,
            )
        }
        # Membership retains each original Entity and all its source fragments,
        # rather than the temporary footprint processed last.
        self.entities = {
            entity.semantic_id: entity
            for entity in self.build_input.entities
            if entity.semantic_id in live_ids
        }
        self.contacts: list[
            tuple[InterfacePlanRecord, SemanticEntitySpec, SemanticEntitySpec]
        ] = []
        for interface in self.interfaces:
            if interface.recognition_rule != "coplanar_conductor_contact_patch":
                continue
            lower_id, upper_id = interface.owner_semantic_ids
            lower = self.entities.get(lower_id)
            upper = self.entities.get(upper_id)
            if lower is None or upper is None:
                raise ValueError(
                    f"{interface.interface_id} references non-live conductors"
                )
            if not lower.net_id or not upper.net_id:
                raise ValueError(
                    f"{interface.interface_id} contact requires resolved nets"
                )
            if lower.net_id != upper.net_id:
                raise ValueError(
                    f"{interface.interface_id} shorts different nets {lower.net_id!r} and {upper.net_id!r}"
                )
            self.contacts.append((interface, lower, upper))

    def join_components(self) -> None:
        """Join contact and normalized-overlap members in source order."""
        self.parent = {entity_id: entity_id for entity_id in self.entities}

        def find(entity_id: str) -> str:
            while self.parent[entity_id] != entity_id:
                self.parent[entity_id] = self.parent[self.parent[entity_id]]
                entity_id = self.parent[entity_id]
            return entity_id

        def union(first: str, second: str) -> None:
            first_root, second_root = (find(first), find(second))
            if first_root != second_root:
                self.parent[max(first_root, second_root)] = min(first_root, second_root)

        for _, lower, upper in self.contacts:
            union(lower.semantic_id, upper.semantic_id)
        for lower, upper, _ in self.normalizations:
            union(lower.semantic_id, upper.semantic_id)
        self.component_ids = {
            entity_id: f"COMP__{find(entity_id)}" for entity_id in self.entities
        }

    def resolve_component_values(self) -> None:
        """Resolve each component's single net and equipotential identity."""
        self.component_values: dict[str, tuple[str | None, str | None]] = {}
        for component_id in set(self.component_ids.values()):
            members = [
                entity
                for entity_id, entity in self.entities.items()
                if self.component_ids[entity_id] == component_id
            ]
            nets = {entity.net_id for entity in members if entity.net_id}
            equipotentials = {
                str(entity.metadata["equipotential_id"])
                for entity in members
                if entity.metadata.get("equipotential_id") is not None
            }
            if len(nets) > 1 or len(equipotentials) > 1:
                raise ValueError(
                    f"{component_id} has conflicting net/equipotential ids"
                )
            self.component_values[component_id] = (
                next(iter(nets), None),
                next(iter(equipotentials), None),
            )

    def emit_contacts(self) -> None:
        """Emit ordered contact records and enrich their interface metadata."""
        self.records: list[MMContactRecord] = []
        self.enriched: list[InterfacePlanRecord] = []
        for interface in self.interfaces:
            if interface.recognition_rule != "coplanar_conductor_contact_patch":
                self.enriched.append(interface)
                continue
            lower_id, upper_id = interface.owner_semantic_ids
            lower, upper = (self.entities[lower_id], self.entities[upper_id])
            loop = clean_loop(interface.metadata["outer_loop"])
            component_id = self.component_ids[lower_id]
            component_net, component_equipotential = self.component_values[component_id]
            contact_id = str(
                interface.metadata.get("contact_id") or interface.interface_id
            )
            lower_face = str(interface.metadata.get("lower_face", "top"))
            upper_face = str(interface.metadata.get("upper_face", "bottom"))
            self.records.append(
                MMContactRecord(
                    contact_id=contact_id,
                    lower_entity_id=lower_id,
                    upper_entity_id=upper_id,
                    lower_source_face_id=f"{lower_id}__{lower_face}",
                    upper_source_face_id=f"{upper_id}__{upper_face}",
                    lower_source_fragment_ids=lower.polygon_ids,
                    upper_source_fragment_ids=upper.polygon_ids,
                    outer_loop=loop,
                    area_um2=abs(polygon_area(loop)),
                    normal=(0.0, 0.0, 1.0),
                    conductor_component_id=component_id,
                    net_id=component_net,
                    equipotential_id=component_equipotential,
                    layer_provenance={
                        "lower": source_layer_provenance(lower),
                        "upper": source_layer_provenance(upper),
                    },
                    material_provenance={
                        "lower": lower.material_id,
                        "upper": upper.material_id,
                    },
                    source_provenance={
                        "interface_id": interface.interface_id,
                        "route": self.route,
                        "recognition_rule": interface.recognition_rule,
                        "source_polygon_ids": interface.source_polygon_ids,
                    },
                )
            )
            self.enriched.append(
                replace(
                    interface,
                    metadata={
                        **dict(interface.metadata),
                        "conductor_component_id": component_id,
                        "net_id": component_net,
                        "equipotential_id": component_equipotential,
                        "hidden_solver_contact": self.route in {"A", "B"}
                        or (
                            self.route is None
                            and self.representation_context is not None
                        ),
                    },
                )
            )

    def emit_normalizations(self) -> None:
        """Append source-normalized volumetric contact records."""
        for lower, upper, loops in self.normalizations:
            for index, loop in enumerate(loops):
                self.records.append(
                    MMContactRecord(
                        contact_id=f"NORMALIZED__{lower.semantic_id}__{upper.semantic_id}__{index:04d}",
                        lower_entity_id=lower.semantic_id,
                        upper_entity_id=upper.semantic_id,
                        lower_source_face_id=f"{lower.semantic_id}__normalized_overlap",
                        upper_source_face_id=f"{upper.semantic_id}__normalized_overlap",
                        lower_source_fragment_ids=lower.polygon_ids,
                        upper_source_fragment_ids=upper.polygon_ids,
                        outer_loop=loop,
                        area_um2=abs(polygon_area(loop)),
                        normal=(0.0, 0.0, 1.0),
                        conductor_component_id=self.component_ids[lower.semantic_id],
                        net_id=self.component_values[
                            self.component_ids[lower.semantic_id]
                        ][0],
                        equipotential_id=self.component_values[
                            self.component_ids[lower.semantic_id]
                        ][1],
                        layer_provenance={
                            "lower": source_layer_provenance(lower),
                            "upper": source_layer_provenance(upper),
                        },
                        material_provenance={
                            "lower": lower.material_id,
                            "upper": upper.material_id,
                        },
                        source_provenance={
                            "route": self.route,
                            "source_polygon_ids": (
                                *lower.polygon_ids,
                                *upper.polygon_ids,
                            ),
                        },
                    )
                )

    def assemble(
        self,
    ) -> tuple[
        tuple[InterfacePlanRecord, ...],
        tuple[MMContactRecord, ...],
        tuple[dict[str, Any], ...],
    ]:
        """Return enriched interfaces, contacts and the ordered component ledger."""
        records_tuple = tuple(self.records)
        component_ledger = tuple(
            (
                {
                    "conductor_component_id": component_id,
                    "members": tuple(
                        sorted(
                            (
                                entity_id
                                for entity_id, owner_component_id in self.component_ids.items()
                                if owner_component_id == component_id
                            )
                        )
                    ),
                    "contact_ids": tuple(
                        sorted(
                            (
                                record.contact_id
                                for record in records_tuple
                                if record.conductor_component_id == component_id
                            )
                        )
                    ),
                    "net_id": self.component_values[component_id][0],
                    "equipotential_id": self.component_values[component_id][1],
                }
                for component_id in sorted(self.component_values)
            )
        )
        return (tuple(self.enriched), records_tuple, component_ledger)
