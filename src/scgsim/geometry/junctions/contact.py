"""Resolve authored support planes and the conductors incident to a junction, in the source and in the effective input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction
from typing import Any

from scgsim.geometry.grid.grid_regions import grid_change_message
from scgsim.geometry.junctions.core import (
    SUPPORT_PLANE_SCHEMA,
    canonical_sha256,
    plain_record,
)
from scgsim.geometry.junctions.exact_geometry import closed_contact
from scgsim.geometry.models.grid import format_box, geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput, LayoutPolygonSpec
from scgsim.geometry.models.junctions import junction_net_id
from scgsim.geometry.models.physical import (
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.semantics.route_a import geometry_z_range


@dataclass(frozen=True)
class JunctionContactRequest:
    """Carry source sheet identity, terminal nets and optional support binding."""

    source_polygon_id: str
    terminal_a_net: str
    terminal_b_net: str
    stored_binding: Mapping[str, Any] | None = None
    _: KW_ONLY
    support_schema: str = SUPPORT_PLANE_SCHEMA


def junction_contact_inventory(
    build_input: GeometryBuildInput, request: JunctionContactRequest
) -> dict[str, Any]:
    """Resolve one authored support and its source/effective incident conductors."""
    return _ContactInventory(build_input, request).build()


@dataclass
class _ContactInventory:
    """Carry source support facts through ordered contact and projection stages."""

    build_input: GeometryBuildInput
    request: JunctionContactRequest
    regions: list[PortSheetRegionRecord] = field(init=False)
    polygons: dict[str, LayoutPolygonSpec] = field(init=False)
    region: PortSheetRegionRecord | None = field(init=False)
    sheet_rings: tuple = field(init=False)
    plan: PhysicalRepresentationPlan | None = field(init=False)
    facts: dict[str, dict[str, Any]] = field(init=False)
    hosts: list[dict[str, Any]] = field(init=False)
    source_z: float = field(init=False)
    basis: str = field(init=False)
    binding: dict[str, Any] = field(init=False)
    effective_z: float = field(init=False)

    def build(self) -> dict[str, Any]:
        """Bind authored contacts before projecting the effective support plane."""
        self.initialize()
        self.collect_facts()
        self.hosts = []
        if self.region is not None and self.region.overlaps:
            self.from_port_hosts(self.region)
        else:
            self.from_terminal_contacts()
        self.freeze_binding()
        self.project_effective_plane()
        return self.assemble()

    def initialize(self):
        """Resolve the unique source sheet and its physical representation plan."""
        self.regions = [
            region
            for region in self.build_input.port_sheet_regions
            if region.source_polygon_id == self.request.source_polygon_id
        ]
        if len(self.regions) > 1:
            raise ValueError("junction source sheet has ambiguous port-region identity")
        self.polygons = {
            polygon.polygon_id: polygon for polygon in self.build_input.polygons
        }
        self.region = self.regions[0] if self.regions else None
        sheet = (
            self.region
            if self.region is not None
            else self.polygons[self.request.source_polygon_id]
        )
        self.sheet_rings = _source_contact_rings(sheet)
        raw_plan = self.build_input.metadata.get("physical_representation")
        self.plan = (
            PhysicalRepresentationPlan.from_payload(raw_plan)
            if raw_plan is not None
            else None
        )

    def collect_facts(self):
        """Collect each conductor's source/effective intervals and authored anchor."""
        self.facts = {}
        for entity in self.build_input.entities:
            if entity.material_kind != "conductor":
                continue
            projection = entity.metadata.get("physical_representation")
            if projection is not None:
                if self.plan is None:
                    raise ValueError(
                        "junction lowered source requires its common physical plan"
                    )
                source_interval = tuple(projection["source_interval_um"])
                physical_id = projection["physical_layer_id"]
                declaration = (
                    self.plan.declaration_index[physical_id]
                    if physical_id is not None
                    else None
                )
            else:
                source_interval = geometry_z_range(entity.geometry, entity.semantic_id)
                raw = entity.metadata.get("physical_layer")
                declaration = (
                    PhysicalLayerDeclaration.from_payload(raw)
                    if raw is not None
                    else None
                )
                physical_id = (
                    declaration.physical_layer_id if declaration is not None else None
                )
            anchor = None
            if declaration is not None and declaration.feature_kind in {
                "stack_film",
                "local_film",
            }:
                authored = declaration.attachment_z_um
                if source_interval[0] <= authored <= source_interval[1]:
                    anchor = authored
            elif source_interval[0] == source_interval[1]:
                anchor = source_interval[0]
            self.facts[entity.semantic_id] = {
                "entity": entity,
                "source_interval_um": source_interval,
                "effective_interval_um": geometry_z_range(
                    entity.geometry, entity.semantic_id
                ),
                "physical_layer_id": physical_id,
                "anchor": anchor,
            }

    def from_port_hosts(self, region: PortSheetRegionRecord):
        """Bind exact port-host incidences to their common authored support plane."""
        for overlap in region.overlaps:
            if (
                overlap.port_sheet_id != region.port_sheet_id
                or overlap.port_polygon_id != self.request.source_polygon_id
            ):
                raise ValueError(
                    "junction port overlap references another source sheet"
                )
            matches = [
                eid
                for eid, fact in self.facts.items()
                if (
                    eid == overlap.host_semantic_id
                    or fact["entity"].metadata.get("source_semantic_id")
                    == overlap.host_semantic_id
                )
                and overlap.host_polygon_id in fact["entity"].polygon_ids
            ]
            if len(matches) != 1 or overlap.host_polygon_id not in self.polygons:
                raise ValueError(
                    "junction port host lacks exact Entity/polygon ownership"
                )
            self.hosts.append(self.host_record(matches[0], overlap.host_polygon_id))
        anchors = {
            self.facts[host["source_entity_id"]]["anchor"]
            for host in self.hosts
            if self.facts[host["source_entity_id"]]["anchor"] is not None
        }
        if len(anchors) != 1:
            raise ValueError(self.no_common_plane(region))
        self.source_z = next(iter(anchors))
        if any(
            (
                not self.contains(
                    self.facts[host["source_entity_id"]]["source_interval_um"],
                    self.source_z,
                )
                for host in self.hosts
            )
        ):
            raise ValueError(
                "junction port host is disjoint from its authored support plane"
            )
        self.basis = "port_overlap_hosts"

    def no_common_plane(self, region: PortSheetRegionRecord) -> str:
        """The refusal of port hosts without one film plane: each host's interval and plane, and the sheet."""
        grid_um = geometry_grid_context(self.build_input).grid_size_um
        hosts = []
        for host in self.hosts:
            fact = self.facts[host["source_entity_id"]]
            low, high = fact["source_interval_um"]
            plane = (
                "no film plane"
                if fact["anchor"] is None
                else f"its film plane at z = {fact['anchor']:g} um"
            )
            hosts.append(
                f"{host['source_entity_id']} (z {low:g}..{high:g} um, {plane})"
            )
        xs, ys = [x for x, _ in region.exterior], [y for _, y in region.exterior]
        box = format_box((min(xs), min(ys), max(xs), max(ys)), grid_um)
        name = region.metadata.get("source_name", region.port_sheet_id)
        return grid_change_message(
            f"junction port hosts lack one common authored support plane: lumped port {name!r}",
            [region.source_layer],
            grid_um,
            [f"the sheet at {box} lands on {', '.join(hosts)}"],
            "Fix the declaration: land the port on conductors of one film level "
            "(stack_film or local_film), or declare its target level as a film.",
        )

    def from_terminal_contacts(self):
        """Bind terminal closed contacts to the unique authored support plane."""
        contacting = []
        for eid, fact in self.facts.items():
            entity = fact["entity"]
            if junction_net_id(entity) not in {
                self.request.terminal_a_net,
                self.request.terminal_b_net,
            }:
                continue
            for pid in entity.polygon_ids:
                if pid != self.request.source_polygon_id and closed_contact(
                    [self.sheet_rings], [_source_contact_rings(self.polygons[pid])]
                ):
                    contacting.append((eid, pid))
        candidates = {
            self.facts[eid]["anchor"]
            for eid, _ in contacting
            if self.facts[eid]["anchor"] is not None
        }
        candidates = {
            z
            for z in candidates
            if all(
                (
                    any(
                        (
                            junction_net_id(self.facts[eid]["entity"]) == net
                            and self.contains(self.facts[eid]["source_interval_um"], z)
                            and (self.facts[eid]["anchor"] in {None, z})
                            for eid, _ in contacting
                        )
                    )
                    for net in (
                        self.request.terminal_a_net,
                        self.request.terminal_b_net,
                    )
                )
            )
        }
        if len(candidates) != 1:
            raise ValueError(
                "junction terminal contacts lack a unique authored support plane"
            )
        self.source_z = next(iter(candidates))
        self.hosts = [
            self.host_record(eid, pid)
            for eid, pid in contacting
            if self.contains(self.facts[eid]["source_interval_um"], self.source_z)
            and self.facts[eid]["anchor"] in {None, self.source_z}
        ]
        self.basis = "terminal_closed_contact"

    def freeze_binding(self):
        """Order source-only hosts and compare any stored support binding."""
        self.hosts = sorted(
            {canonical_sha256(host): host for host in self.hosts}.values(),
            key=lambda item: (item["source_entity_id"], item["source_polygon_id"]),
        )
        self.binding = {
            "schema_version": self.request.support_schema,
            "source_polygon_id": self.request.source_polygon_id,
            "source_z_um": self.source_z,
            "basis": self.basis,
            "source_hosts": self.hosts,
        }
        if (
            self.request.stored_binding is not None
            and plain_record(self.request.stored_binding) != self.binding
        ):
            raise ValueError(
                "junction stored support-plane binding differs from current source"
            )

    def project_effective_plane(self):
        """Project authored anchors and require their effective support incidence."""
        effective_planes = set()
        for host in self.hosts:
            fact = self.facts[host["source_entity_id"]]
            if fact["anchor"] is None:
                continue
            if self.plan is None:
                self.effective_z = self.source_z
            elif fact["physical_layer_id"] is None:
                self.effective_z = self.plan.map_point((0.0, 0.0, self.source_z))[2]
            else:
                self.effective_z = self.plan.project_support_point(
                    (0.0, 0.0, self.source_z), fact["physical_layer_id"]
                )[2]
            effective_planes.add(self.effective_z)
        if len(effective_planes) != 1:
            raise ValueError(
                "junction anchor hosts map to different effective support planes"
            )
        self.effective_z = next(iter(effective_planes))
        if any(
            (
                not self.contains(
                    self.facts[host["source_entity_id"]]["effective_interval_um"],
                    self.effective_z,
                )
                for host in self.hosts
            )
        ):
            raise ValueError(
                "junction host does not meet its mapped effective support plane"
            )

    def assemble(self):
        """Return the source binding and eligible source/effective conductors."""
        eligible = sorted(
            (
                eid
                for eid, fact in self.facts.items()
                if self.contains(fact["source_interval_um"], self.source_z)
                and self.contains(fact["effective_interval_um"], self.effective_z)
            )
        )
        return {
            **self.binding,
            "effective_z_um": self.effective_z,
            "eligible_entity_ids": eligible,
        }

    def contains(self, interval, z):
        """Test interval incidence."""
        return interval[0] <= z <= interval[1]

    def host_record(self, eid, pid):
        """Detach the source fields of one bound host."""
        fact = self.facts[eid]
        return {
            "source_entity_id": eid,
            "source_polygon_id": pid,
            "physical_layer_id": fact["physical_layer_id"],
            "source_interval_um": list(fact["source_interval_um"]),
        }


def _source_contact_rings(polygon):
    return tuple(
        (
            tuple(
                (
                    tuple((Fraction(str(float(value))) for value in point))
                    for point in ring
                )
            )
            for ring in (polygon.exterior, *polygon.holes)
        )
    )


def junction_source_binding(inventory):
    """Extract the source-support binding fields from a contact inventory."""
    return {
        name: plain_record(inventory[name])
        for name in (
            "schema_version",
            "source_polygon_id",
            "source_z_um",
            "basis",
            "source_hosts",
        )
    }
