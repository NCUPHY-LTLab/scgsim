"""Map exact compiler physical records onto live Gmsh groups without inventing identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from .provenance import SGB_RUNTIME_AUTHORITY, STRUCTURED_INTERFACE_TYPES
from .record_validation import (
    has_structured_record_marker,
    is_solver_active_volume,
    normalized_direction,
    optional_string,
    required_distinct_strings,
    required_string,
    structured_conductor_source_layer,
    validate_route_representation,
    validate_structured_domain_boundary_record,
    validate_structured_surface_record,
    validate_structured_volume_record,
    validate_surface_contribution_ledger,
)


def build_groups_from_sgb_records() -> dict[str, dict[str, Any]]:
    """Pre-allocate group dicts before physical mapping enrichment."""

    return {
        "volumes": {},
        "boundary_surfaces": {},
        "surface_groups": {},
        "pec_surfaces": {},
        "port_surfaces": {},
    }


def augment_groups_from_live_records(
    *,
    records: Sequence[Any],
    groups: dict[str, dict[str, Any]],
    live: Mapping[tuple[int, str], tuple[int, tuple[int, ...]]],
    route: str,
    solution_material_kinds: Mapping[str, str],
) -> None:
    """Populate mesh groups by exact (dimension, physical_name) mapping."""
    route = route.upper()
    missing: list[str] = []

    _validate_record_routes(records, route, missing)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        dim = int(record.get("dimension", 0) or 0)
        name = optional_string(record.get("physical_name"))
        if not name:
            if record.get("solver_use") == "solver_active":
                missing.append(f"{dim}:<missing physical_name>")
            continue

        if optional_string(record.get("route", "")) != route:
            raise ValueError(
                f"SGB physical-group route must be {route} for this mesh mode."
            )
        if record.get("solver_use") == "not_solver_active":
            continue

        if (dim, name) not in live:
            if str(record.get("solver_use", "")).lower() == "solver_active":
                missing.append(f"{dim}:{name}")
            continue

        _validate_live_record_role(record, dim, name)

        phys_group, entity_tags = live[(dim, name)]
        if dim == 3 and is_solver_active_volume(record):
            _add_volume_group(
                groups=groups,
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
            )
            continue

        if dim == 2:
            _SurfaceGroup(
                groups,
                name,
                phys_group,
                entity_tags,
                record,
                route,
                solution_material_kinds,
            ).add_surface()

    if missing:
        raise ValueError(
            "SGB XAO is missing solver-active physical groups from sidecar: "
            + ", ".join(sorted(missing))
        )


def _add_volume_group(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
) -> None:
    """Add one solver-active solution volume."""
    validate_structured_volume_record(name, record)
    role = str(record.get("role", "")).lower()
    if "construction" in role or role == "mm_contact":
        raise ValueError(
            f"SGB volume {name!r} has construction-only/MM-contact role {role!r} "
            "and must not become a solver volume."
        )

    info = {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 3,
        "source": SGB_RUNTIME_AUTHORITY,
        "physical_name": name,
        "stack_layer": record.get("source_provenance", {}).get("stack_layer"),
        "source_layer": record.get("source_provenance", {}).get(
            "conductor_source_layer_name"
        ),
        "representation": record.get("representation"),
        "route": record.get("route"),
        "source_provenance": dict(record.get("source_provenance", {})),
        "physical_attribute": dict(record.get("physical_attribute", {})),
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use", "solver_active"),
        "role": record.get("role"),
        "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        ),
        "metadata": dict(record.get("metadata", {})),
    }

    groups["volumes"][name] = info


def solution_material_kinds(
    records: Sequence[Any], stack: Mapping[str, Any]
) -> dict[str, str]:
    """Map SGB solution-volume stable IDs to explicit stack material kinds."""
    materials = stack.get("materials")
    if not isinstance(materials, Mapping):
        raise TypeError("stack must carry explicit materials while meshing.")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or not is_solver_active_volume(record):
            continue
        physical = record.get("physical_attribute")
        provenance = record.get("source_provenance")
        ids = physical.get("material_ids") if isinstance(physical, Mapping) else None
        kinds = (
            physical.get("material_kinds") if isinstance(physical, Mapping) else None
        )
        volume_ids = (
            provenance.get("volume_ids") if isinstance(provenance, Mapping) else None
        )
        owner_ids = record.get("owner_semantic_ids")
        if not all(
            isinstance(value, (list, tuple)) and value
            for value in (ids, kinds, volume_ids, owner_ids)
        ):
            raise ValueError(
                "SGB solution volume needs explicit material_ids/material_kinds, provenance.volume_ids, and owner_semantic_ids."
            )
        ids = cast(Sequence[Any], ids)
        kinds = cast(Sequence[Any], kinds)
        volume_ids = cast(Sequence[Any], volume_ids)
        owner_ids = cast(Sequence[Any], owner_ids)
        _validate_material_cardinalities(ids, kinds, volume_ids, owner_ids)
        is_broadcast = len(ids) == 1
        for material_id, material_kind, volume_id, owner_id in zip(
            ids if not is_broadcast else [ids[0]] * len(volume_ids),
            kinds if not is_broadcast else [kinds[0]] * len(volume_ids),
            volume_ids,
            owner_ids,
            strict=not is_broadcast,
        ):
            if not all(
                isinstance(value, str) and value
                for value in (material_id, material_kind, volume_id, owner_id)
            ):
                raise ValueError(
                    "SGB solution volume stable ids, material ids, and material kinds must be non-empty strings."
                )
            material = materials.get(material_id)
            stack_kind = material.get("kind") if isinstance(material, Mapping) else None
            if stack_kind not in {"vacuum", "dielectric"}:
                raise ValueError(
                    "SGB solution volume material must resolve vacuum or dielectric kind."
                )
            if material_kind not in {"vacuum", "dielectric"}:
                raise ValueError(
                    "SGB solution volume material_kinds must be vacuum or dielectric."
                )
            if material_kind != stack_kind:
                raise ValueError(
                    "SGB solution volume material_kind disagrees with explicit stack material kind."
                )
            for stable_id in (volume_id, owner_id):
                previous = result.setdefault(stable_id, material_kind)
                if previous != material_kind:
                    raise ValueError(
                        f"Structured solution id {stable_id!r} has conflicting material kinds."
                    )
    if not result:
        raise ValueError(
            "SGB sidecar has no explicit solution-volume material records."
        )
    return result


def _validate_surface_adjacency(
    name: str,
    record: Mapping[str, Any],
    interface_type: str,
    material_kinds: Mapping[str, str],
) -> None:
    adjacent_groups = _surface_local_adjacencies(record)
    if not adjacent_groups:
        raise ValueError(
            f"SGB surface {name!r} has no explicit adjacent solution volumes."
        )
    for adjacent in adjacent_groups:
        kinds: list[str] = []
        for identifier in adjacent:
            if not isinstance(identifier, str) or identifier not in material_kinds:
                raise ValueError(
                    f"SGB surface {name!r} adjacent solution id {identifier!r} "
                    "cannot resolve explicit material kind."
                )
            kinds.append(material_kinds[identifier])
        if interface_type == "MS_MA":
            if str(record.get("face_kind", "")).lower() != "interface" or sorted(
                kinds
            ) != ["dielectric", "vacuum"]:
                raise ValueError(
                    f"SGB MS_MA surface {name!r} requires face_kind=interface "
                    "and exactly dielectric/vacuum adjacency in every local source."
                )
        elif interface_type in {"MA", "MS", "SA"}:
            # The required material pairing is an explicit semantic compatibility
            # check, never a geometric orientation or material-name inference.
            expected = {"MA": "vacuum", "MS": "dielectric", "SA": "vacuum"}[
                interface_type
            ]
            if interface_type == "SA":
                if sorted(kinds) != ["dielectric", "vacuum"]:
                    raise ValueError(
                        f"SGB SA surface {name!r} requires exactly "
                        "dielectric/vacuum adjacency in every local source."
                    )
            elif expected not in kinds:
                raise ValueError(
                    f"SGB {interface_type} surface {name!r} lacks explicit adjacent "
                    f"{expected} solution material in a local source."
                )


def _surface_local_adjacencies(
    record: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Read exact source-local adjacency without treating a group union as a pair."""
    provenance = record.get("source_provenance")
    members = (
        _surface_provenance_members(provenance)
        if isinstance(provenance, Mapping)
        else ()
    )
    sources = tuple(
        source
        for source in members
        if isinstance(source.get("surface_contribution_ledger"), (list, tuple))
    )
    if sources and len(sources) != len(members):
        raise ValueError(
            "SGB surface contribution provenance mixes ledgered and legacy sources."
        )
    local: list[tuple[str, ...]] = []
    for source in sources:
        ledger = source.get("surface_contribution_ledger")
        if not isinstance(ledger, (list, tuple)):
            continue
        identifiers: list[str] = []
        for contribution in ledger:
            if not isinstance(contribution, Mapping):
                continue
            effective = contribution.get("effective_domain_ids")
            if not isinstance(effective, (list, tuple)):
                continue
            for identifier in effective:
                if isinstance(identifier, str) and identifier not in identifiers:
                    identifiers.append(identifier)
        if identifiers:
            local.append(tuple(identifiers))
    if local:
        return tuple(local)
    adjacent = record.get("adjacent_solution_volume_ids")
    if not isinstance(adjacent, (list, tuple)) or not adjacent:
        return ()
    return (tuple(adjacent),)


def _surface_provenance_members(
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    sources = provenance.get("sources")
    if not isinstance(sources, (list, tuple)):
        return (provenance,)
    return tuple(
        member
        for source in sources
        if isinstance(source, Mapping)
        for member in _surface_provenance_members(source)
    )


def live_physical_groups() -> dict[tuple[int, str], tuple[int, tuple[int, ...]]]:
    """Map live gmsh (dimension, physical name) -> (physical-group, tags)."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    live: dict[tuple[int, str], tuple[int, tuple[int, ...]]] = {}
    for dim, phys_group in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_group)
        if not name:
            continue
        live[(int(dim), str(name))] = (
            int(phys_group),
            tuple(
                int(tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_group)
            ),
        )
    return live


@dataclass
class _SurfaceGroup:
    """One exact live surface record and its destination mappings."""

    groups: dict[str, dict[str, Any]]
    name: str
    phys_group: int
    entity_tags: tuple[int, ...]
    record: Mapping[str, Any]
    route: str
    solution_material_kinds: Mapping[str, str]

    def add_surface(self) -> None:
        """Classify structured surface records into boundary + interface mappings."""
        if self.record.get("solver_use") == "not_solver_active":
            return

        role = str(self.record.get("role", "")).lower()
        source_record_ids = self._source_record_ids()
        if role == "domain_boundary":
            self._add_domain_boundary(source_record_ids)
            return

        if not has_structured_record_marker(self.record):
            if str(self.record.get("solver_use", "")).lower() == "solver_active":
                raise ValueError(
                    f"SGB surface {self.name!r} is missing structured metadata."
                )
            self.groups["surface_groups"][self.name] = {
                "physical_name": self.name,
                "dim": 2,
            }
            return

        if str(self.record.get("solver_use")) == "not_solver_active":
            return

        interface_type = str(self.record.get("interface_type", "")).upper()
        if interface_type == "LUMPED_PORT" or role in {"lumped_port", "port_surface"}:
            self.add_port()
            return
        validate_structured_surface_record(name=self.name, record=self.record)
        validate_route_representation(
            name=self.name, record=self.record, route=self.route
        )
        if interface_type not in STRUCTURED_INTERFACE_TYPES:
            raise ValueError(
                f"SGB surface {self.name!r} has unsupported interface_type {interface_type!r}."
            )
        validate_surface_contribution_ledger(self.name, self.record, interface_type)

        component_id = optional_string(self.record.get("conductor_component_id"))
        net_id = optional_string(self.record.get("net_id"))
        if interface_type in {"MA", "MS", "MS_MA"} and (
            component_id is None or net_id is None
        ):
            raise ValueError(
                f"SGB conductor surface {self.name!r} must include conductor_component_id and net_id."
            )
        if interface_type == "SA" and net_id is not None:
            raise ValueError(
                f"SGB SA surface {self.name!r} must not carry conductor/net identity."
            )

        if interface_type in {"MM", "SS", "AA"}:
            self.groups["boundary_surfaces"][self.name] = self.info(
                alias_name=self.name,
                interface_type=interface_type,
                surface_epr=False,
            )
            return
        _validate_surface_adjacency(
            self.name, self.record, interface_type, self.solution_material_kinds
        )
        self._add_interface_aliases(interface_type, component_id)

    def _add_interface_aliases(
        self, interface_type: str, component_id: str | None
    ) -> None:
        surface_id = str(self.record["surface_id"])
        if interface_type == "MS_MA":
            source_id = structured_conductor_source_layer(
                name=self.name, record=self.record
            )
            # One SGB physical attribute carries both semantic aliases.  Do not use
            # geometric orientation or material names to invent top/bottom meaning.
            parsed_records = (
                ("MS", source_id, "interface", f"{surface_id}__MS"),
                ("MA", source_id, "interface", f"{surface_id}__MA"),
            )
        else:
            source_id = (
                surface_id
                if interface_type == "SA"
                else structured_conductor_source_layer(
                    name=self.name, record=self.record
                )
            )
            parsed_records = (
                (interface_type, source_id, str(self.record["face_kind"]), surface_id),
            )

        for parsed in parsed_records:
            interface_type, source_id, face_kind, alias_name = parsed
            if interface_type not in STRUCTURED_INTERFACE_TYPES:
                continue
            alias_info = self.info(
                alias_name=alias_name,
                interface_type=interface_type,
                surface_epr=interface_type in {"MA", "MS", "SA"},
                source_id=source_id,
                face_kind=face_kind,
            )
            self.groups["boundary_surfaces"][alias_name] = alias_info
            if interface_type in {"MA", "MS"} and component_id is not None:
                self.groups["pec_surfaces"][alias_name] = {
                    "phys_group": self.phys_group,
                    "tags": list(self.entity_tags),
                    "dim": 2,
                    "source": SGB_RUNTIME_AUTHORITY,
                    "source_id": source_id,
                    "layer": source_id,
                    "representation": str(
                        self.record.get("representation", "")
                    ).upper(),
                    "owner_semantic_ids": tuple(
                        self.record.get("owner_semantic_ids", ())
                    ),
                }

    def add_port(self) -> None:
        """Preserve one SGB-authored layout sheet without reconstructing its meaning."""
        if self.record.get("solver_use") != "solver_active":
            raise ValueError(
                f"SGB lumped-port sheet {self.name!r} must be solver_active."
            )
        if self.record.get("representation") != "lumped_port_sheet":
            raise ValueError(
                f"SGB lumped-port sheet {self.name!r} requires lumped_port_sheet representation."
            )
        surface_id = required_string(
            self.record.get("surface_id"), f"{self.name} surface_id"
        )
        owners = required_distinct_strings(
            self.record.get("owner_semantic_ids"),
            f"{self.name} owner_semantic_ids",
            count=2,
        )
        adjacent = required_distinct_strings(
            self.record.get("adjacent_solution_volume_ids"),
            f"{self.name} adjacent_solution_volume_ids",
            count=2 if self.route == "A" else 1,
        )
        attribute = self.record.get("physical_attribute")
        provenance = self.record.get("source_provenance")
        if not isinstance(attribute, Mapping) or not isinstance(provenance, Mapping):
            raise TypeError(
                f"SGB lumped-port sheet {self.name!r} requires physical_attribute and source_provenance."
            )
        port_index = attribute.get("port_index")
        if (
            isinstance(port_index, bool)
            or not isinstance(port_index, int)
            or port_index < 1
        ):
            raise ValueError(
                f"SGB lumped-port sheet {self.name!r} requires 1-based port_index."
            )
        port_name = required_string(
            attribute.get("port_name"), f"{self.name} port_name"
        )
        source_layer = required_string(
            attribute.get("source_layer"), f"{self.name} source_layer"
        )
        target_layer = required_string(
            attribute.get("target_layer"), f"{self.name} target_layer"
        )
        embedded_volume_id = required_string(
            attribute.get("embedded_volume_id"), f"{self.name} embedded_volume_id"
        )
        direction = normalized_direction(attribute.get("direction"), self.name)
        attribute_owners = required_distinct_strings(
            attribute.get("owner_semantic_ids"),
            f"{self.name} physical_attribute.owner_semantic_ids",
            count=2,
        )
        if attribute_owners != owners:
            raise ValueError(
                f"SGB lumped-port sheet {self.name!r} owner identities disagree."
            )
        self._validate_owner_provenance(attribute, owners)
        if self.route == "A":
            if embedded_volume_id not in adjacent:
                raise ValueError(
                    f"SGB Route-A lumped-port sheet {self.name!r} embedded volume must be in actual adjacency."
                )
        elif adjacent != (embedded_volume_id,):
            raise ValueError(
                f"SGB Route-B lumped-port sheet {self.name!r} requires one matching embedded volume."
            )
        if (
            provenance.get("source_name") != port_name
            or provenance.get("port_index") != port_index
            or provenance.get("source_layer") != source_layer
            or provenance.get("target_layer") != target_layer
            or normalized_direction(provenance.get("direction"), self.name) != direction
            or not optional_string(provenance.get("source_polygon_id"))
            or not optional_string(provenance.get("direction_sign_convention"))
        ):
            raise ValueError(
                f"SGB lumped-port sheet {self.name!r} source provenance is inconsistent."
            )
        key = f"P{port_index}"
        if key in self.groups["port_surfaces"]:
            raise ValueError(f"Duplicate SGB lumped-port sheet index {port_index}.")
        self.groups["port_surfaces"][key] = {
            "phys_group": self.phys_group,
            "tags": list(self.entity_tags),
            "dim": 2,
            "type": "lumped_sheet",
            "source": SGB_RUNTIME_AUTHORITY,
            "surface_epr": False,
            "sgb_record": "final_physical_group",
            "solver_use": self.record.get("solver_use"),
            "role": self.record.get("role"),
            "route": self.route,
            "representation": self.record.get("representation"),
            "surface_id": surface_id,
            "interface_type": "lumped_port",
            "face_kind": self.record.get("face_kind"),
            "owner_semantic_ids": owners,
            "adjacent_solution_volume_ids": adjacent,
            "interface_of": adjacent if self.route == "A" else (),
            "embedded_volume_id": embedded_volume_id,
            "direction": direction,
            "physical_name": self.name,
            "source_provenance": dict(provenance),
            "physical_attribute": dict(attribute),
            "conductor_component_id": self.record.get("conductor_component_id"),
            "net_id": self.record.get("net_id"),
            "equipotential_id": self.record.get("equipotential_id"),
        }

    def info(
        self,
        *,
        alias_name: str,
        interface_type: str,
        surface_epr: bool,
        source_id: str | None = None,
        face_kind: str | None = None,
    ) -> dict[str, Any]:
        """Keep every structured identity field when lowering a surface group."""
        return {
            "phys_group": self.phys_group,
            "tags": list(self.entity_tags),
            "dim": 2,
            "source": "volume_interface",
            "surface_epr": surface_epr,
            "interface_id": alias_name,
            "interface_type": interface_type,
            "sgb_record": "final_physical_group",
            "solver_use": self.record.get("solver_use"),
            "conductor_component_id": optional_string(
                self.record.get("conductor_component_id")
            ),
            "net_id": optional_string(self.record.get("net_id")),
            "equipotential_id": optional_string(self.record.get("equipotential_id")),
            "physical_attribute": dict(self.record.get("physical_attribute", {})),
            "metadata": dict(self.record.get("metadata", {})),
            "source_id": source_id
            or optional_string(self.record.get("surface_id"))
            or alias_name,
            "face_kind": face_kind or optional_string(self.record.get("face_kind")),
            "contact_kind": optional_string(self.record.get("contact_kind")),
            "representation": optional_string(self.record.get("representation")),
            "route": optional_string(self.record.get("route")),
            "source_provenance": dict(self.record.get("source_provenance", {})),
            "source_record_ids": tuple(
                self.record.get("metadata", {}).get("source_record_ids", ())
            )
            if isinstance(self.record.get("metadata"), Mapping)
            else (),
            "surface_id": optional_string(self.record.get("surface_id")) or alias_name,
            "owner_semantic_ids": tuple(self.record.get("owner_semantic_ids", ())),
            "adjacent_solution_volume_ids": tuple(
                self.record.get("adjacent_solution_volume_ids", ())
            ),
        }

    def _source_record_ids(self) -> tuple[str, ...]:
        metadata = self.record.get("metadata")
        source_record_ids = (
            tuple(
                value
                for value in (
                    metadata.get("source_record_ids", ())
                    if isinstance(metadata, Mapping)
                    else ()
                )
                if isinstance(value, str) and value
            )
            if isinstance(metadata, Mapping)
            else ()
        )

        return source_record_ids

    def _add_domain_boundary(self, source_record_ids: tuple[str, ...]) -> None:
        validate_structured_domain_boundary_record(self.name, self.record)
        info = {
            "phys_group": self.phys_group,
            "tags": list(self.entity_tags),
            "dim": 2,
            "source": "domain_boundary",
            "surface_epr": False,
            "representation": str(self.record.get("representation", "")),
            "route": self.record.get("route"),
            "interface_type": self.record.get("interface_type"),
            "contact_kind": self.record.get("contact_kind"),
            "face_kind": optional_string(self.record.get("face_kind")),
            "source_id": optional_string(self.record.get("surface_id")) or self.name,
            "owner_semantic_ids": tuple(self.record.get("owner_semantic_ids", ())),
            "adjacent_solution_volume_ids": tuple(
                self.record.get("adjacent_solution_volume_ids", ())
            ),
            "surface_id": optional_string(self.record.get("surface_id")),
            "physical_attribute": dict(self.record.get("physical_attribute", {})),
            "source_provenance": dict(self.record.get("source_provenance", {})),
            "conductor_component_id": optional_string(
                self.record.get("conductor_component_id")
            ),
            "net_id": optional_string(self.record.get("net_id")),
            "equipotential_id": optional_string(self.record.get("equipotential_id")),
            "sgb_record": "final_physical_group",
            "solver_use": self.record.get("solver_use", "solver_active"),
            "sgb_metadata": dict(self.record.get("metadata", {})),
            "source_record_ids": source_record_ids,
        }
        self.groups["boundary_surfaces"][self.name] = info

    def _validate_owner_provenance(
        self, attribute: Mapping[str, Any], owners: tuple[str, ...]
    ) -> None:
        owner_provenance = attribute.get("owner_provenance")
        if (
            not isinstance(owner_provenance, (list, tuple))
            or len(owner_provenance) != 2
        ):
            raise ValueError(
                f"SGB lumped-port sheet {self.name!r} requires two owner provenance rows."
            )
        for item in owner_provenance:
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"SGB lumped-port sheet {self.name!r} owner provenance must be mappings."
                )
            owner_id = required_string(
                item.get("semantic_id"), f"{self.name} owner semantic_id"
            )
            if owner_id not in owners:
                raise ValueError(
                    f"SGB lumped-port sheet {self.name!r} has foreign owner provenance."
                )
            for field in ("conductor_component_id", "net_id", "equipotential_id"):
                value = item.get(field)
                if value is not None:
                    required_string(value, f"{self.name} owner {field}")


def _validate_record_routes(
    records: Sequence[Any], route: str, missing: list[str]
) -> None:
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        if optional_string(record.get("route", "")) != route:
            raise ValueError(f"SGB physical-group record route must be {route}.")
        if (
            record.get("sgb_record")
            not in {None, "final_physical_group", "preliminary_physical_group"}
            and str(record.get("solver_use")) == "solver_active"
        ):
            missing.append("unknown_sgb_record")


def _validate_material_cardinalities(
    ids: Any, kinds: Any, volume_ids: Any, owner_ids: Any
) -> None:
    if not (
        (len(ids) == len(kinds) == len(volume_ids) == len(owner_ids))
        or (len(ids) == 1 and len(kinds) == 1 and len(volume_ids) == len(owner_ids))
    ):
        raise ValueError(
            "SGB solution volume material kind and stable-id cardinalities must agree."
        )


def _validate_live_record_role(record: Mapping[str, Any], dim: int, name: str) -> None:
    if dim == 3 and not is_solver_active_volume(record):
        raise ValueError(
            f"SGB physical volume {name!r} is not an explicit solver-active "
            "material_volume and cannot enter the Palace solution."
        )
    if dim == 2 and record.get("role") == "material_volume":
        raise ValueError(f"SGB surface {name!r} cannot claim material_volume role.")
