"""Validate route, terminal, port and Physical Group correspondence in sealed inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from scgsim.palace.results.artifacts import (
    expect_list,
    expect_mapping,
    expect_scalar,
    optional_dict,
)
from scgsim.palace.results.surface_contract import (
    surface_mask_entries,
    validate_surface_mask_config,
)


@dataclass(frozen=True)
class RouteConsistency:
    """Recorded route, source metadata, mesh and physical indices from one run."""

    route: str
    run_metadata: dict[str, Any]
    handoff_metadata: dict[str, Any]
    mesh_manifest: dict[str, Any]
    index_map: dict[str, Any]
    index_counts: dict[str, int]
    problem: str

    def validate(self) -> None:
        """Check mesh/index schema and route agreement before interpreting configured groups."""
        if expect_scalar(self.mesh_manifest, "schema_version", int, fallback=0) != 1:
            raise ValueError("mesh manifest schema_version must be integer 1.")
        if expect_scalar(self.mesh_manifest, "route", str, fallback="") != self.route:
            raise ValueError("mesh manifest route does not match resolved route.")
        thin_film = self.handoff_metadata.get("route_a_thin_film")
        if (
            self.mesh_manifest.get("route_a_thin_film") != thin_film
            or self.run_metadata.get("route_a_thin_film") != thin_film
        ):
            raise ValueError(
                "mesh manifest and run metadata Route-A thin-film identity mismatch."
            )

        run_route = expect_scalar(self.run_metadata, "route", str)
        handoff_route = expect_scalar(self.handoff_metadata, "route", str)
        if run_route != self.route or handoff_route != self.route:
            raise ValueError("metadata route does not match resolved route.")

        index_schema = expect_scalar(self.index_map, "schema_version", int, fallback=0)
        if index_schema not in {1, 2}:
            raise ValueError("palace index map schema_version must be integer 1 or 2.")
        if self.problem == "Eigenmode" and index_schema != 1:
            raise ValueError(
                "Eigenmode palace index map schema_version must be integer 1."
            )
        validate_index_entries(self.index_map)
        entries = expect_list(self.index_map.get("entries"), "palace_index_map.entries")
        for entry in entries:
            candidate = expect_scalar(
                cast(dict[str, Any], optional_dict(entry)).get("metadata", {}),
                "route",
                str,
                fallback="",
            )
            if not candidate:
                candidate = expect_scalar(entry, "route", str, fallback="")
            if candidate in {"A", "B"} and candidate != self.route:
                raise ValueError("index map route is inconsistent with run route.")

        run_problem = expect_scalar(self.run_metadata, "problem", str)
        handoff_problem = expect_scalar(self.handoff_metadata, "problem", str)
        if run_problem != self.problem or handoff_problem != self.problem:
            raise ValueError("metadata problem does not match resolved problem.")


def validate_config_index_correspondence(
    config: dict[str, Any],
    index_map: dict[str, Any],
    index_counts: dict[str, int],
    problem: str,
) -> None:
    """Bind configured terminals, ports and postprocessing rows to their indexed source identities."""
    boundaries = optional_dict(config.get("Boundaries")) or {}
    terminals: list[Any] = []
    ports: list[Any] = []
    if problem == "Electrostatic":
        terminals = expect_list(
            boundaries.get("Terminal"), "config.Boundaries.Terminal"
        )
        if index_counts["terminal"] != len(terminals):
            raise ValueError("terminal count mismatch between config and index map.")
        if expect_scalar(index_map, "schema_version", int) == 2:
            _validate_ground_boundary_resolution(boundaries, index_map)
    else:
        ports = expect_list(
            boundaries.get("LumpedPort", []),
            "config.Boundaries.LumpedPort",
            allow_empty=True,
        )
        if index_counts["lumped_port"] != len(ports):
            raise ValueError("lumped port count mismatch between config and index map.")

    postprocessing = optional_dict(boundaries.get("Postprocessing")) or {}
    post_diel = expect_list(
        postprocessing.get("Dielectric", []),
        "config.Boundaries.Postprocessing.Dielectric",
        allow_empty=True,
    )
    if post_diel and index_counts["surface"] < len(post_diel):
        raise ValueError(
            "surface count in index map does not cover postprocessing list."
        )
    expected_sections = {
        "Boundaries.Terminal": terminals if problem == "Electrostatic" else [],
        "Boundaries.LumpedPort": ports if problem == "Eigenmode" else [],
        "Domains.Postprocessing.Energy": expect_list(
            expect_mapping(
                expect_mapping(config.get("Domains"), "config.Domains").get(
                    "Postprocessing"
                ),
                "config.Domains.Postprocessing",
            ).get("Energy"),
            "config.Domains.Postprocessing.Energy",
        ),
        "Boundaries.Postprocessing.Dielectric": post_diel,
    }
    entries = expect_list(index_map.get("entries"), "palace_index_map.entries")
    for section, configured in expected_sections.items():
        indexed = sorted(
            (
                expect_mapping(entry, "palace index entry")
                for entry in entries
                if expect_scalar(entry, "section", str) == section
            ),
            key=lambda entry: expect_scalar(entry, "index", int),
        )
        if len(indexed) != len(configured):
            raise ValueError(f"config/index entry count mismatch for {section}.")
        for position, (entry, configured_item) in enumerate(
            zip(indexed, configured, strict=True), start=1
        ):
            item = expect_mapping(configured_item, f"config {section} item")
            if expect_scalar(entry, "index", int) != expect_scalar(item, "Index", int):
                raise ValueError(
                    f"config/index Index mismatch for {section} {position}."
                )
            if expect_list(entry.get("attributes"), "index attributes") != expect_list(
                item.get("Attributes"), "config Attributes"
            ):
                raise ValueError(
                    f"config/index Attributes mismatch for {section} {position}."
                )
    validate_surface_mask_config(config, index_map)


def validate_config_problem(config: dict[str, Any], problem: str) -> None:
    """Require the solver problem and numerical controls recorded for this workflow."""
    problem_block = optional_dict(config.get("Problem"))
    if not problem_block:
        raise ValueError("config.json missing Problem block.")
    config_problem = expect_scalar(problem_block, "Type", str, fallback="")
    if config_problem != problem:
        raise ValueError(
            f"config Problem.Type {config_problem!r} does not match metadata problem {problem!r}."
        )
    model_block = optional_dict(config.get("Model"))
    mesh_name = (
        expect_scalar(model_block, "Mesh", str, fallback="") if model_block else ""
    )
    if mesh_name != "palace.msh":
        raise ValueError("config Model.Mesh must reference palace.msh.")

    if problem_block.get("Output") not in {"results/palace", "results/palace/", None}:
        raise ValueError("config Problem.Output must be results/palace.")


def _validate_ground_boundary_resolution(
    boundaries: dict[str, Any], index_map: dict[str, Any]
) -> None:
    """Verify physical/exterior Ground attributes and their exact configured union."""
    record = expect_mapping(
        index_map.get("ground_boundary"), "palace_index_map.ground_boundary"
    )
    list_fields = (
        "requested_net_ids",
        "physical_net_ids",
        "conductor_component_ids",
        "physical_names",
        "physical_attributes",
        "exterior_physical_names",
        "exterior_attributes",
        "union_attributes",
    )
    if set(record) != set(list_fields):
        raise ValueError("ground boundary index evidence has unexpected fields.")
    values = {
        field: expect_list(
            record.get(field), f"ground_boundary.{field}", allow_empty=True
        )
        for field in list_fields
    }
    for field in (
        "requested_net_ids",
        "physical_net_ids",
        "conductor_component_ids",
        "physical_names",
        "exterior_physical_names",
    ):
        if not all(isinstance(item, str) and item for item in values[field]) or values[
            field
        ] != sorted(set(values[field])):
            raise ValueError(f"ground_boundary.{field} must be sorted unique text.")
    for field in (
        "physical_attributes",
        "exterior_attributes",
        "union_attributes",
    ):
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in values[field]
        ) or values[field] != sorted(set(values[field])):
            raise ValueError(
                f"ground_boundary.{field} must be sorted unique positive integers."
            )
    physical = set(values["physical_attributes"])
    exterior = set(values["exterior_attributes"])
    union = sorted(physical | exterior)
    if physical & exterior or values["union_attributes"] != union:
        raise ValueError("physical/exterior Ground evidence has an invalid union.")
    configured = boundaries.get("Ground")
    if union:
        if configured != {"Attributes": union}:
            raise ValueError("config Ground attributes do not match index evidence.")
    elif configured is not None:
        raise ValueError("config contains Ground without indexed attributes.")


def validate_index_entries(index_map: dict[str, Any]) -> None:
    """Check unique section-local identities, physical attributes and source-role metadata."""
    entries = index_map.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("palace_index_map.json must include non-empty entries list.")
    expected_roles = {
        "Boundaries.Terminal": "terminal",
        "Boundaries.LumpedPort": "port_surface",
        "Domains.Postprocessing.Energy": "domain_energy",
        "Boundaries.Postprocessing.Dielectric": "surface_epr",
    }
    indices: dict[str, list[int]] = {section: [] for section in expected_roles}
    names: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry:
            raise ValueError("index map entries must be non-empty mappings.")
        section = expect_scalar(entry, "section", str)
        if section not in expected_roles:
            raise ValueError(f"index map has unsupported section {section!r}.")
        if expect_scalar(entry, "role", str) != expected_roles[section]:
            raise ValueError("index map section and role are inconsistent.")
        index = expect_scalar(entry, "index", int)
        if index < 1:
            raise ValueError("index map indices must be positive.")
        indices[section].append(index)
        entry_name = expect_scalar(entry, "entry_name", str)
        if not entry_name or (section, entry_name) in names:
            raise ValueError(
                "index entry names must be non-empty and unique per section."
            )
        names.add((section, entry_name))
        physical_names = expect_list(
            entry.get("physical_names"), "index physical_names"
        )
        if not all(isinstance(name, str) and name for name in physical_names):
            raise ValueError("index physical_names must be non-empty strings.")
        entry_attributes = expect_list(entry.get("attributes"), "index attributes")
        for attribute in entry_attributes:
            if (
                not isinstance(attribute, int)
                or isinstance(attribute, bool)
                or attribute < 1
            ):
                raise ValueError("index attributes must be positive integers.")
        metadata = expect_mapping(entry.get("metadata"), "index metadata")
        _validate_entry_role(entry, metadata, section)
    for section, values in indices.items():
        if values and sorted(values) != list(range(1, len(values) + 1)):
            raise ValueError(f"index map {section} entries must be ordered 1..N.")
    surface_mask_entries(index_map)


def _validate_entry_role(
    entry: dict[str, Any], metadata: dict[str, Any], section: str
) -> None:
    """Validate the source identities required by one configured role."""
    if section == "Boundaries.Terminal":
        expect_scalar(entry, "terminal_name", str)
        expect_scalar(entry, "net_id", str)
        expect_list(
            entry.get("conductor_component_ids"), "terminal conductor components"
        )
    elif section == "Boundaries.LumpedPort":
        expect_scalar(entry, "port_name", str)
        owners = expect_list(
            metadata.get("owner_semantic_ids"), "port owner semantic ids"
        )
        if len(owners) != 2 or not all(
            isinstance(owner, str) and owner for owner in owners
        ):
            raise ValueError("lumped port index entry must have exactly two owners.")
    elif section == "Domains.Postprocessing.Energy":
        expect_mapping(metadata.get("material"), "domain index material")
    elif section == "Boundaries.Postprocessing.Dielectric":
        expect_mapping(metadata.get("physical_attribute"), "surface index attributes")
