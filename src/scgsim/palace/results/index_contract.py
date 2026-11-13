"""Validate route, terminal, port and Physical Group correspondence in sealed inputs.

Only the schemas the run-folder contract names (``MANIFEST_SCHEMA_VERSION`` and ``INDEX_MAP_SCHEMA_VERSION``)
are read.  A run folder of another schema is refused; ``require_manifest_schema`` and ``require_index_map_schema`` reject them with the recorded schema
in the message.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_index_map import (
    DIELECTRIC_SECTION,
    ENERGY_SECTION,
    LUMPED_PORT_SECTION,
    TERMINAL_SECTION,
)
from scgsim.contracts.palace_run_folder import (
    CONFIG,
    INDEX_MAP,
    INDEX_MAP_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    MESH,
    OUTPUT_DIR,
    palace_problem,
)
from scgsim.palace.results.artifacts import (
    expect_list,
    expect_mapping,
    expect_scalar,
    optional_dict,
    refusal,
)
from scgsim.palace.results.surface_contract import (
    entry_rows,
    surface_epr_entries,
    validate_surface_epr_config,
)


def require_manifest_schema(manifest: Mapping[str, Any]) -> None:
    """Reject a mesh manifest that is not the current schema, naming the recorded one."""
    recorded = manifest.get("schema_version")
    if type(recorded) is not int or recorded != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"mesh manifest schema_version {recorded!r} is not readable: only schema_version "
            f"{MANIFEST_SCHEMA_VERSION} is read, and run folders of older schemas are not."
        )


def require_index_map_schema(index_map: Mapping[str, Any]) -> None:
    """Reject a palace index map that is not the current schema, naming the recorded one."""
    recorded = index_map.get("schema_version")
    if type(recorded) is not int or recorded != INDEX_MAP_SCHEMA_VERSION:
        raise ValueError(
            f"palace index map schema_version {recorded!r} is not readable: only schema_version "
            f"{INDEX_MAP_SCHEMA_VERSION} is read, and run folders of older schemas are not."
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
        require_manifest_schema(self.mesh_manifest)
        require_index_map_schema(self.index_map)
        manifest_route = expect_scalar(self.mesh_manifest, "route", str, fallback="")
        if manifest_route != self.route:
            raise ValueError(
                refusal(
                    "mesh manifest route does not match resolved route",
                    found=manifest_route,
                    expected=self.route,
                )
            )
        thin_film = self.handoff_metadata.get("route_a_thin_film")
        if (
            self.mesh_manifest.get("route_a_thin_film") != thin_film
            or self.run_metadata.get("route_a_thin_film") != thin_film
        ):
            raise ValueError(
                refusal(
                    "mesh manifest and run metadata Route-A thin-film identity mismatch",
                    found={
                        "mesh manifest": self.mesh_manifest.get("route_a_thin_film"),
                        "run metadata": self.run_metadata.get("route_a_thin_film"),
                    },
                    expected=thin_film,
                )
            )

        run_route = expect_scalar(self.run_metadata, "route", str)
        handoff_route = expect_scalar(self.handoff_metadata, "route", str)
        if run_route != self.route or handoff_route != self.route:
            raise ValueError(
                refusal(
                    "metadata route does not match resolved route",
                    found={
                        "run metadata": run_route,
                        "handoff metadata": handoff_route,
                    },
                    expected=self.route,
                )
            )

        validate_index_entries(self.index_map)
        _validate_index_against_manifest(self.index_map, self.mesh_manifest)

        run_problem = expect_scalar(self.run_metadata, "problem", str)
        handoff_problem = expect_scalar(self.handoff_metadata, "problem", str)
        if run_problem != self.problem or handoff_problem != self.problem:
            raise ValueError(
                refusal(
                    "metadata problem does not match resolved problem",
                    found={
                        "run metadata": run_problem,
                        "handoff metadata": handoff_problem,
                    },
                    expected=self.problem,
                )
            )


_SECTION_DIMENSIONS = {
    TERMINAL_SECTION: 2,
    LUMPED_PORT_SECTION: 2,
    DIELECTRIC_SECTION: 2,
    ENERGY_SECTION: 3,
}


def _validate_index_against_manifest(
    index_map: dict[str, Any], mesh_manifest: dict[str, Any]
) -> None:
    """Every attribute an index entry lists is a manifest attribute of the right dimension and group.

    The manifest names the primary SGB group of each solver attribute, so an entry's ``physical_names``
    (and, for a Surface-EPR row, the row's ``group``) must be the groups of the attributes it lists.
    """
    primary_group: dict[tuple[int, int], str] = {}
    for record in expect_list(mesh_manifest.get("groups"), "mesh manifest groups"):
        for attribute in expect_list(record.get("attributes"), "manifest attributes"):
            primary_group[(record["dimension"], attribute)] = record["group"]
    for entry in expect_list(index_map.get("entries"), "palace_index_map.entries"):
        dimension = _SECTION_DIMENSIONS[entry["section"]]
        try:
            groups = {primary_group[(dimension, a)] for a in entry["attributes"]}
        except KeyError as exc:
            raise ValueError(
                f"index attribute {exc.args[0][1]} is not a mesh manifest attribute."
            ) from exc
        if groups != set(entry["physical_names"]):
            raise ValueError(
                refusal(
                    f"index entry {entry.get('entry_name')!r} physical_names disagree with the mesh manifest",
                    found=sorted(entry["physical_names"]),
                    expected=sorted(groups),
                )
            )
    for entry in surface_epr_entries(index_map):
        for row in entry_rows(entry):
            if {primary_group[(2, a)] for a in row["attributes"]} != {row["group"]}:
                raise ValueError(
                    refusal(
                        f"index surface row {row.get('row')!r} group disagrees with the mesh manifest",
                        found=row["group"],
                        expected=sorted(
                            {primary_group[(2, a)] for a in row["attributes"]}
                        ),
                    )
                )


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
    if not palace_problem(problem).solves_modes:
        terminals = expect_list(
            boundaries.get("Terminal"), "config.Boundaries.Terminal"
        )
        if index_counts["terminal"] != len(terminals):
            raise ValueError(
                refusal(
                    "terminal count mismatch between config and index map",
                    found=index_counts["terminal"],
                    expected=len(terminals),
                )
            )
        _validate_ground_boundary_resolution(boundaries, index_map)
    else:
        ports = expect_list(
            boundaries.get("LumpedPort", []),
            "config.Boundaries.LumpedPort",
            allow_empty=True,
        )
        if index_counts["lumped_port"] != len(ports):
            raise ValueError(
                refusal(
                    "lumped port count mismatch between config and index map",
                    found=index_counts["lumped_port"],
                    expected=len(ports),
                )
            )

    postprocessing = optional_dict(boundaries.get("Postprocessing")) or {}
    post_diel = expect_list(
        postprocessing.get("Dielectric", []),
        "config.Boundaries.Postprocessing.Dielectric",
        allow_empty=True,
    )
    if post_diel and index_counts["surface"] < len(post_diel):
        raise ValueError(
            refusal(
                "surface count in index map does not cover the config's Dielectric postprocessing list",
                found=index_counts["surface"],
                expected=f"at least {len(post_diel)}",
            )
        )
    expected_sections = {
        TERMINAL_SECTION: terminals,
        LUMPED_PORT_SECTION: ports,
        ENERGY_SECTION: expect_list(
            expect_mapping(
                expect_mapping(config.get("Domains"), "config.Domains").get(
                    "Postprocessing"
                ),
                "config.Domains.Postprocessing",
            ).get("Energy"),
            "config.Domains.Postprocessing.Energy",
        ),
        DIELECTRIC_SECTION: post_diel,
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
    validate_surface_epr_config(config, index_map)


def validate_config_problem(config: dict[str, Any], problem: str) -> None:
    """Require the solver problem and numerical controls recorded for this workflow."""
    problem_block = optional_dict(config.get("Problem"))
    if not problem_block:
        raise ValueError(f"{CONFIG} missing Problem block.")
    config_problem = expect_scalar(problem_block, "Type", str, fallback="")
    if config_problem != problem:
        raise ValueError(
            f"config Problem.Type {config_problem!r} does not match metadata problem {problem!r}."
        )
    model_block = optional_dict(config.get("Model"))
    mesh_name = (
        expect_scalar(model_block, "Mesh", str, fallback="") if model_block else ""
    )
    if mesh_name != MESH:
        raise ValueError(f"config Model.Mesh must reference {MESH}.")

    if problem_block.get("Output") not in {OUTPUT_DIR, f"{OUTPUT_DIR}/", None}:
        raise ValueError(f"config Problem.Output must be {OUTPUT_DIR}.")


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
        raise ValueError(
            refusal(
                "ground boundary index evidence has unexpected fields",
                found=sorted(record),
                expected=sorted(list_fields),
            )
        )
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
        raise ValueError(
            refusal(
                "physical/exterior Ground evidence has an invalid union",
                found={
                    "physical": sorted(physical),
                    "exterior": sorted(exterior),
                    "union": values["union_attributes"],
                },
                expected=f"disjoint sets whose union is {union}",
            )
        )
    configured = boundaries.get("Ground")
    if union:
        if configured != {"Attributes": union}:
            raise ValueError(
                refusal(
                    "config Ground attributes do not match index evidence",
                    found=configured,
                    expected={"Attributes": union},
                )
            )
    elif configured is not None:
        raise ValueError(
            refusal(
                "config contains Ground without indexed attributes", found=configured
            )
        )


def validate_index_entries(index_map: dict[str, Any]) -> None:
    """Check unique section-local identities, physical attributes and source-role metadata."""
    entries = index_map.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{Path(INDEX_MAP).name} must include non-empty entries list.")
    expected_roles = {
        TERMINAL_SECTION: "terminal",
        LUMPED_PORT_SECTION: "port_surface",
        ENERGY_SECTION: "domain_energy",
        DIELECTRIC_SECTION: "surface_epr",
    }
    indices: dict[str, list[int]] = {section: [] for section in expected_roles}
    names: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry:
            raise ValueError(
                refusal("index map entries must be non-empty mappings", found=entry)
            )
        section = expect_scalar(entry, "section", str)
        if section not in expected_roles:
            raise ValueError(f"index map has unsupported section {section!r}.")
        role = expect_scalar(entry, "role", str)
        if role != expected_roles[section]:
            raise ValueError(
                refusal(
                    f"index map section {section} and role are inconsistent",
                    found=role,
                    expected=expected_roles[section],
                )
            )
        index = expect_scalar(entry, "index", int)
        if index < 1:
            raise ValueError(
                refusal(f"index map indices of {section} must be positive", found=index)
            )
        indices[section].append(index)
        entry_name = expect_scalar(entry, "entry_name", str)
        if not entry_name or (section, entry_name) in names:
            raise ValueError(
                refusal(
                    f"index entry names must be non-empty and unique per section ({section})",
                    found=entry_name,
                )
            )
        names.add((section, entry_name))
        physical_names = expect_list(
            entry.get("physical_names"), "index physical_names"
        )
        if not all(isinstance(name, str) and name for name in physical_names):
            raise ValueError(
                refusal(
                    f"index entry {entry_name!r} physical_names must be non-empty strings",
                    found=physical_names,
                )
            )
        entry_attributes = expect_list(entry.get("attributes"), "index attributes")
        for attribute in entry_attributes:
            if (
                not isinstance(attribute, int)
                or isinstance(attribute, bool)
                or attribute < 1
            ):
                raise ValueError(
                    refusal(
                        f"index entry {entry_name!r} attributes must be positive integers",
                        found=attribute,
                    )
                )
        metadata = expect_mapping(entry.get("metadata"), "index metadata")
        _validate_entry_role(entry, metadata, section)
    for section, values in indices.items():
        if values and sorted(values) != list(range(1, len(values) + 1)):
            raise ValueError(f"index map {section} entries must be ordered 1..N.")
    surface_epr_entries(index_map)


def _validate_entry_role(
    entry: dict[str, Any], metadata: dict[str, Any], section: str
) -> None:
    """Validate the source identities required by one configured role."""
    if section == TERMINAL_SECTION:
        expect_scalar(entry, "terminal_name", str)
        expect_scalar(entry, "net_id", str)
        expect_list(
            entry.get("conductor_component_ids"), "terminal conductor components"
        )
    elif section == LUMPED_PORT_SECTION:
        expect_scalar(entry, "port_name", str)
        hosts = expect_list(metadata.get("host_entity_ids"), "port host entity ids")
        if len(hosts) != 2 or not all(isinstance(host, str) and host for host in hosts):
            raise ValueError(
                refusal(
                    f"lumped port index entry {entry.get('port_name')!r} must have exactly two hosts",
                    found=hosts,
                )
            )
    elif section == ENERGY_SECTION:
        expect_mapping(metadata.get("material"), "domain index material")
