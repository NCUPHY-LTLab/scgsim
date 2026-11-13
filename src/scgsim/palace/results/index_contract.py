"""Validate route, terminal, port and Physical Group correspondence in sealed inputs.

Only the schemas the run-folder contract names (``MANIFEST_SCHEMA_VERSION`` and ``INDEX_MAP_SCHEMA_VERSION``)
are read.  A run folder of another schema is refused; ``require_manifest_schema`` and ``require_index_map_schema`` reject them with the recorded schema
in the message.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

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
from scgsim.geometry.models.refusals import Declared
from scgsim.palace.results.artifacts import (
    RETURNED_RUN_FOLDER,
    expect_list,
    expect_mapping,
    expect_scalar,
    optional_dict,
)
from scgsim.palace.results.surface_contract import (
    entry_rows,
    surface_epr_entries,
    validate_surface_epr_config,
)
from scgsim.refusal import found_text


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
                Declared(
                    "mesh manifest route does not match resolved route",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(manifest_route, self.route))
            )
        thin_film = self.handoff_metadata.get("route_a_thin_film")
        if (
            self.mesh_manifest.get("route_a_thin_film") != thin_film
            or self.run_metadata.get("route_a_thin_film") != thin_film
        ):
            raise ValueError(
                Declared(
                    "mesh manifest and run metadata Route-A thin-film identity mismatch",
                    RETURNED_RUN_FOLDER,
                ).refusal(
                    found_text(
                        {
                            "mesh manifest": self.mesh_manifest.get(
                                "route_a_thin_film"
                            ),
                            "run metadata": self.run_metadata.get("route_a_thin_film"),
                        },
                        thin_film,
                    )
                )
            )

        run_route = expect_scalar(self.run_metadata, "route", str)
        handoff_route = expect_scalar(self.handoff_metadata, "route", str)
        if run_route != self.route or handoff_route != self.route:
            raise ValueError(
                Declared(
                    "metadata route does not match resolved route", RETURNED_RUN_FOLDER
                ).refusal(
                    found_text(
                        {
                            "run metadata": run_route,
                            "handoff metadata": handoff_route,
                        },
                        self.route,
                    )
                )
            )

        validate_index_entries(self.index_map)
        _validate_index_against_manifest(self.index_map, self.mesh_manifest)

        run_problem = expect_scalar(self.run_metadata, "problem", str)
        handoff_problem = expect_scalar(self.handoff_metadata, "problem", str)
        if run_problem != self.problem or handoff_problem != self.problem:
            raise ValueError(
                Declared(
                    "metadata problem does not match resolved problem",
                    RETURNED_RUN_FOLDER,
                ).refusal(
                    found_text(
                        {
                            "run metadata": run_problem,
                            "handoff metadata": handoff_problem,
                        },
                        self.problem,
                    )
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
                Declared(
                    f"index entry {entry.get('entry_name')!r} physical_names disagree with the mesh manifest",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(sorted(entry["physical_names"]), sorted(groups)))
            )
    for entry in surface_epr_entries(index_map):
        for row in entry_rows(entry):
            if {primary_group[(2, a)] for a in row["attributes"]} != {row["group"]}:
                raise ValueError(
                    Declared(
                        f"index surface row {row.get('row')!r} group disagrees with the mesh manifest",
                        RETURNED_RUN_FOLDER,
                    ).refusal(
                        found_text(
                            row["group"],
                            sorted({primary_group[(2, a)] for a in row["attributes"]}),
                        )
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
    excitations = _configured_excitations(boundaries, index_map, index_counts, problem)
    dielectric = _configured_dielectric(boundaries, index_counts)
    configured_sections = {
        TERMINAL_SECTION: excitations.terminals,
        LUMPED_PORT_SECTION: excitations.ports,
        ENERGY_SECTION: expect_list(
            expect_mapping(
                expect_mapping(config.get("Domains"), "config.Domains").get(
                    "Postprocessing"
                ),
                "config.Domains.Postprocessing",
            ).get("Energy"),
            "config.Domains.Postprocessing.Energy",
        ),
        DIELECTRIC_SECTION: dielectric,
    }
    entries = expect_list(index_map.get("entries"), "palace_index_map.entries")
    for section, configured in configured_sections.items():
        _match_section_rows(section, configured, entries)
    validate_surface_epr_config(config, index_map)


class _Excitations(NamedTuple):
    """The configured Terminal rows of an Electrostatic config, or the LumpedPort rows of an Eigenmode one."""

    terminals: list[Any]
    ports: list[Any]


def _configured_excitations(
    boundaries: dict[str, Any],
    index_map: dict[str, Any],
    index_counts: dict[str, int],
    problem: str,
) -> _Excitations:
    """Read the problem's excitation rows and check that the index map counts as many."""
    if not palace_problem(problem).solves_modes:
        terminals = expect_list(
            boundaries.get("Terminal"), "config.Boundaries.Terminal"
        )
        if index_counts["terminal"] != len(terminals):
            raise ValueError(
                Declared(
                    "terminal count mismatch between config and index map",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(index_counts["terminal"], len(terminals)))
            )
        _validate_ground_boundary_resolution(boundaries, index_map)
        return _Excitations(terminals, [])
    ports = expect_list(
        boundaries.get("LumpedPort", []),
        "config.Boundaries.LumpedPort",
        allow_empty=True,
    )
    if index_counts["lumped_port"] != len(ports):
        raise ValueError(
            Declared(
                "lumped port count mismatch between config and index map",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(index_counts["lumped_port"], len(ports)))
        )
    return _Excitations([], ports)


def _configured_dielectric(
    boundaries: dict[str, Any], index_counts: dict[str, int]
) -> list[Any]:
    """Read the config's Dielectric postprocessing rows and check that the index map covers them."""
    postprocessing = optional_dict(boundaries.get("Postprocessing")) or {}
    dielectric = expect_list(
        postprocessing.get("Dielectric", []),
        "config.Boundaries.Postprocessing.Dielectric",
        allow_empty=True,
    )
    if dielectric and index_counts["surface"] < len(dielectric):
        raise ValueError(
            Declared(
                "surface count in index map does not cover the config's Dielectric postprocessing list",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(index_counts["surface"], f"at least {len(dielectric)}")
            )
        )
    return dielectric


def _match_section_rows(
    section: str, configured: list[Any], entries: list[Any]
) -> None:
    """Pair the section's index entries with its config rows by index, and require equal attributes."""
    indexed = sorted(
        (
            expect_mapping(entry, "palace index entry")
            for entry in entries
            if expect_scalar(entry, "section", str) == section
        ),
        key=lambda entry: expect_scalar(entry, "index", int),
    )
    if len(indexed) != len(configured):
        raise ValueError(
            Declared(
                f"config/index entry count mismatch for {section}", RETURNED_RUN_FOLDER
            ).refusal(found_text(len(indexed), len(configured)))
        )
    for position, (entry, configured_item) in enumerate(
        zip(indexed, configured, strict=True), start=1
    ):
        item = expect_mapping(configured_item, f"config {section} item")
        index = expect_scalar(entry, "index", int)
        config_index = expect_scalar(item, "Index", int)
        if index != config_index:
            raise ValueError(
                Declared(
                    f"config/index Index mismatch for {section} {position}",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(index, config_index))
            )
        attributes = expect_list(entry.get("attributes"), "index attributes")
        config_attributes = expect_list(item.get("Attributes"), "config Attributes")
        if attributes != config_attributes:
            raise ValueError(
                Declared(
                    f"config/index Attributes mismatch for {section} {position}",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(attributes, config_attributes))
            )


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
    values = _ground_resolution_fields(index_map)
    _validate_ground_union(boundaries, values)


def _ground_resolution_fields(index_map: dict[str, Any]) -> dict[str, list[Any]]:
    """The index map's Ground resolution, each field a sorted list of unique names or positive attributes."""
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
            Declared(
                "ground boundary index evidence has unexpected fields",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(sorted(record), sorted(list_fields)))
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
            raise ValueError(
                Declared(
                    f"ground_boundary.{field} must be sorted unique text",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(values[field]))
            )
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
                Declared(
                    f"ground_boundary.{field} must be sorted unique positive integers",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(values[field]))
            )
    return values


def _validate_ground_union(
    boundaries: dict[str, Any], values: dict[str, list[Any]]
) -> None:
    """Physical and exterior Ground attributes are disjoint, and their union is the config's Ground."""
    physical = set(values["physical_attributes"])
    exterior = set(values["exterior_attributes"])
    union = sorted(physical | exterior)
    if physical & exterior or values["union_attributes"] != union:
        raise ValueError(
            Declared(
                "physical/exterior Ground evidence has an invalid union",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    {
                        "physical": sorted(physical),
                        "exterior": sorted(exterior),
                        "union": values["union_attributes"],
                    },
                    f"disjoint sets whose union is {union}",
                )
            )
        )
    configured = boundaries.get("Ground")
    if union:
        if configured != {"Attributes": union}:
            raise ValueError(
                Declared(
                    "config Ground attributes do not match index evidence",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(configured, {"Attributes": union}))
            )
    elif configured is not None:
        raise ValueError(
            Declared(
                "config contains Ground without indexed attributes", RETURNED_RUN_FOLDER
            ).refusal(found_text(configured))
        )


def validate_index_entries(index_map: dict[str, Any]) -> None:
    """Check unique section-local identities, physical attributes and source-role metadata."""
    entries = index_map.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            Declared(
                f"{Path(INDEX_MAP).name} must include a non-empty entries list",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(entries))
        )
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
                Declared(
                    "index map entries must be non-empty mappings", RETURNED_RUN_FOLDER
                ).refusal(found_text(entry))
            )
        section = expect_scalar(entry, "section", str)
        if section not in expected_roles:
            raise ValueError(f"index map has unsupported section {section!r}.")
        role = expect_scalar(entry, "role", str)
        if role != expected_roles[section]:
            raise ValueError(
                Declared(
                    f"index map section {section} and role are inconsistent",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(role, expected_roles[section]))
            )
        index = expect_scalar(entry, "index", int)
        if index < 1:
            raise ValueError(
                Declared(
                    f"index map indices of {section} must be positive",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(index))
            )
        indices[section].append(index)
        _validate_entry_fields(entry, section, names)
    for section, values in indices.items():
        if values and sorted(values) != list(range(1, len(values) + 1)):
            raise ValueError(
                Declared(
                    f"index map {section} entries must be numbered 1..N",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(sorted(values)))
            )
    surface_epr_entries(index_map)


def _validate_entry_fields(
    entry: dict[str, Any], section: str, names: set[tuple[str, str]]
) -> None:
    """One entry's name (unique in its section), physical names, attributes and role metadata."""
    entry_name = expect_scalar(entry, "entry_name", str)
    if not entry_name or (section, entry_name) in names:
        raise ValueError(
            Declared(
                f"index entry names must be non-empty and unique per section ({section})",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(entry_name))
        )
    names.add((section, entry_name))
    physical_names = expect_list(entry.get("physical_names"), "index physical_names")
    if not all(isinstance(name, str) and name for name in physical_names):
        raise ValueError(
            Declared(
                f"index entry {entry_name!r} physical_names must be non-empty strings",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(physical_names))
        )
    entry_attributes = expect_list(entry.get("attributes"), "index attributes")
    for attribute in entry_attributes:
        if (
            not isinstance(attribute, int)
            or isinstance(attribute, bool)
            or attribute < 1
        ):
            raise ValueError(
                Declared(
                    f"index entry {entry_name!r} attributes must be positive integers",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(attribute))
            )
    metadata = expect_mapping(entry.get("metadata"), "index metadata")
    _validate_entry_role(entry, metadata, section)


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
                Declared(
                    f"lumped port index entry {entry.get('port_name')!r} must have exactly two hosts",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(hosts))
            )
    elif section == ENERGY_SECTION:
        expect_mapping(metadata.get("material"), "domain index material")
