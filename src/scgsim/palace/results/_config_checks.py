"""Check that ``config.json`` agrees with the run metadata and the index map.

``config.json`` is what Palace actually solved. The index map and the handoff
metadata are what SCGSim says it asked for. Every check here compares the two
and raises when they differ, because a result table is only meaningful for the
configuration that produced it.
"""

from __future__ import annotations

import math
from typing import Any

from ._expect import _expect_list, _expect_mapping, _expect_scalar, _optional_dict
from ._index_map import (
    IndexCounts,
    _surface_mask_entries,
    _validate_index_entries,
)
from ._run_documents import HandoffMetadata, IndexMap, RunMetadata

# ------------------------------------------------- problem, route, mesh name


def _validate_config_problem(config: dict[str, Any], problem: str) -> None:
    problem_block = _optional_dict(config.get("Problem"))
    if not problem_block:
        raise ValueError("config.json missing Problem block.")
    config_problem = _expect_scalar(problem_block, "Type", str, fallback="")
    if config_problem != problem:
        raise ValueError(
            f"config Problem.Type {config_problem!r} does not match metadata problem {problem!r}."
        )
    model_block = _optional_dict(config.get("Model"))
    mesh_name = (
        _expect_scalar(model_block, "Mesh", str, fallback="") if model_block else ""
    )
    if mesh_name != "palace.msh":
        raise ValueError("config Model.Mesh must reference palace.msh.")

    # Both config writers set Output to exactly this directory, and the result
    # tables are read from it.
    output = problem_block.get("Output")
    if output is None:
        raise ValueError(
            "config Problem.Output is missing; supported: 'results/palace'."
        )
    if output != "results/palace":
        raise ValueError("config Problem.Output must be results/palace.")


def _validate_route_consistency(
    route: str,
    run_metadata: RunMetadata,
    handoff_metadata: HandoffMetadata,
    mesh_manifest: dict[str, Any],
    index_map: IndexMap,
    index_counts: IndexCounts,
    problem: str,
) -> None:
    if _expect_scalar(mesh_manifest, "schema_version", int, fallback=0) != 1:
        raise ValueError("mesh manifest schema_version must be integer 1.")
    if _expect_scalar(mesh_manifest, "route", str, fallback="") != route:
        raise ValueError("mesh manifest route does not match resolved route.")
    thin_film = handoff_metadata.get("route_a_thin_film")
    if (
        mesh_manifest.get("route_a_thin_film") != thin_film
        or run_metadata.get("route_a_thin_film") != thin_film
    ):
        raise ValueError(
            "mesh manifest and run metadata Route-A thin-film identity mismatch."
        )

    run_route = _expect_scalar(run_metadata, "route", str)
    handoff_route = _expect_scalar(handoff_metadata, "route", str)
    if run_route != route or handoff_route != route:
        raise ValueError("metadata route does not match resolved route.")

    _validate_index_map_schema(index_map, problem)
    _validate_index_entries(index_map)
    _validate_index_routes(index_map, route)

    run_problem = _expect_scalar(run_metadata, "problem", str)
    handoff_problem = _expect_scalar(handoff_metadata, "problem", str)
    if run_problem != problem or handoff_problem != problem:
        raise ValueError("metadata problem does not match resolved problem.")


def _validate_index_map_schema(index_map: IndexMap, problem: str) -> None:
    """Electrostatic writes index-map schema 2 (adds the Ground split), Eigenmode 1."""
    supported = 1 if problem == "Eigenmode" else 2
    found = _expect_scalar(index_map, "schema_version", int, fallback=0)
    if found != supported:
        raise ValueError(
            f"palace index map schema_version is {found}; supported for {problem}: "
            f"{supported}. Prepare the handoff again with the current SCGSim."
        )


def _validate_index_routes(index_map: IndexMap, route: str) -> None:
    """An entry that records a route in its metadata must record the run's route."""
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    for entry in entries:
        recorded = _expect_scalar(entry["metadata"], "route", str, fallback="")
        if recorded in {"A", "B"} and recorded != route:
            raise ValueError("index map route is inconsistent with run route.")


# ----------------------------------------- config sections vs index entries


def _validate_config_index_correspondence(
    config: dict[str, Any],
    index_map: IndexMap,
    index_counts: IndexCounts,
    problem: str,
) -> None:
    boundaries = _optional_dict(config.get("Boundaries")) or {}
    if problem == "Electrostatic":
        terminals = _expect_list(
            boundaries.get("Terminal"), "config.Boundaries.Terminal"
        )
        if index_counts["terminal"] != len(terminals):
            raise ValueError("terminal count mismatch between config and index map.")
        _validate_ground_boundary_resolution(boundaries, index_map)
    else:
        ports = _expect_list(
            boundaries.get("LumpedPort", []),
            "config.Boundaries.LumpedPort",
            allow_empty=True,
        )
        if index_counts["lumped_port"] != len(ports):
            raise ValueError("lumped port count mismatch between config and index map.")

    postprocessing = _optional_dict(boundaries.get("Postprocessing")) or {}
    post_diel = _expect_list(
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
        "Domains.Postprocessing.Energy": _expect_list(
            _expect_mapping(
                _expect_mapping(config.get("Domains"), "config.Domains").get(
                    "Postprocessing"
                ),
                "config.Domains.Postprocessing",
            ).get("Energy"),
            "config.Domains.Postprocessing.Energy",
        ),
        "Boundaries.Postprocessing.Dielectric": post_diel,
    }
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    _check_indexed_sections(expected_sections, entries)
    _validate_surface_mask_config(config, index_map)


def _check_indexed_sections(
    expected_sections: dict[str, list[Any]], entries: list[Any]
) -> None:
    """Per section, config items and index entries match one-to-one by Index."""
    for section, configured in expected_sections.items():
        indexed = sorted(
            (
                _expect_mapping(entry, "palace index entry")
                for entry in entries
                if _expect_scalar(entry, "section", str) == section
            ),
            key=lambda entry: _expect_scalar(entry, "index", int),
        )
        if len(indexed) != len(configured):
            raise ValueError(f"config/index entry count mismatch for {section}.")
        for position, (entry, configured_item) in enumerate(
            zip(indexed, configured, strict=True), start=1
        ):
            item = _expect_mapping(configured_item, f"config {section} item")
            if _expect_scalar(entry, "index", int) != _expect_scalar(
                item, "Index", int
            ):
                raise ValueError(
                    f"config/index Index mismatch for {section} {position}."
                )
            if _expect_list(
                entry.get("attributes"), "index attributes"
            ) != _expect_list(item.get("Attributes"), "config Attributes"):
                raise ValueError(
                    f"config/index Attributes mismatch for {section} {position}."
                )


# ------------------------------------------------------ Ground (schema 2)


# The index map records physical-Ground and exterior-Ground attributes separately;
# their union must equal ``Boundaries.Ground.Attributes``
# (docs/specs/palace-electrostatic-sgb.qmd). Only Electrostatic index maps carry it.
def _validate_ground_boundary_resolution(
    boundaries: dict[str, Any], index_map: IndexMap
) -> None:
    record = _expect_mapping(
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
        field: _expect_list(
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


# ------------------------------------------------- masked Surface EPR config


def _validate_surface_mask_config(config: dict[str, Any], index_map: IndexMap) -> None:
    masks = _surface_mask_entries(index_map)
    boundaries = _optional_dict(config.get("Boundaries")) or {}
    postprocessing = _optional_dict(boundaries.get("Postprocessing")) or {}
    declared_rows = postprocessing.get("Dielectric")
    config_declares_masks = isinstance(declared_rows, list) and any(
        isinstance(row, dict) and "Mask" in row for row in declared_rows
    )
    if not masks:
        if config_declares_masks:
            raise ValueError("config contains unindexed surface EPR masks.")
        return
    rows = _expect_list(
        postprocessing.get("Dielectric"),
        "config.Boundaries.Postprocessing.Dielectric",
        allow_empty=True,
    )
    configured = {
        _expect_scalar(
            _expect_mapping(row, "config dielectric row"), "Index", int
        ): _expect_mapping(row, "config dielectric row")
        for row in rows
    }
    model = _expect_mapping(config.get("Model"), "config.Model")
    model_l0_m = model.get("L0")
    if (
        isinstance(model_l0_m, bool)
        or not isinstance(model_l0_m, (int, float))
        or not math.isfinite(float(model_l0_m))
        or model_l0_m <= 0
    ):
        raise ValueError("config Model.L0 must be a finite positive length.")
    dielectric_entries = [
        _expect_mapping(entry, "palace dielectric index entry")
        for entry in _expect_list(index_map.get("entries"), "palace_index_map.entries")
        if _expect_mapping(entry, "palace index entry").get("section")
        == "Boundaries.Postprocessing.Dielectric"
    ]
    indexed_dielectric_indexes = {
        _expect_scalar(entry, "index", int) for entry in dielectric_entries
    }
    if (
        len(configured) != len(rows)
        or set(configured) != indexed_dielectric_indexes
        or len(rows) != len(dielectric_entries)
    ):
        raise ValueError("config and index evidence disagree on dielectric rows.")
    indexed_mask_indexes = {_expect_scalar(entry, "index", int) for entry in masks}
    configured_mask_indexes = {
        index for index, row in configured.items() if "Mask" in row
    }
    if configured_mask_indexes != indexed_mask_indexes:
        raise ValueError("config and index evidence disagree on masked surface rows.")
    _check_configured_dielectric_rows(
        dielectric_entries, configured, model_l0_m, indexed_mask_indexes
    )


def _check_configured_dielectric_rows(
    dielectric_entries: list[dict[str, Any]],
    configured: dict[int, dict[str, Any]],
    model_l0_m: Any,
    indexed_mask_indexes: set[int],
) -> None:
    """Each configured dielectric row repeats its index evidence field for field."""
    for entry in dielectric_entries:
        index = _expect_scalar(entry, "index", int)
        row = configured.get(index)
        if row is None:
            raise ValueError("surface EPR index is absent from config.")
        if _expect_list(
            row.get("Attributes"), "config dielectric Attributes"
        ) != _expect_list(entry.get("attributes"), "surface EPR index attributes"):
            raise ValueError(
                "config surface EPR attributes do not match index evidence."
            )
        metadata = _expect_mapping(entry.get("metadata"), "surface EPR metadata")
        epr_spec = _expect_mapping(entry.get("epr_spec"), "surface EPR spec")
        if float(epr_spec["model_l0_m"]) != float(model_l0_m):
            raise ValueError("surface film Model.L0 disagrees with config")
        expected_fields = {
            "Type": metadata.get("interface_type"),
            "Thickness": epr_spec.get("thickness"),
            "Permittivity": epr_spec.get("permittivity"),
            "LossTan": 0.0
            if epr_spec.get("loss_tangent") is None
            else epr_spec["loss_tangent"],
        }
        if any(
            key not in row or row[key] != value
            for key, value in expected_fields.items()
        ):
            raise ValueError(
                "config surface EPR fields do not match source index evidence."
            )
        if index not in indexed_mask_indexes:
            continue
        mask = _expect_mapping(row.get("Mask"), "config dielectric Mask")
        evidence = _expect_mapping(entry.get("mask"), "surface EPR mask")
        if mask != {"Type": "Inset", "Margin": evidence["native_margin"]}:
            raise ValueError("config surface EPR mask does not match index evidence.")
        if float(evidence["model_l0_m"]) != float(model_l0_m):
            raise ValueError("surface EPR mask Model.L0 disagrees with config.")
