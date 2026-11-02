"""Extract verified mode EPR and apply detached film-assumption reanalysis.

EPR means energy participation ratio; native integration and solving stay external."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from scgsim.palace.presentation.epr import show_epr as render
from scgsim.palace.results.artifacts import confined_path, sha256
from scgsim.palace.results.epr_model import (
    PalaceEprResult,
    checked_number,
    surface_distance_band,
)
from scgsim.palace.results.models import ResolvedRunData
from scgsim.palace.results.surface_contract import surface_mask_entries
from scgsim.palace.results.tables import (
    count_index_entries,
    eigenmode_count,
    output_families,
    read_csv_table,
    validate_surface_mask_table,
    validate_table,
)
from scgsim.semantics.epr import film_assumptions, inverse_quality_factor


def resolve_epr_result(path: str | Path) -> PalaceEprResult:
    """Load a detached Palace EPR record without reading a solver directory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    return PalaceEprResult.from_payload(json.loads(source.read_text(encoding="utf-8")))


def _iteration_result(
    result: ResolvedRunData, iteration: int
) -> tuple[ResolvedRunData, str]:
    """Read one sealed native iteration with the existing table validators."""

    if type(iteration) is not int or iteration <= 0:
        raise ValueError("palace_iteration must be a positive integer")
    records = {item["path"]: item for item in result.returned_receipt.output_files}
    prefixes = set()
    for path in records:
        parts = Path(path).parts
        if (
            len(parts) != 4
            or parts[:2] != ("results", "palace")
            or parts[3] != "palace.json"
        ):
            continue
        suffix = parts[2].removeprefix("iteration")
        if (
            parts[2].startswith("iteration")
            and suffix.isdigit()
            and int(suffix) == iteration
        ):
            prefixes.add("/".join(parts[:3]))
    if len(prefixes) != 1:
        raise ValueError("requested native iteration lacks one receipt-bound directory")
    prefix = prefixes.pop()

    def verified(name: str) -> Path:
        """Return an iteration artifact only when its receipt size and hash still match."""
        relative = f"{prefix}/{name}"
        record = records.get(relative)
        path = confined_path(result.run_dir, relative)
        if (
            record is None
            or record.get("present") is not True
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            raise ValueError(
                f"native iteration artifact is missing or changed: {relative}"
            )
        return path

    native = json.loads(verified("palace.json").read_text(encoding="utf-8"))
    if (
        not isinstance(native, dict)
        or not isinstance(native.get("Problem"), dict)
        or native["Problem"].get("Iteration") != iteration
    ):
        raise ValueError("native iteration metadata disagrees with selected directory")
    counts = count_index_entries(result.provenance.index_map)
    tables = {}
    for family in output_families(result.problem, counts):
        table = read_csv_table(verified(f"{family}.csv"))
        validate_table(result.problem, table, counts, result.provenance.config)
        tables[family] = table
    masks = surface_mask_entries(result.provenance.index_map)
    if masks:
        for family in ("surface-mask-Q", "surface-mask-energy"):
            table = read_csv_table(verified(f"{family}.csv"))
            validate_surface_mask_table(
                table,
                problem=result.problem,
                expected_rows=eigenmode_count(result.provenance.config),
                entries=masks,
            )
            tables[family] = table
    return replace(
        result, tables=tables, provenance=replace(result.provenance, palace_json=native)
    ), prefix


def epr_result(
    result: ResolvedRunData, *, palace_iteration: int | None = None
) -> PalaceEprResult:
    """Extract final or explicitly selected sealed-iteration Eigenmode EPR."""

    if not isinstance(result, ResolvedRunData) or result.problem != "Eigenmode":
        raise TypeError("a resolved Palace Eigenmode run is required")
    output_prefix = "results/palace"
    if palace_iteration is not None:
        result, output_prefix = _iteration_result(result, palace_iteration)
    index = result.provenance.index_map
    config = result.provenance.config
    l0 = checked_number(config["Model"]["L0"], "Model.L0", positive=True)
    entries = index["entries"]
    surfaces = {
        item["index"]: item
        for item in entries
        if item["section"] == "Boundaries.Postprocessing.Dielectric"
    }
    domains = {
        item["index"]: item
        for item in entries
        if item["section"] == "Domains.Postprocessing.Energy"
    }
    junctions = {
        item["index"]: item
        for item in entries
        if item["section"] == "Boundaries.LumpedPort"
    }
    request, selected = _selected_outputs(domains, index, junctions, surfaces)
    eig = result.tables["eig"].rows
    surface_table = (
        result.tables["surface-Q"].rows
        if surfaces
        else tuple({"m": item["m"]} for item in eig)
    )
    domain_table = result.tables["domain-E"].rows
    port_table = (
        result.tables["port-EPR"].rows
        if junctions
        else tuple({"m": item["m"]} for item in eig)
    )
    masks_requested = any("mask" in entry for entry in surfaces.values())
    mask_q_table = (
        result.tables["surface-mask-Q"].rows if masks_requested else (None,) * len(eig)
    )
    mask_energy_table = (
        result.tables["surface-mask-energy"].rows
        if masks_requested
        else (None,) * len(eig)
    )
    rows: list[dict[str, Any]] = []
    _ModeExtraction(
        domain_table,
        domains,
        eig,
        junctions,
        l0,
        mask_energy_table,
        mask_q_table,
        masks_requested,
        port_table,
        rows,
        selected,
        surface_table,
        surfaces,
    ).extract()
    provenance = _source_provenance(index, output_prefix, request, result)
    return PalaceEprResult(tuple(rows), provenance)


def reanalyze_epr(
    result: PalaceEprResult,
    *,
    surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> PalaceEprResult:
    """Apply a new whole-group film snapshot to stored native participation."""

    if not isinstance(result, PalaceEprResult):
        raise TypeError("result must be PalaceEprResult")
    defaults = (
        {}
        if surface_defaults is None
        else {
            key: film_assumptions(value, partial=True)
            for key, value in surface_defaults.items()
        }
    )
    overrides = (
        {}
        if group_overrides is None
        else {
            key: film_assumptions(value, partial=True)
            for key, value in group_overrides.items()
        }
    )
    if set(defaults) - {"MA", "MS", "SA"}:
        raise ValueError("surface_defaults supports only MA, MS, and SA")
    known = {item["group_id"] for row in result.rows for item in row["surface"]}
    if set(overrides) - known:
        raise ValueError("group_overrides contains an unknown group")
    rows: list[dict[str, Any]] = []
    for row in result.rows:
        surfaces: list[dict[str, Any]] = []
        for item in row["surface"]:
            original = film_assumptions(item["original_assumptions"])
            current = film_assumptions(item["assumptions"])
            interface_update = defaults.get(item["interface_kind"], {})
            group_update = overrides.get(item["group_id"], {})
            applied = {**interface_update, **group_update}
            candidate = {**current, **applied}
            if any(
                name in applied
                for name in (
                    "film_thickness_m",
                    "film_relative_permittivity",
                    "loss_tangent",
                )
            ):
                if "source" not in applied:
                    candidate["source"] = (
                        "offline_group_override"
                        if group_update
                        else "offline_interface_default"
                    )
                if "preset" not in applied:
                    candidate["preset"] = None
            updated = film_assumptions(candidate)
            ratio = updated["film_thickness_m"] / original["film_thickness_m"]
            if item["interface_kind"] in {"MA", "MS"}:
                ratio *= (
                    original["film_relative_permittivity"]
                    / updated["film_relative_permittivity"]
                )
            elif (
                updated["film_relative_permittivity"]
                != original["film_relative_permittivity"]
            ):
                raise ValueError(
                    "SA epsilon change requires stored normal/tangential integrals"
                )
            participation = item["original_participation"] * ratio
            surfaces.append(
                {
                    **item,
                    "assumptions": updated,
                    **(
                        {
                            "assumption_update": {
                                "scope": "group_override"
                                if group_update
                                else "interface_default",
                                "fields": sorted(applied),
                                "original_source": original["source"],
                                "original_preset": original["preset"],
                            }
                        }
                        if applied
                        else {}
                    ),
                    "participation": participation,
                    "inverse_q": inverse_quality_factor(
                        participation, updated["loss_tangent"]
                    ),
                }
            )
        rows.append({**row, "surface": surfaces})
    original_hash = result.provenance.get("original_result_sha256")
    if original_hash is None:
        original_hash = hashlib.sha256(
            json.dumps(
                result.to_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    return PalaceEprResult(
        tuple(rows),
        {
            **result.provenance,
            "original_result_sha256": original_hash,
            "film_analysis": {
                "schema_version": "scgsim.epr.film-analysis.v1",
                "surface_defaults": defaults,
                "group_overrides": overrides,
                "status": "derived_offline",
            },
        },
        _verified_source=result,
    )


def show_epr(result: PalaceEprResult, *, mode: int | None = None) -> Any:
    """Plot one mode with separate surface, bulk, and signed Palace port axes."""

    if not isinstance(result, PalaceEprResult):
        raise TypeError("result must be PalaceEprResult")

    return render(result, mode=mode)


def _source_provenance(
    index: Mapping[str, Any],
    output_prefix: str,
    request: Mapping[str, Any],
    result: ResolvedRunData,
) -> dict[str, Any]:
    """Record the exact config, index and solver-output identities behind extracted EPR."""
    source_inputs = {"config.json", "metadata/palace_index_map.json"}
    source_outputs = {
        f"{output_prefix}/{name}.csv"
        for name in (
            "eig",
            "domain-E",
            "surface-Q",
            "port-EPR",
            "surface-mask-Q",
            "surface-mask-energy",
        )
    }
    source_outputs.add(f"{output_prefix}/palace.json")
    palace_output = next(
        (
            item
            for item in result.returned_receipt.output_files
            if item.get("path") == f"{output_prefix}/palace.json"
        ),
        None,
    )
    palace_problem = result.provenance.palace_json.get("Problem")
    palace_iteration = (
        palace_problem.get("Iteration") if isinstance(palace_problem, Mapping) else None
    )
    provenance = {
        "handoff_id": result.returned_receipt.handoff_id,
        "route": result.route,
        "source": "verified_final_palace_tables"
        if output_prefix == "results/palace"
        else "verified_iteration_palace_tables",
        "analysis_request": request,
        "surface_scope": "recorded_native_MA_MS_SA; sidewalls_uncomputed_if_unselected",
        "palace_native_iteration": palace_iteration,
        "palace_json_sha256": (
            palace_output.get("sha256") if palace_output is not None else None
        ),
        **(
            {"surface_distance_bands": index["surface_distance_bands"]}
            if "surface_distance_bands" in index
            else {}
        ),
        "source_reference": {
            "receipt_schema": result.returned_receipt.schema,
            "receipt_timestamp_utc": result.returned_receipt.timestamp_utc,
            "solver_identity": result.returned_receipt.solver_identity,
            "input_files": [
                item
                for item in result.returned_receipt.input_hashes
                if item.get("path") in source_inputs
            ],
            "output_files": [
                item
                for item in result.returned_receipt.output_files
                if item.get("path") in source_outputs
            ],
        },
    }
    return provenance


@dataclass
class _ModeExtraction:
    """Aligned native mode tables and source selections used by EPR extraction."""

    domain_table: tuple[dict[str, Any], ...]
    domains: Mapping[int, Mapping[str, Any]]
    eig: tuple[dict[str, Any], ...]
    junctions: Mapping[int, Mapping[str, Any]]
    l0: float
    mask_energy_table: tuple[dict[str, Any] | None, ...]
    mask_q_table: tuple[dict[str, Any] | None, ...]
    masks_requested: bool
    port_table: tuple[dict[str, Any], ...]
    rows: list[dict[str, Any]]
    selected: Mapping[str, set[str]]
    surface_table: tuple[dict[str, Any], ...]
    surfaces: Mapping[int, Mapping[str, Any]]

    def extract(self) -> None:
        """Append mode rows in the native table order."""
        for (
            mode_row,
            surface_row,
            domain_row,
            port_row,
            mask_q_row,
            mask_energy_row,
        ) in zip(
            self.eig,
            self.surface_table,
            self.domain_table,
            self.port_table,
            self.mask_q_table,
            self.mask_energy_table,
            strict=True,
        ):
            mode = int(mode_row["m"])
            if any(
                int(item.get("m", item.get("i"))) != mode
                for item in (surface_row, domain_row, port_row)
            ):
                raise ValueError("Palace EPR mode tables do not align")
            if self.masks_requested and (
                cast(Mapping[str, Any], mask_q_row)["m"] != mode
                or cast(Mapping[str, Any], mask_energy_row)["m"] != mode
            ):
                raise ValueError("Palace EPR mask tables do not align with the mode")
            normalization = checked_number(
                domain_row["E_elec (J)"], "electric energy"
            ) + checked_number(domain_row["E_cap (J)"], "capacitive energy")
            normalization = checked_number(
                normalization, "normalization energy", positive=True
            )
            surface_items = _mode_surface_items(
                self.surfaces, self.selected, self.l0, surface_row, mask_q_row
            )
            bulk = [
                {
                    "domain_id": item["entry_name"],
                    "index": number,
                    "material": item["metadata"]["material"],
                    "participation": checked_number(
                        domain_row[f"p_elec[{number}]"], "bulk participation"
                    ),
                }
                for number, item in sorted(self.domains.items())
                if item["entry_name"] in self.selected["bulk_domain_ids"]
            ]
            port = [
                {
                    "port_name": item["port_name"],
                    "owner_semantic_ids": item["metadata"]["owner_semantic_ids"],
                    "index": number,
                    "participation": float(port_row[f"p[{number}]"]),
                    "signed_palace": True,
                }
                for number, item in sorted(self.junctions.items())
                if item["port_name"] in self.selected["port_names"]
            ]
            self.rows.append(
                {
                    "mode": mode,
                    "frequency_hz": checked_number(
                        mode_row["Re{f} (GHz)"], "frequency", positive=True
                    )
                    * 1e9,
                    "normalization_energy_j": normalization,
                    "surface": surface_items,
                    "bulk": bulk,
                    "junction": port,
                }
            )


def _selected_outputs(
    domains: Mapping[int, Mapping[str, Any]],
    index: Mapping[str, Any],
    junctions: Mapping[int, Mapping[str, Any]],
    surfaces: Mapping[int, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, set[str]]]:
    """Resolve explicit surface, bulk and port selections against the written index."""
    request = index.get(
        "epr_request",
        {
            "schema_version": "scgsim.palace.epr-selection.v1",
            "surface_interfaces": None,
            "bulk_domain_ids": None,
            "port_names": None,
        },
    )
    if (
        not isinstance(request, Mapping)
        or set(request)
        != {"schema_version", "surface_interfaces", "bulk_domain_ids", "port_names"}
        or request["schema_version"] != "scgsim.palace.epr-selection.v1"
    ):
        raise ValueError("Palace EPR selection payload is invalid")
    available = {
        "surface_interfaces": {
            item["metadata"]["interface_type"] for item in surfaces.values()
        },
        "bulk_domain_ids": {item["entry_name"] for item in domains.values()},
        "port_names": {item["port_name"] for item in junctions.values()},
    }
    selected: dict[str, set[str]] = {}
    for name, configured in available.items():
        values = request[name]
        if values is None:
            selected[name] = configured
        elif (
            not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
            or not set(values) <= configured
        ):
            raise ValueError(f"Palace EPR selection has unknown or invalid {name}")
        else:
            selected[name] = set(values)
    return request, selected


def _mode_surface_items(
    surfaces: Mapping[int, Mapping[str, Any]],
    selected: Mapping[str, set[str]],
    l0: float,
    surface_row: Mapping[str, Any],
    mask_q_row: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Bind one mode surface participation to recorded film assumptions."""
    surface_items: list[dict[str, Any]] = []
    for number, entry in sorted(surfaces.items()):
        metadata = entry["metadata"]
        if metadata["interface_type"] not in selected["surface_interfaces"]:
            continue
        spec = entry["epr_spec"]
        native_thickness = checked_number(
            spec["thickness"], "native film thickness", positive=True
        )
        recorded_l0 = checked_number(
            spec.get("model_l0_m", l0), "film Model.L0", positive=True
        )
        if recorded_l0 != l0:
            raise ValueError("film Model.L0 disagrees with resolved config")
        physical = checked_number(
            spec.get("film_thickness_m", native_thickness * l0),
            "SI film thickness",
            positive=True,
        )
        if not math.isclose(physical, native_thickness * l0, rel_tol=1e-12):
            raise ValueError("film SI/native thickness mismatch")
        recorded_loss = spec.get("loss_tangent")
        historical_ambiguous_zero = (
            spec.get("schema_version") != "scgsim.palace.surface-film.v2"
            and recorded_loss == 0
        )
        loss = None if historical_ambiguous_zero else recorded_loss
        assumption = film_assumptions(
            {
                "film_thickness_m": physical,
                "film_relative_permittivity": spec["permittivity"],
                "loss_tangent": loss,
                "source": spec.get(
                    "source",
                    "historical_ambiguous_zero"
                    if historical_ambiguous_zero
                    else "historical",
                ),
                "preset": spec.get("preset"),
            }
        )
        mask = entry.get("mask")
        distance_band = (
            surface_distance_band(metadata["surface_distance_band"])
            if "surface_distance_band" in metadata
            else None
        )
        if distance_band is not None and mask is not None:
            raise ValueError("Palace geometric bands cannot use fork-only mask rows")
        participation = checked_number(
            (
                cast(Mapping[str, Any], mask_q_row)[f"p_surf_mask[{number}]"]
                if mask is not None
                else surface_row[f"p_surf[{number}]"]
            ),
            "surface participation",
        )
        surface_items.append(
            {
                "group_id": f"palace_surface_{entry.get('baseline_index', number)}",
                "index": number,
                "baseline_index": entry.get("baseline_index", number),
                "owner_semantic_ids": metadata["owner_semantic_ids"],
                "interface_kind": metadata["interface_type"],
                "field_side": metadata["face_kind"],
                "surface_id": metadata["surface_id"],
                "members": metadata["source_provenance"],
                **(
                    {"attributes": list(entry["attributes"])}
                    if "attributes" in entry
                    else {}
                ),
                "evaluation_kind": "geometric_band"
                if distance_band is not None
                else "requested_margin"
                if mask is not None
                else "unmasked_baseline",
                "margin_um": (
                    None
                    if distance_band is not None
                    else float(mask["margin_um"])
                    if mask is not None
                    else 0.0
                ),
                "margin_index": mask["margin_index"] if mask is not None else None,
                **(
                    {"surface_distance_band": distance_band}
                    if distance_band is not None
                    else {}
                ),
                "original_participation": participation,
                "participation": participation,
                "original_assumptions": assumption,
                "assumptions": assumption,
                "recorded_native_assumptions": {
                    "schema_version": spec.get("schema_version"),
                    "thickness_model_l0_units": native_thickness,
                    "model_l0_m": l0,
                    "relative_permittivity": spec["permittivity"],
                    "loss_tangent": recorded_loss,
                    "source": spec.get("source"),
                    "preset": spec.get("preset"),
                },
                "loss_provenance": (
                    "historical_ambiguous_zero"
                    if historical_ambiguous_zero
                    else "recorded"
                ),
                "inverse_q": inverse_quality_factor(participation, loss),
            }
        )
    return surface_items
