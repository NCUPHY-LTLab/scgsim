"""Extract verified mode EPR and apply detached film-assumption reanalysis.

EPR means energy participation ratio; native integration and solving stay external."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_index_map import (
    DIELECTRIC_SECTION,
    ENERGY_SECTION,
    LUMPED_PORT_SECTION,
)
from scgsim.contracts.palace_run_folder import (
    CONFIG,
    EIGENMODE,
    EPR_SELECTION_SCHEMA,
    INDEX_MAP,
    NATIVE_METADATA_NAME,
    OUTPUT_DIR,
    confined_path,
    iteration_dir,
    sha256,
    table_name,
)
from scgsim.palace.results.artifacts import refusal
from scgsim.palace.results.epr_records import (
    PalaceEprResult,
    checked_number,
)
from scgsim.palace.results.resolved_records import ResolvedRunData
from scgsim.palace.results.surface_contract import entry_rows
from scgsim.palace.results.tables import (
    count_index_entries,
    output_families,
    read_result_table,
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
    """Read one sealed native iteration with the table validators of ``results.tables``."""

    if type(iteration) is not int or iteration <= 0:
        raise ValueError(
            refusal("palace_iteration must be a positive integer", found=iteration)
        )
    records = {item["path"]: item for item in result.returned_receipt.output_files}
    prefixes = set()
    for path in records:
        parts = Path(path).parts
        if (
            len(parts) != 4
            or parts[:2] != Path(OUTPUT_DIR).parts
            or parts[3] != NATIVE_METADATA_NAME
        ):
            continue
        if iteration_dir(parts[2]) == iteration:
            prefixes.add("/".join(parts[:3]))
    if len(prefixes) != 1:
        raise ValueError(
            refusal(
                f"requested native iteration {iteration} lacks one receipt-bound directory",
                found=sorted(prefixes),
            )
        )
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

    native = json.loads(verified(NATIVE_METADATA_NAME).read_text(encoding="utf-8"))
    if (
        not isinstance(native, dict)
        or not isinstance(native.get("Problem"), dict)
        or native["Problem"].get("Iteration") != iteration
    ):
        raise ValueError(
            refusal(
                f"native iteration metadata in {prefix}/{NATIVE_METADATA_NAME} disagrees with selected directory",
                found=native.get("Problem", {}).get("Iteration")
                if isinstance(native, dict) and isinstance(native.get("Problem"), dict)
                else None,
                expected=iteration,
            )
        )
    counts = count_index_entries(result.provenance.index_map)
    tables = {}
    for family in output_families(result.problem, counts):
        tables[family] = read_result_table(
            verified(table_name(family)),
            result.problem,
            counts,
            result.provenance.config,
        )
    return replace(
        result, tables=tables, provenance=replace(result.provenance, palace_json=native)
    ), prefix


def epr_result(
    result: ResolvedRunData, *, palace_iteration: int | None = None
) -> PalaceEprResult:
    """Extract final or explicitly selected sealed-iteration Eigenmode EPR."""

    if not isinstance(result, ResolvedRunData) or result.problem != EIGENMODE.name:
        raise TypeError(
            refusal(
                "epr_result requires a resolved Palace Eigenmode run",
                found=getattr(result, "problem", type(result).__name__),
                expected=EIGENMODE.name,
            )
        )
    output_prefix = OUTPUT_DIR
    if palace_iteration is not None:
        result, output_prefix = _iteration_result(result, palace_iteration)
    index = result.provenance.index_map
    config = result.provenance.config
    l0 = checked_number(config["Model"]["L0"], "Model.L0", positive=True)
    entries = index["entries"]
    surfaces = {
        item["index"]: item for item in entries if item["section"] == DIELECTRIC_SECTION
    }
    domains = {
        item["index"]: item for item in entries if item["section"] == ENERGY_SECTION
    }
    junctions = {
        item["index"]: item
        for item in entries
        if item["section"] == LUMPED_PORT_SECTION
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
    rows: list[dict[str, Any]] = []
    _ModeExtraction(
        domain_table,
        domains,
        eig,
        junctions,
        l0,
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
        raise TypeError(
            f"reanalyze_epr requires a PalaceEprResult; got {type(result).__name__}."
        )
    updates = _FilmUpdates.checked(result, surface_defaults, group_overrides)
    rows = tuple(
        {
            **row,
            "surface": [_reanalyzed_surface(item, updates) for item in row["surface"]],
        }
        for row in result.rows
    )
    return PalaceEprResult(
        rows, _reanalysis_provenance(result, updates), _verified_source=result
    )


@dataclass(frozen=True)
class _FilmUpdates:
    """The checked film changes of one reanalysis: by interface type, and by group over those."""

    defaults: dict[str, dict[str, Any]]
    overrides: dict[str, dict[str, Any]]

    @classmethod
    def checked(
        cls,
        result: PalaceEprResult,
        surface_defaults: Mapping[str, Mapping[str, Any]] | None,
        group_overrides: Mapping[str, Mapping[str, Any]] | None,
    ) -> _FilmUpdates:
        """Read both mappings as partial film snapshots, refusing an unknown interface type or group."""
        defaults = {
            key: film_assumptions(value, partial=True)
            for key, value in (surface_defaults or {}).items()
        }
        overrides = {
            key: film_assumptions(value, partial=True)
            for key, value in (group_overrides or {}).items()
        }
        unknown_types = set(defaults) - {"MA", "MS", "SA"}
        if unknown_types:
            raise ValueError(
                f"surface_defaults supports only MA, MS, and SA; got {sorted(unknown_types)}."
            )
        known = {item["group_id"] for row in result.rows for item in row["surface"]}
        unknown_groups = set(overrides) - known
        if unknown_groups:
            raise ValueError(
                f"group_overrides contains an unknown group: {sorted(unknown_groups)}; "
                f"this result's groups are {sorted(known)}."
            )
        return cls(defaults, overrides)


def _reanalyzed_surface(
    item: Mapping[str, Any], updates: _FilmUpdates
) -> dict[str, Any]:
    """One surface row with its film snapshot updated and its participation and 1/Q rescaled."""
    original = film_assumptions(item["original_assumptions"])
    current = film_assumptions(item["assumptions"])
    group_update = updates.overrides.get(item["group_id"], {})
    applied = {**updates.defaults.get(item["interface_kind"], {}), **group_update}
    candidate = {**current, **applied}
    if any(
        name in applied
        for name in ("film_thickness_m", "film_relative_permittivity", "loss_tangent")
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
    participation = item["original_participation"] * _participation_ratio(
        item, original, updated
    )
    return {
        **item,
        "assumptions": updated,
        **(
            {
                "assumption_update": {
                    "scope": "group_override" if group_update else "interface_default",
                    "fields": sorted(applied),
                    "original_source": original["source"],
                    "original_preset": original["preset"],
                }
            }
            if applied
            else {}
        ),
        "participation": participation,
        "inverse_q": inverse_quality_factor(participation, updated["loss_tangent"]),
    }


def _participation_ratio(
    item: Mapping[str, Any], original: Mapping[str, Any], updated: Mapping[str, Any]
) -> float:
    """Scale a thin-film participation by thickness, and for MA and MS by the inverse film permittivity.

    An SA film's participation splits into normal and tangential field integrals that the result does not
    store, so its permittivity cannot change offline.
    """
    ratio = updated["film_thickness_m"] / original["film_thickness_m"]
    if item["interface_kind"] in {"MA", "MS"}:
        ratio *= (
            original["film_relative_permittivity"]
            / updated["film_relative_permittivity"]
        )
    elif (
        updated["film_relative_permittivity"] != original["film_relative_permittivity"]
    ):
        raise ValueError(
            "SA epsilon change requires stored normal/tangential integrals: surface "
            f"{item['name']!r} (group {item['group_id']!r}) is SA, whose result stores no such "
            "integrals; change only film_thickness_m or loss_tangent for SA surfaces."
        )
    return ratio


def _reanalysis_provenance(
    result: PalaceEprResult, updates: _FilmUpdates
) -> dict[str, Any]:
    """The source provenance with the hash of the first, unreanalyzed result and the applied film change."""
    original_hash = result.provenance.get("original_result_sha256")
    if original_hash is None:
        original_hash = hashlib.sha256(
            json.dumps(
                result.to_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    return {
        **result.provenance,
        "original_result_sha256": original_hash,
        "film_analysis": {
            "schema_version": "scgsim.epr.film-analysis.v1",
            "surface_defaults": updates.defaults,
            "group_overrides": updates.overrides,
            "status": "derived_offline",
        },
    }


def _source_provenance(
    index: Mapping[str, Any],
    output_prefix: str,
    request: Mapping[str, Any],
    result: ResolvedRunData,
) -> dict[str, Any]:
    """Record the exact config, index and solver-output identities behind extracted EPR."""
    source_inputs = {CONFIG, INDEX_MAP}
    source_outputs = {
        f"{output_prefix}/{table_name(family)}"
        for family in ("eig", "domain-E", "surface-Q", "port-EPR")
    }
    source_outputs.add(f"{output_prefix}/{NATIVE_METADATA_NAME}")
    palace_output = next(
        (
            item
            for item in result.returned_receipt.output_files
            if item.get("path") == f"{output_prefix}/{NATIVE_METADATA_NAME}"
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
        if output_prefix == OUTPUT_DIR
        else "verified_iteration_palace_tables",
        "analysis_request": request,
        "surface_scope": "recorded_native_MA_MS_SA; sidewalls_uncomputed_if_unselected",
        "palace_native_iteration": palace_iteration,
        "palace_json_sha256": (
            palace_output.get("sha256") if palace_output is not None else None
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
    port_table: tuple[dict[str, Any], ...]
    rows: list[dict[str, Any]]
    selected: Mapping[str, set[str]]
    surface_table: tuple[dict[str, Any], ...]
    surfaces: Mapping[int, Mapping[str, Any]]

    def extract(self) -> None:
        """Append mode rows in the native table order."""
        for mode_row, surface_row, domain_row, port_row in zip(
            self.eig,
            self.surface_table,
            self.domain_table,
            self.port_table,
            strict=True,
        ):
            mode = int(mode_row["m"])
            if any(
                _row_index(row) != mode for row in (surface_row, domain_row, port_row)
            ):
                raise ValueError(
                    refusal(
                        "Palace EPR mode tables do not align",
                        found={
                            "surface-Q": _row_index(surface_row),
                            "domain-E": _row_index(domain_row),
                            "port-EPR": _row_index(port_row),
                        },
                        expected=mode,
                    )
                )
            normalization = checked_number(
                domain_row["E_elec (J)"], "electric energy"
            ) + checked_number(domain_row["E_cap (J)"], "capacitive energy")
            normalization = checked_number(
                normalization, "normalization energy", positive=True
            )
            surface_items = _mode_surface_items(
                self.surfaces, self.selected, self.l0, surface_row
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
                    "host_entity_ids": item["metadata"]["host_entity_ids"],
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


def _row_index(row: Mapping[str, Any]) -> int:
    """The mode (``m``) or excitation (``i``) a native table row belongs to."""
    return int(row["m"] if "m" in row else row["i"])


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
            "schema_version": EPR_SELECTION_SCHEMA,
            "surface_interfaces": None,
            "bulk_domain_ids": None,
            "port_names": None,
        },
    )
    if (
        not isinstance(request, Mapping)
        or set(request)
        != {"schema_version", "surface_interfaces", "bulk_domain_ids", "port_names"}
        or request["schema_version"] != EPR_SELECTION_SCHEMA
    ):
        raise ValueError(
            refusal(
                f"Palace EPR selection payload in {INDEX_MAP} is invalid",
                found=request,
                expected=f"keys schema_version ({EPR_SELECTION_SCHEMA}), surface_interfaces, "
                "bulk_domain_ids and port_names",
            )
        )
    available = {
        "surface_interfaces": {
            entry_rows(item)[0]["interface_type"] for item in surfaces.values()
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
) -> list[dict[str, Any]]:
    """Bind one mode's surface participation to the recorded sidecar rows and film assumptions."""
    surface_items: list[dict[str, Any]] = []
    for number, entry in sorted(surfaces.items()):
        rows = entry_rows(entry)
        interface_type = rows[0]["interface_type"]
        if interface_type not in selected["surface_interfaces"]:
            continue
        film = _RecordedFilm.checked(entry["epr_spec"], l0, number)
        participation = checked_number(
            surface_row[f"p_surf[{number}]"], "surface participation"
        )
        surface_items.append(_surface_item(surfaces, number, rows, film, participation))
    return surface_items


@dataclass(frozen=True)
class _RecordedFilm:
    """The film one Surface-EPR entry was solved with: its native thickness, its loss and its SI snapshot."""

    spec: Mapping[str, Any]
    l0: float
    native_thickness: float
    assumption: dict[str, Any]

    @classmethod
    def checked(cls, spec: Mapping[str, Any], l0: float, number: int) -> _RecordedFilm:
        """Check the entry's recorded length unit and SI thickness against the config before reading the film."""
        native_thickness = checked_number(
            spec["thickness"], "native film thickness", positive=True
        )
        recorded_l0 = checked_number(
            spec.get("model_l0_m", l0), "film Model.L0", positive=True
        )
        if recorded_l0 != l0:
            raise ValueError(
                f"film Model.L0 disagrees with resolved config: Surface-EPR entry {number} records "
                f"{recorded_l0} m, config.json Model.L0 is {l0} m."
            )
        physical = checked_number(
            spec.get("film_thickness_m", native_thickness * l0),
            "SI film thickness",
            positive=True,
        )
        if not math.isclose(physical, native_thickness * l0, rel_tol=1e-12):
            raise ValueError(
                f"film SI/native thickness mismatch: Surface-EPR entry {number} records "
                f"{physical} m, but {native_thickness} x Model.L0 {l0} m is {native_thickness * l0} m."
            )
        assumption = film_assumptions(
            {
                "film_thickness_m": physical,
                "film_relative_permittivity": spec["permittivity"],
                "loss_tangent": spec.get("loss_tangent"),
                "source": spec.get("source", "recorded"),
                "preset": spec.get("preset"),
            }
        )
        return cls(spec, l0, native_thickness, assumption)

    def native_record(self) -> dict[str, Any]:
        """The film as the native spec recorded it, in Model.L0 units."""
        return {
            "schema_version": self.spec.get("schema_version"),
            "thickness_model_l0_units": self.native_thickness,
            "model_l0_m": self.l0,
            "relative_permittivity": self.spec["permittivity"],
            "loss_tangent": self.spec.get("loss_tangent"),
            "source": self.spec.get("source"),
            "preset": self.spec.get("preset"),
        }


def _surface_item(
    surfaces: Mapping[int, Mapping[str, Any]],
    number: int,
    rows: list[dict[str, Any]],
    film: _RecordedFilm,
    participation: float,
) -> dict[str, Any]:
    """One surface row of a mode: its identity from the whole surface, its margin, film and participation."""
    entry = surfaces[number]
    inset = entry.get("inset")
    # A margin row is named and attributed by the whole surface it is part of.
    whole = entry_rows(surfaces[entry["baseline_index"]])
    return {
        "group_id": f"palace_surface_{entry['baseline_index']}",
        "index": number,
        "baseline_index": entry["baseline_index"],
        "interface_kind": rows[0]["interface_type"],
        "name": surfaces[entry["baseline_index"]]["entry_name"],
        "rows": [row["row"] for row in rows],
        "face_kinds": sorted({row["face_kind"] for row in whole}),
        "entities": sorted({e for row in whole for e in row["entities"]}),
        "nets": sorted({row["net"] for row in whole if row["net"] is not None}),
        "attributes": list(entry["attributes"]),
        # Lineage membership is attribution only: it weights nothing.
        "lineage": {
            face: dict(owners)
            for row in rows
            for face, owners in row["lineage"]["membership"].items()
        },
        "evaluation_kind": "requested_margin"
        if inset is not None
        else "unmasked_baseline",
        "margin_um": 0.0 if inset is None else float(inset["margin_um"]),
        "margin_index": None if inset is None else inset["margin_index"],
        **({"bands": list(inset["bands"])} if inset is not None else {}),
        "original_participation": participation,
        "participation": participation,
        "original_assumptions": film.assumption,
        "assumptions": film.assumption,
        "recorded_native_assumptions": film.native_record(),
        "inverse_q": inverse_quality_factor(
            participation, film.spec.get("loss_tangent")
        ),
    }
