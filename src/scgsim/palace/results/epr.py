"""Detached Palace EPR values, serialization, and offline reanalysis.

EPR is the energy participation ratio of one part (a surface film, a bulk domain or
a junction port) in one eigenmode. A surface film's inverse quality factor is its
participation times its loss tangent (``inverse_quality_factor``).

``epr_result`` turns a resolved Eigenmode run into a ``PalaceEprResult``: a small
value that no longer needs the run directory. ``reanalyze_epr`` re-weights the stored
surface participation with other film assumptions, offline.

Not here: reading and checking the run (``resolve.py``) and plotting
(``palace/presentation/epr.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

from scgsim.semantics.epr import film_assumptions, inverse_quality_factor

from ._epr_checks import (
    _INTERFACE_KINDS,
    _check_provenance,
    _check_row,
    _number,
)
from ._index_map import FILM_SPEC_SCHEMA
from .resolve import ResolvedPalaceResult

_PAYLOAD_SCHEMA = "scgsim.palace.epr-result.v1"
_SELECTION_SCHEMA = "scgsim.palace.epr-selection.v1"
_SELECTION_FIELDS = ("surface_interfaces", "bulk_domain_ids", "port_names")


class _SurfaceItem(TypedDict):
    """One surface film contribution to one mode, as stored in a row."""

    group_id: str
    index: int
    baseline_index: int
    owner_semantic_ids: list[str]
    interface_kind: str
    field_side: str
    surface_id: str
    members: dict[str, Any]
    evaluation_kind: str
    margin_um: float
    margin_index: int | None
    original_participation: float
    participation: float
    original_assumptions: dict[str, Any]
    assumptions: dict[str, Any]
    recorded_native_assumptions: dict[str, Any]
    loss_provenance: str
    inverse_q: float | None


class _BulkItem(TypedDict):
    """One bulk-domain contribution to one mode."""

    domain_id: str
    index: int
    material: dict[str, Any]
    participation: float


class _JunctionItem(TypedDict):
    """One lumped-port (junction) contribution to one mode."""

    port_name: str
    owner_semantic_ids: list[str]
    index: int
    participation: float
    signed_palace: bool


class _ModeRow(TypedDict):
    """Everything stored for one eigenmode."""

    mode: int
    frequency_hz: float
    normalization_energy_j: float
    surface: list[_SurfaceItem]
    bulk: list[_BulkItem]
    junction: list[_JunctionItem]


# ------------------------------------------------------- number and freeze helpers


def _plain_epr_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_epr_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_epr_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported EPR value {type(value).__name__}")


def _freeze_coercing_keys(value: Any, *, verified: Any = None) -> Any:
    """Make nested JSON read-only. A part already verified is reused as it is."""
    if verified is not None and value is verified:
        return verified
    if isinstance(value, Mapping):
        prior = verified if isinstance(verified, Mapping) else {}
        return MappingProxyType(
            {
                str(key): _freeze_coercing_keys(item, verified=prior.get(str(key)))
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        prior = verified if isinstance(verified, (tuple, list)) else ()
        return tuple(
            _freeze_coercing_keys(
                item, verified=prior[index] if index < len(prior) else None
            )
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported EPR value {type(value).__name__}")


# ------------------------------------------------------------- the stored result


@dataclass(frozen=True)
class PalaceEprResult:
    """Portable exact-mode snapshot without a native run or mutable Sim handle."""

    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    _verified_source: PalaceEprResult | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("Palace EPR requires non-empty row mappings")
        seen_modes: set[int] = set()
        group_sources: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            _check_row(row, seen_modes, group_sources)
        _check_provenance(self.provenance)
        if self._verified_source is not None and not isinstance(
            self._verified_source, PalaceEprResult
        ):
            raise TypeError("verified source must be a PalaceEprResult")
        self._freeze(rows)

    def _freeze(self, rows: tuple[Mapping[str, Any], ...]) -> None:
        """Replace rows and provenance by read-only copies.

        Parts that came unchanged from the verified source keep their identity, so
        ``reanalyze_epr`` does not copy what it did not touch.
        """
        source = self._verified_source
        prior_rows = source.rows if source is not None else ()
        object.__setattr__(
            self,
            "rows",
            tuple(
                _freeze_coercing_keys(
                    row, verified=prior_rows[index] if index < len(prior_rows) else None
                )
                for index, row in enumerate(rows)
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_coercing_keys(
                self.provenance,
                verified=source.provenance if source is not None else None,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        """The JSON-ready form that ``from_payload`` reads back."""
        return {
            "schema_version": _PAYLOAD_SCHEMA,
            "rows": _plain_epr_value(self.rows),
            "provenance": _plain_epr_value(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PalaceEprResult:
        """Rebuild a result from ``to_payload`` output, refusing any other shape."""
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "rows", "provenance"}
            or payload.get("schema_version") != _PAYLOAD_SCHEMA
        ):
            raise ValueError("Palace EPR payload is not canonical")
        return cls(tuple(payload["rows"]), payload["provenance"])


# ------------------------------------------------------------ row validation


# ----------------------------------------------------------------- public API


def resolve_epr_result(path: str | Path) -> PalaceEprResult:
    """Load a detached Palace EPR record without reading a solver directory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    return PalaceEprResult.from_payload(json.loads(source.read_text(encoding="utf-8")))


def epr_result(result: ResolvedPalaceResult) -> PalaceEprResult:
    """Extract compact verified final Eigenmode quantities and source bindings."""

    if not isinstance(result, ResolvedPalaceResult) or result.problem != "Eigenmode":
        raise TypeError("a resolved Palace Eigenmode run is required")
    index_map = result.provenance.index_map
    l0 = _number(result.provenance.config["Model"]["L0"], "Model.L0", positive=True)
    entries = index_map["entries"]
    surfaces = _entries_by_index(entries, "Boundaries.Postprocessing.Dielectric")
    domains = _entries_by_index(entries, "Domains.Postprocessing.Energy")
    junctions = _entries_by_index(entries, "Boundaries.LumpedPort")

    request = _read_selection_request(index_map)
    selected = _select_requested(request, surfaces, domains, junctions)
    tables = _mode_tables(result, surfaces, junctions)

    masks_requested = any("mask" in entry for entry in surfaces.values())
    rows: list[_ModeRow] = []
    for mode_row, surface_row, domain_row, port_row, mask_q_row, mask_energy_row in zip(
        *tables, strict=True
    ):
        mode = int(mode_row["m"])
        _check_tables_align(
            mode,
            (surface_row, domain_row, port_row),
            (mask_q_row, mask_energy_row),
            masks_requested=masks_requested,
        )
        # Participation is relative to the mode's electric plus capacitive energy.
        normalization = _number(domain_row["E_elec (J)"], "electric energy") + _number(
            domain_row["E_cap (J)"], "capacitive energy"
        )
        normalization = _number(normalization, "normalization energy", positive=True)
        surface = _surface_items(
            surfaces, selected["surface_interfaces"], l0, surface_row, mask_q_row
        )
        bulk = _bulk_items(domains, selected["bulk_domain_ids"], domain_row)
        junction = _junction_items(junctions, selected["port_names"], port_row)
        frequency_ghz = _number(mode_row["Re{f} (GHz)"], "frequency", positive=True)
        rows.append(
            {
                "mode": mode,
                "frequency_hz": frequency_ghz * 1e9,
                "normalization_energy_j": normalization,
                "surface": surface,
                "bulk": bulk,
                "junction": junction,
            }
        )
    return PalaceEprResult(tuple(rows), _build_provenance(result, request))


# ------------------------------------------------- reading the run into rows


def _entries_by_index(
    entries: Sequence[dict[str, Any]], section: str
) -> dict[int, dict[str, Any]]:
    return {item["index"]: item for item in entries if item["section"] == section}


def _read_selection_request(index_map: Mapping[str, Any]) -> dict[str, Any]:
    """Which surfaces, domains and ports the Eigenmode authoring asked to report.

    ``None`` for a field means all of them. An index map without the record, which
    Eigenmode authoring writes only when a selection was requested, means all.
    """
    request = index_map.get(
        "epr_request",
        {
            "schema_version": _SELECTION_SCHEMA,
            "surface_interfaces": None,
            "bulk_domain_ids": None,
            "port_names": None,
        },
    )
    if (
        not isinstance(request, Mapping)
        or set(request) != {"schema_version", *_SELECTION_FIELDS}
        or request["schema_version"] != _SELECTION_SCHEMA
    ):
        raise ValueError("Palace EPR selection payload is invalid")
    return dict(request)


def _select_requested(
    request: Mapping[str, Any],
    surfaces: Mapping[int, Mapping[str, Any]],
    domains: Mapping[int, Mapping[str, Any]],
    junctions: Mapping[int, Mapping[str, Any]],
) -> dict[str, set[str]]:
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
    return selected


def _mode_tables(
    result: ResolvedPalaceResult,
    surfaces: Mapping[int, Mapping[str, Any]],
    junctions: Mapping[int, Mapping[str, Any]],
) -> tuple[Sequence[Any], ...]:
    """The result tables, one row per mode each, in the order ``epr_result`` zips them.

    A table Palace did not write (no surfaces, no ports, no masks) is replaced by
    placeholder rows so the zip still has one entry per mode.
    """
    eig = result.tables["eig"].rows
    no_rows = tuple({"m": item["m"]} for item in eig)
    surface_table = result.tables["surface-Q"].rows if surfaces else no_rows
    domain_table = result.tables["domain-E"].rows
    port_table = result.tables["port-EPR"].rows if junctions else no_rows
    masks_requested = any("mask" in entry for entry in surfaces.values())
    if masks_requested:
        mask_q_table = result.tables["surface-mask-Q"].rows
        mask_energy_table = result.tables["surface-mask-energy"].rows
    else:
        mask_q_table = mask_energy_table = (None,) * len(eig)
    return (
        eig,
        surface_table,
        domain_table,
        port_table,
        mask_q_table,
        mask_energy_table,
    )


def _check_tables_align(
    mode: int,
    tables: tuple[Mapping[str, Any], ...],
    mask_tables: tuple[Mapping[str, Any] | None, ...],
    *,
    masks_requested: bool,
) -> None:
    # The surface, domain and port tables name their mode column "m" (Eigenmode) or "i".
    if any(int(item.get("m", item.get("i"))) != mode for item in tables):
        raise ValueError("Palace EPR mode tables do not align")
    if masks_requested and any(item["m"] != mode for item in mask_tables if item):
        raise ValueError("Palace EPR mask tables do not align with the mode")


def _surface_items(
    surfaces: Mapping[int, Mapping[str, Any]],
    selected_interfaces: set[str],
    l0: float,
    surface_row: Mapping[str, Any],
    mask_q_row: Mapping[str, Any] | None,
) -> list[_SurfaceItem]:
    items: list[_SurfaceItem] = []
    for number, entry in sorted(surfaces.items()):
        metadata = entry["metadata"]
        if metadata["interface_type"] not in selected_interfaces:
            continue
        spec = entry["epr_spec"]
        film = _read_film(spec, l0)
        mask = entry.get("mask")
        # A masked row reports its own participation; an unmasked one the baseline's.
        participation = _number(
            (
                mask_q_row[f"p_surf_mask[{number}]"]
                if mask is not None
                else surface_row[f"p_surf[{number}]"]
            ),
            "surface participation",
        )
        items.append(
            {
                "group_id": f"palace_surface_{entry.get('baseline_index', number)}",
                "index": number,
                "baseline_index": entry.get("baseline_index", number),
                "owner_semantic_ids": metadata["owner_semantic_ids"],
                "interface_kind": metadata["interface_type"],
                "field_side": metadata["face_kind"],
                "surface_id": metadata["surface_id"],
                "members": metadata["source_provenance"],
                "evaluation_kind": "requested_margin"
                if mask is not None
                else "unmasked_baseline",
                "margin_um": float(mask["margin_um"]) if mask is not None else 0.0,
                "margin_index": mask["margin_index"] if mask is not None else None,
                "original_participation": participation,
                "participation": participation,
                "original_assumptions": film.assumptions,
                "assumptions": film.assumptions,
                "recorded_native_assumptions": {
                    "schema_version": spec["schema_version"],
                    "thickness_model_l0_units": film.native_thickness,
                    "model_l0_m": l0,
                    "relative_permittivity": spec["permittivity"],
                    "loss_tangent": film.loss_tangent,
                    "source": spec["source"],
                    "preset": spec["preset"],
                },
                "loss_provenance": "recorded",
                "inverse_q": inverse_quality_factor(participation, film.loss_tangent),
            }
        )
    return items


@dataclass(frozen=True)
class _Film:
    """The recorded dielectric film of one surface, in SI and in Model.L0 units."""

    assumptions: dict[str, Any]
    native_thickness: float  # in Model.L0 units
    loss_tangent: float | None  # None: not specified (config.json got 0.0)


def _read_film(spec: Mapping[str, Any], l0: float) -> _Film:
    if spec.get("schema_version") != FILM_SPEC_SCHEMA:
        raise ValueError(
            "surface EPR film spec schema_version is "
            f"{spec.get('schema_version')!r}; supported: {FILM_SPEC_SCHEMA!r}."
        )
    native_thickness = _number(
        spec["thickness"], "native film thickness", positive=True
    )
    recorded_l0 = _number(spec["model_l0_m"], "film Model.L0", positive=True)
    if recorded_l0 != l0:
        raise ValueError("film Model.L0 disagrees with resolved config")
    physical = _number(spec["film_thickness_m"], "SI film thickness", positive=True)
    if not math.isclose(physical, native_thickness * l0, rel_tol=1e-12):
        raise ValueError("film SI/native thickness mismatch")
    loss = spec["loss_tangent"]
    assumptions = film_assumptions(
        {
            "film_thickness_m": physical,
            "film_relative_permittivity": spec["permittivity"],
            "loss_tangent": loss,
            "source": spec["source"],
            "preset": spec["preset"],
        }
    )
    return _Film(assumptions, native_thickness, loss)


def _bulk_items(
    domains: Mapping[int, Mapping[str, Any]],
    selected_ids: set[str],
    domain_row: Mapping[str, Any],
) -> list[_BulkItem]:
    return [
        {
            "domain_id": item["entry_name"],
            "index": number,
            "material": item["metadata"]["material"],
            "participation": _number(
                domain_row[f"p_elec[{number}]"], "bulk participation"
            ),
        }
        for number, item in sorted(domains.items())
        if item["entry_name"] in selected_ids
    ]


def _junction_items(
    junctions: Mapping[int, Mapping[str, Any]],
    selected_names: set[str],
    port_row: Mapping[str, Any],
) -> list[_JunctionItem]:
    return [
        {
            "port_name": item["port_name"],
            "owner_semantic_ids": item["metadata"]["owner_semantic_ids"],
            "index": number,
            "participation": float(port_row[f"p[{number}]"]),
            "signed_palace": True,
        }
        for number, item in sorted(junctions.items())
        if item["port_name"] in selected_names
    ]


def _build_provenance(
    result: ResolvedPalaceResult, request: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = result.returned_receipt
    source_inputs = {"config.json", "metadata/palace_index_map.json"}
    source_outputs = {
        f"results/palace/{name}.csv"
        for name in (
            "eig",
            "domain-E",
            "surface-Q",
            "port-EPR",
            "surface-mask-Q",
            "surface-mask-energy",
        )
    }
    return {
        "handoff_id": receipt.handoff_id,
        "route": result.route,
        "source": "verified_final_palace_tables",
        "analysis_request": request,
        "surface_scope": "recorded_native_MA_MS_SA; sidewalls_uncomputed_if_unselected",
        "source_reference": {
            "receipt_schema": receipt.schema,
            "receipt_timestamp_utc": receipt.timestamp_utc,
            "solver_identity": receipt.solver_identity,
            "input_files": [
                item
                for item in receipt.input_hashes
                if item.get("path") in source_inputs
            ],
            "output_files": [
                item
                for item in receipt.output_files
                if item.get("path") in source_outputs
            ],
        },
    }


# ------------------------------------------------------------ offline reanalysis


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
    if set(defaults) - _INTERFACE_KINDS:
        raise ValueError("surface_defaults supports only MA, MS, and SA")
    known = {item["group_id"] for row in result.rows for item in row["surface"]}
    if set(overrides) - known:
        raise ValueError("group_overrides contains an unknown group")
    rows = [
        {
            **row,
            "surface": [
                _reanalyzed_surface(item, defaults, overrides)
                for item in row["surface"]
            ],
        }
        for row in result.rows
    ]
    return PalaceEprResult(
        tuple(rows),
        {
            **result.provenance,
            "original_result_sha256": _original_result_hash(result),
            "film_analysis": {
                "schema_version": "scgsim.epr.film-analysis.v1",
                "surface_defaults": defaults,
                "group_overrides": overrides,
                "status": "derived_offline",
            },
        },
        _verified_source=result,
    )


def _reanalyzed_surface(
    item: Mapping[str, Any],
    defaults: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    original = film_assumptions(item["original_assumptions"])
    current = film_assumptions(item["assumptions"])
    interface_update = defaults.get(item["interface_kind"], {})
    group_update = overrides.get(item["group_id"], {})
    applied = {**interface_update, **group_update}
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
        item["interface_kind"], original, updated
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
    interface_kind: str, original: Mapping[str, Any], updated: Mapping[str, Any]
) -> float:
    """How a surface film's participation scales when its thickness or epsilon change.

    Participation scales with film thickness, and for MA and MS also with
    1/epsilon. TODO(human): why SA (substrate-air) is exempt from the 1/epsilon
    scaling; the error below only says the stored data lacks the integrals.
    """
    ratio = updated["film_thickness_m"] / original["film_thickness_m"]
    if interface_kind in {"MA", "MS"}:
        ratio *= (
            original["film_relative_permittivity"]
            / updated["film_relative_permittivity"]
        )
    elif (
        updated["film_relative_permittivity"] != original["film_relative_permittivity"]
    ):
        raise ValueError(
            "SA epsilon change requires stored normal/tangential integrals"
        )
    return ratio


def _original_result_hash(result: PalaceEprResult) -> str:
    """Hash of the result before any reanalysis, kept through repeated reanalysis."""
    recorded = result.provenance.get("original_result_sha256")
    if recorded is not None:
        return recorded
    return hashlib.sha256(
        json.dumps(result.to_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def show_epr(result: PalaceEprResult, *, mode: int | None = None) -> Any:
    """Plot one mode with separate surface, bulk, and signed Palace port axes."""

    if not isinstance(result, PalaceEprResult):
        raise TypeError("result must be PalaceEprResult")
    from ..presentation.epr import show_epr as render

    return render(result, mode=mode)
