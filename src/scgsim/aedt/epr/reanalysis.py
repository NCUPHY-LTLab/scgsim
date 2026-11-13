"""Offline EPR film assumption reanalysis over stored result rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.epr.selection import finite_value
from scgsim.aedt.epr.surface import surface_energy_j
from scgsim.semantics.epr import (
    FilmAssumptions,
    FilmUpdate,
    film_assumptions,
    inverse_quality_factor,
)


def _surface_original_assumptions(surface: Mapping[str, Any]) -> FilmAssumptions:
    if "original_assumptions" in surface:
        return film_assumptions(surface["original_assumptions"], allow_unknown=True)
    if "assumptions" in surface:
        return film_assumptions(surface["assumptions"], allow_unknown=True)
    return film_assumptions(
        {
            "film_thickness_m": surface.get("film_thickness_m"),
            "film_relative_permittivity": surface.get("film_relative_permittivity"),
            "loss_tangent": None,
            "source": None,
            "preset": None,
        },
        allow_unknown=True,
    )


def _film_updates(
    updates: Mapping[str, Mapping[str, Any]] | None, *, name: str
) -> dict[str, FilmUpdate]:
    if updates is None:
        return {}
    if not isinstance(updates, Mapping) or any(
        not isinstance(key, str) or not key for key in updates
    ):
        raise TypeError(f"{name} must be a non-empty-string-keyed mapping")
    return {
        key: film_assumptions(value, partial=True) for key, value in updates.items()
    }


def reanalyze_epr(
    result: EprResult,
    *,
    surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> EprResult:
    """Recompute film estimates from one resolved result; never reopen AEDT."""
    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    defaults = _film_updates(surface_defaults, name="surface_defaults")
    if set(defaults) - {"MA", "MS", "SA"}:
        raise ValueError("surface_defaults supports only MA, MS, and SA")
    overrides = _film_updates(group_overrides, name="group_overrides")
    _validate_known_group_overrides(result, overrides)
    rows = _reanalyzed_rows(result, defaults, overrides)
    provenance = _reanalysis_provenance(result, defaults, overrides)
    return EprResult(
        result_kind=result.result_kind,
        setup_name=result.setup_name,
        rows=tuple(rows),
        provenance=provenance,
        _verified_source=result,
    )


def _validate_known_group_overrides(
    result: EprResult, overrides: Mapping[str, Mapping[str, Any]]
) -> None:
    """Reject overrides that do not name an original grouped contribution."""
    known_groups = {
        str(surface["group_id"])
        for row in result.rows
        if row["status"] == "complete"
        for surface in row["surface_contributions"]
        if "group_id" in surface
    }
    if set(overrides) - known_groups:
        raise ValueError("group_overrides contains an unknown original group ID")


def _reanalyzed_rows(
    result: EprResult,
    defaults: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Retain partial rows and reanalyze complete rows in their stored order."""
    rows: list[Mapping[str, Any]] = []
    for row in result.rows:
        if row["status"] != "complete":
            rows.append(row)
            continue
        normalization = finite_value(
            row["normalization_energy_j"], "normalization energy"
        )
        if normalization <= 0:
            raise ValueError("normalization energy must be positive")
        surfaces = [
            _reanalyze_surface(surface, defaults, overrides, normalization)
            for surface in row["surface_contributions"]
        ]
        rows.append(
            {
                **row,
                "schema_version": "scgsim.aedt.epr-mode-energy.v2",
                "surface_contributions": surfaces,
            }
        )
    return rows


def _reanalyze_surface(
    surface: Mapping[str, Any],
    defaults: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
    normalization: float,
) -> dict[str, Any]:
    """Apply one contribution's assumptions and recompute its saved energy."""
    original = _surface_original_assumptions(surface)
    current = film_assumptions(surface.get("assumptions", original), allow_unknown=True)
    interface_update = defaults.get(surface["interface_kind"], {})
    group_update = overrides.get(cast(str, surface.get("group_id")), {})
    applied = {**interface_update, **group_update}
    updated = _updated_assumptions(current, applied, group_update)
    original_participation = finite_value(
        surface.get("original_participation", surface["participation"]),
        "original participation",
    )
    original_energy = finite_value(
        surface.get("original_energy_j", surface["energy_j"]),
        "original film energy",
    )
    energy, participation = _recomputed_energy_and_participation(
        SurfaceReanalysisInputs(
            surface,
            original,
            updated,
            original_energy,
            original_participation,
            normalization,
        )
    )
    return {
        **surface,
        "original_assumptions": original,
        "original_participation": original_participation,
        "original_energy_j": original_energy,
        "assumptions": updated,
        **_assumption_update_record(original, applied, group_update),
        "energy_j": energy,
        "participation": participation,
        "inverse_q": inverse_quality_factor(participation, updated["loss_tangent"]),
    }


def _updated_assumptions(
    current: Mapping[str, Any],
    applied: Mapping[str, Any],
    group_update: Mapping[str, Any],
) -> FilmAssumptions:
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
    return film_assumptions(candidate, allow_unknown=True)


@dataclass(frozen=True)
class SurfaceReanalysisInputs:
    """Stored calculation terms used to recompute one surface contribution."""

    surface: Mapping[str, Any]
    original: Mapping[str, Any]
    updated: Mapping[str, Any]
    original_energy: float
    original_participation: float
    normalization: float


def _recomputed_energy_and_participation(
    inputs: SurfaceReanalysisInputs,
) -> tuple[float, float]:
    surface = inputs.surface
    original = inputs.original
    updated = inputs.updated
    original_energy = inputs.original_energy
    original_participation = inputs.original_participation
    normalization = inputs.normalization
    changed_coefficient = any(
        updated[name] != original[name]
        for name in ("film_thickness_m", "film_relative_permittivity")
    )
    if not changed_coefficient:
        return original_energy, original_participation
    _validate_reanalysis_coefficients(original, updated)
    recombination_terms = {"normal_integral_v2", "tangential_integral_v2"}
    if surface["interface_kind"] == "MS":
        recombination_terms.add("substrate_relative_permittivity")
    missing = sorted(recombination_terms - set(surface))
    if missing:
        raise ValueError(
            f"surface row lacks {missing} needed to recompute it; reanalyze a "
            "result of a current run, whose surface rows store them"
        )
    energy = surface_energy_j(
        surface["interface_kind"],
        updated["film_thickness_m"],
        updated["film_relative_permittivity"],
        finite_value(
            surface.get("substrate_relative_permittivity", 1.0),
            "substrate epsilon",
        ),
        (
            finite_value(surface["normal_integral_v2"], "normal integral"),
            finite_value(surface["tangential_integral_v2"], "tangential integral"),
        ),
    )
    return energy, energy / normalization


def _validate_reanalysis_coefficients(
    original: Mapping[str, Any], updated: Mapping[str, Any]
) -> None:
    if (
        original["film_thickness_m"] is None
        or original["film_relative_permittivity"] is None
    ):
        raise ValueError("stored film coefficients are unavailable for reanalysis")
    if (
        updated["film_thickness_m"] is None
        or updated["film_relative_permittivity"] is None
    ):
        raise ValueError("film coefficients cannot be cleared")


def _assumption_update_record(
    original: Mapping[str, Any],
    applied: Mapping[str, Any],
    group_update: Mapping[str, Any],
) -> dict[str, Any]:
    if not applied:
        return {}
    return {
        "assumption_update": {
            "scope": "group_override" if group_update else "interface_default",
            "fields": sorted(applied),
            "original_source": original["source"],
            "original_preset": original["preset"],
        }
    }


def _reanalysis_provenance(
    result: EprResult,
    defaults: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind offline overrides to the original result's stable digest."""
    original_digest = result.provenance.get("original_result_sha256")
    if original_digest is None:
        original_digest = hashlib.sha256(
            json.dumps(
                result.to_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    provenance = {
        **result.provenance,
        "original_result_sha256": original_digest,
        "film_analysis": {
            "schema_version": "scgsim.epr.film-analysis.v1",
            "surface_defaults": defaults,
            "group_overrides": overrides,
            "status": "derived_offline",
        },
    }
    return provenance
