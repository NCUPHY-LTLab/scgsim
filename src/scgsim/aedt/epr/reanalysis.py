"""Offline reassessment of film assumptions on a stored EPR result.

Owns: recomputing thin-film energy, participation and 1/Q of the surface rows of
an `EprResult` when the film thickness, permittivity or loss tangent assumptions
change. It never reopens AEDT.
Does not own: producing the result (`analysis.py`) or the energy formulas
(`analysis._surface_energy_j`).

The film thickness, permittivity and loss tangent are the user's assumptions; the
layout does not determine them (docs/eigenmode-epr.qmd). A surface contribution
to the loss is 1/Q = p * tan(delta), with p the surface participation ratio.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from scgsim.aedt.epr.models import (
    EprResult,
)
from scgsim.aedt.epr.selection import finite
from scgsim.aedt.epr.surface_epr import surface_energy_j
from scgsim.semantics.epr import film_assumptions, inverse_quality_factor


def reanalyze_epr(
    result: EprResult,
    *,
    surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> EprResult:
    """Recompute film estimates from one resolved result; never reopen AEDT.

    `surface_defaults` sets assumptions per interface kind (MA, MS or SA);
    `group_overrides` sets them for single surface groups and wins over the
    defaults. Rows that are not complete are kept unchanged.
    """

    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    defaults = _film_updates(surface_defaults, name="surface_defaults")
    if set(defaults) - {"MA", "MS", "SA"}:
        raise ValueError("surface_defaults supports only MA, MS, and SA")
    overrides = _film_updates(group_overrides, name="group_overrides")
    known_groups = {
        str(surface["group_id"])
        for row in result.rows
        if row["status"] == "complete"
        for surface in row["surface_contributions"]
        if "group_id" in surface
    }
    if set(overrides) - known_groups:
        raise ValueError("group_overrides contains an unknown original group ID")
    rows: list[Mapping[str, Any]] = []
    for row in result.rows:
        if row["status"] != "complete":
            rows.append(row)
            continue
        normalization = finite(row["normalization_energy_j"], "normalization energy")
        if normalization <= 0:
            raise ValueError("normalization energy must be positive")
        surfaces = [
            _reanalyzed_surface(surface, normalization, defaults, overrides)
            for surface in row["surface_contributions"]
        ]
        rows.append(
            {
                **row,
                "schema_version": "scgsim.aedt.epr-mode-energy.v2",
                "surface_contributions": surfaces,
            }
        )
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
    return EprResult(
        result_kind=result.result_kind,
        setup_name=result.setup_name,
        rows=tuple(rows),
        provenance=provenance,
        _verified_source=result,
    )


def _film_updates(
    updates: Mapping[str, Mapping[str, Any]] | None, *, name: str
) -> dict[str, dict[str, Any]]:
    """Validate partial film assumptions keyed by interface kind or group id."""

    if updates is None:
        return {}
    if not isinstance(updates, Mapping) or any(
        not isinstance(key, str) or not key for key in updates
    ):
        raise TypeError(f"{name} must be a non-empty-string-keyed mapping")
    return {
        key: film_assumptions(value, partial=True) for key, value in updates.items()
    }


def _surface_original_assumptions(surface: Mapping[str, Any]) -> dict[str, Any]:
    """The assumptions the result was first computed with.

    Older records may carry only thickness and permittivity; loss tangent, source
    and preset are then unknown (None), not zero.
    """

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


def _reanalyzed_surface(
    surface: Mapping[str, Any],
    normalization: float,
    defaults: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the interface default and group override to one surface row."""

    original = _surface_original_assumptions(surface)
    current = film_assumptions(surface.get("assumptions", original), allow_unknown=True)
    interface_update = defaults.get(surface["interface_kind"], {})
    group_update = overrides.get(surface.get("group_id"), {})
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
        # Label an unlabelled change so it can be told apart from the original.
        if "source" not in applied:
            candidate["source"] = (
                "offline_group_override"
                if group_update
                else "offline_interface_default"
            )
        if "preset" not in applied:
            candidate["preset"] = None
    updated = film_assumptions(
        candidate,
        allow_unknown=True,
    )
    original_participation = finite(
        surface.get("original_participation", surface["participation"]),
        "original participation",
    )
    original_energy = finite(
        surface.get("original_energy_j", surface["energy_j"]),
        "original film energy",
    )
    # Only thickness and permittivity change the energy. A new loss tangent
    # changes 1/Q alone.
    changed_coefficient = any(
        updated[name] != original[name]
        for name in ("film_thickness_m", "film_relative_permittivity")
    )
    if changed_coefficient:
        participation, energy = _recombined_film_energy(
            surface,
            original,
            updated,
            normalization,
            (original_participation, original_energy),
        )
    else:
        participation, energy = original_participation, original_energy
    return {
        **surface,
        "original_assumptions": original,
        "original_participation": original_participation,
        "original_energy_j": original_energy,
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
        "energy_j": energy,
        "participation": participation,
        "inverse_q": inverse_quality_factor(participation, updated["loss_tangent"]),
    }


def _recombined_film_energy(
    surface: Mapping[str, Any],
    original: Mapping[str, Any],
    updated: Mapping[str, Any],
    normalization: float,
    original_values: tuple[float, float],
) -> tuple[float, float]:
    """Return (participation, energy) for changed film thickness or permittivity.

    With the stored field integrals the energy is recomputed exactly. Historical
    per-binding rows without them are rescaled instead, which is exact only where
    the energy is proportional to the changed coefficient.
    """

    original_participation, original_energy = original_values
    if (
        original["film_thickness_m"] is None
        or original["film_relative_permittivity"] is None
    ):
        raise ValueError("historical film coefficients are unavailable for reanalysis")
    if (
        updated["film_thickness_m"] is None
        or updated["film_relative_permittivity"] is None
    ):
        raise ValueError("film coefficients cannot be cleared")
    recombination_terms = {
        "normal_integral_v2",
        "tangential_integral_v2",
    }
    if surface["interface_kind"] == "MS":
        recombination_terms.add("substrate_relative_permittivity")
    if recombination_terms <= set(surface):
        energy = surface_energy_j(
            surface["interface_kind"],
            updated["film_thickness_m"],
            updated["film_relative_permittivity"],
            finite(
                surface.get("substrate_relative_permittivity", 1.0),
                "substrate epsilon",
            ),
            finite(surface["normal_integral_v2"], "normal integral"),
            finite(surface["tangential_integral_v2"], "tangential integral"),
        )
        return energy / normalization, energy
    if "group_id" in surface:
        raise ValueError("grouped surface result lacks original field integrals")
    # Historical per-binding records may not retain separate field integrals.
    if (
        surface["interface_kind"] == "SA"
        and updated["film_relative_permittivity"]
        != original["film_relative_permittivity"]
    ):
        raise ValueError(
            "SA epsilon change needs stored normal and tangential integrals"
        )
    # MA and MS energy scale as t / eps_f (docs/specs/aedt-runtime.qmd); SA scales
    # with t only when eps_f is unchanged, which the check above guarantees.
    ratio = updated["film_thickness_m"] / original["film_thickness_m"]
    if surface["interface_kind"] in {"MA", "MS"}:
        ratio *= (
            original["film_relative_permittivity"]
            / updated["film_relative_permittivity"]
        )
    return original_participation * ratio, original_energy * ratio
