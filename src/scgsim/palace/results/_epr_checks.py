"""Validate the rows of a detached Palace EPR result.

A ``PalaceEprResult`` may come from a verified run or from a JSON file anyone edited,
so its constructor checks every row: mode numbers, positive energies, surface
identities, inverse Q against participation times loss tangent, and that every row of
one surface group agrees on what the group is. Nothing here reads a run directory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.semantics.epr import film_assumptions, inverse_quality_factor

INTERFACE_KINDS = {"MA", "MS", "SA"}  # metal-air, metal-substrate, substrate-air


_EVALUATION_KINDS = {"unmasked_baseline", "requested_margin"}


def checked_number(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'non-negative'}"
        )
    return float(value)


def check_row(
    row: Mapping[str, Any],
    seen_modes: set[int],
    group_sources: dict[str, tuple[Any, ...]],
) -> None:
    mode = row.get("mode")
    if type(mode) is not int or mode <= 0 or mode in seen_modes:
        raise ValueError("Palace EPR modes must be unique positive integers")
    seen_modes.add(mode)
    checked_number(row.get("frequency_hz"), "frequency_hz", positive=True)
    checked_number(
        row.get("normalization_energy_j"), "normalization_energy_j", positive=True
    )
    for name in ("surface", "bulk", "junction"):
        if not isinstance(row.get(name), (list, tuple)) or any(
            not isinstance(item, Mapping) for item in row[name]
        ):
            raise TypeError(f"Palace EPR {name} must contain mappings")
    for surface in row["surface"]:
        _check_surface(surface, group_sources)
    for bulk in row["bulk"]:
        if not isinstance(bulk.get("domain_id"), str) or not bulk["domain_id"]:
            raise ValueError("Palace EPR bulk domain_id is invalid")
        checked_number(bulk.get("participation"), "bulk participation")
    for port in row["junction"]:
        _check_junction(port)


def _check_surface(
    surface: Mapping[str, Any], group_sources: dict[str, tuple[Any, ...]]
) -> None:
    for name in (
        "group_id",
        "surface_id",
        "interface_kind",
        "field_side",
        "evaluation_kind",
    ):
        if not isinstance(surface.get(name), str) or not surface[name]:
            raise ValueError(f"Palace EPR surface {name} is missing")
    if (
        surface["interface_kind"] not in INTERFACE_KINDS
        or surface["evaluation_kind"] not in _EVALUATION_KINDS
    ):
        raise ValueError("Palace EPR surface identity is invalid")
    if not isinstance(surface.get("members"), Mapping):
        raise TypeError("Palace EPR source provenance must be a mapping")
    owners = surface.get("owner_semantic_ids")
    if (
        not isinstance(owners, (list, tuple))
        or not owners
        or any(not isinstance(owner, str) or not owner for owner in owners)
    ):
        raise ValueError("Palace EPR surface owners are invalid")
    checked_number(surface.get("margin_um"), "surface margin")
    checked_number(surface.get("original_participation"), "original participation")
    checked_number(surface.get("participation"), "participation")
    assumption = film_assumptions(surface.get("assumptions"), allow_unknown=True)
    original = film_assumptions(surface.get("original_assumptions"), allow_unknown=True)
    _check_inverse_q(surface, assumption["loss_tangent"])

    # Every row of one group must agree on what the group is.
    source = (
        surface["surface_id"],
        surface["interface_kind"],
        tuple(owners),
        tuple(sorted(original.items())),
    )
    if source != group_sources.setdefault(surface["group_id"], source):
        raise ValueError(
            "Palace EPR original group identity has conflicting provenance"
        )


def _check_inverse_q(surface: Mapping[str, Any], loss_tangent: float | None) -> None:
    expected_q = inverse_quality_factor(surface["participation"], loss_tangent)
    actual_q = surface.get("inverse_q")
    if (expected_q is None) != (actual_q is None) or (
        expected_q is not None
        and not math.isclose(
            checked_number(actual_q, "inverse_q"),
            expected_q,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Palace EPR inverse_q disagrees with participation and loss")


def _check_junction(port: Mapping[str, Any]) -> None:
    if not isinstance(port.get("port_name"), str) or not port["port_name"]:
        raise ValueError("Palace EPR port_name is invalid")
    value = port.get("participation")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Palace EPR signed port participation is invalid")


def check_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("handoff_id"), str
    ):
        raise ValueError("Palace EPR provenance lacks handoff identity")
