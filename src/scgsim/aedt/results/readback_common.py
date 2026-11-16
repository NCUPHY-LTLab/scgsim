"""Shared strict offline checks for native readback identities."""

from __future__ import annotations

from typing import Any

from scgsim.aedt.preparation.sgb_model import PreparedSgbModel
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard


def positive_unique_ids(value: Any, count: int) -> bool:
    """Check a native identity list for the expected unique positive IDs."""
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
        and len(set(value)) == count
    )


def validate_imported_sgb_model(
    receipt: dict[str, Any], model: PreparedSgbModel
) -> None:
    """Refuse a receipt whose geometry is not ``model`` imported: its identity and the guard's solids."""
    geometry = receipt.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("model_identity_sha256") != model.identity_sha256
    ):
        raise RuntimeError("the receipt's AEDT geometry is not the request's SGB model")
    guard = read_step_guard(model.path("guard"))
    expected = {solid["solid"]: solid["aedt_name"] for solid in guard["solids"]}
    if geometry.get("objects") != expected:
        raise RuntimeError("the receipt's imported objects are not the guard's solids")
