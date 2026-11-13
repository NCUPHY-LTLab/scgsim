"""Classify valid interface material pairs and preserve owner orientation."""

from __future__ import annotations

from scgsim.refusal import refusal

from .materials import is_supported_material_kind, is_vacuum_material_kind


def conductor_solution_interface_kind(solution_material_kind: str) -> str:
    """Return ``MA`` for vacuum and ``MS`` for other supported solution material."""
    if not is_supported_material_kind(solution_material_kind):
        raise ValueError(
            refusal(
                "solution material kind must be vacuum, dielectric, or conductor",
                found=solution_material_kind,
            )
        )
    return "MA" if is_vacuum_material_kind(solution_material_kind) else "MS"


def solution_interface_kind(lower_material_kind: str, upper_material_kind: str) -> str:
    """Return ``AA`` for two vacua, ``SA`` for one, or ``SS`` for neither."""
    if not is_supported_material_kind(
        lower_material_kind
    ) or not is_supported_material_kind(upper_material_kind):
        raise ValueError(
            refusal(
                "solution material kinds must be vacuum, dielectric, or conductor",
                found={"lower": lower_material_kind, "upper": upper_material_kind},
            )
        )
    lower_vacuum = is_vacuum_material_kind(lower_material_kind)
    upper_vacuum = is_vacuum_material_kind(upper_material_kind)
    if lower_vacuum or upper_vacuum:
        # Keep the existing labels for the two vacuum-containing branches.
        return "AA" if lower_vacuum and upper_vacuum else "SA"
    return "SS"


def solution_interface_owner_ids(
    kind: str,
    lower_id: str,
    upper_id: str,
    *,
    lower_is_vacuum: bool,
) -> tuple[str, str]:
    """Return the established owner order, reversing vacuum-lower SA pairs."""
    if kind == "SA" and lower_is_vacuum:
        return (upper_id, lower_id)
    return (lower_id, upper_id)
