"""Validate shared film assumptions and calculate their inverse-Q contribution.

Solver adapters supply heterogeneous input mappings. Their nested values remain
``Any`` at this boundary, where each recognized field is checked and normalized.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast, overload

from scgsim.refusal import refusal

FILM_FIELDS: frozenset[str] = frozenset(
    {
        "film_thickness_m",
        "film_relative_permittivity",
        "loss_tangent",
        "source",
        "preset",
    }
)


class FilmAssumptions(TypedDict):
    """One complete SI film snapshot: thickness, relative permittivity, loss tangent and where they came from.

    An offline record may keep an unknown thickness or permittivity as None (``allow_unknown``).
    """

    film_thickness_m: float | None
    film_relative_permittivity: float | None
    loss_tangent: float | None
    source: str | None
    preset: str | None


class FilmUpdate(TypedDict, total=False):
    """A partial film update: only the fields of a ``FilmAssumptions`` it changes."""

    film_thickness_m: float | None
    film_relative_permittivity: float | None
    loss_tangent: float | None
    source: str | None
    preset: str | None


@overload
def film_assumptions(
    value: Mapping[str, Any],
    *,
    partial: Literal[False] = False,
    allow_unknown: bool = False,
) -> FilmAssumptions: ...


@overload
def film_assumptions(
    value: Mapping[str, Any], *, partial: Literal[True], allow_unknown: bool = False
) -> FilmUpdate: ...


def film_assumptions(
    value: Mapping[str, Any], *, partial: bool = False, allow_unknown: bool = False
) -> FilmAssumptions | FilmUpdate:
    """Validate a complete SI film snapshot or an allowed partial update.

    ``allow_unknown`` lets thickness and permittivity fields retain ``None`` in
    offline records; partial updates may omit fields altogether.
    """

    if not isinstance(value, Mapping) or set(value) - FILM_FIELDS:
        raise ValueError(
            refusal(
                "film assumptions contain unsupported fields",
                found=sorted(set(value) - FILM_FIELDS)
                if isinstance(value, Mapping)
                else type(value).__name__,
                expected=sorted(FILM_FIELDS),
            )
        )
    if not partial and set(value) != FILM_FIELDS:
        raise ValueError(
            refusal(
                "film assumptions require all snapshot fields",
                found=sorted(value),
                expected=sorted(FILM_FIELDS),
            )
        )
    result = dict(value)
    for name in ("film_thickness_m", "film_relative_permittivity"):
        if name not in result:
            continue
        number = result[name]
        if number is None and allow_unknown:
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(
                refusal(f"{name} must be finite and positive", found=number)
            )
        result[name] = float(number)
    if "loss_tangent" in result and result["loss_tangent"] is not None:
        number = result["loss_tangent"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError(
                refusal(
                    "loss_tangent must be finite and non-negative or None", found=number
                )
            )
        result["loss_tangent"] = float(number)
    for name in ("source", "preset"):
        if (
            name in result
            and result[name] is not None
            and (not isinstance(result[name], str) or not result[name].strip())
        ):
            raise ValueError(
                refusal(f"{name} must be non-empty text or None", found=result[name])
            )
    # Every field is checked above, so the mapping is the snapshot it claims to be.
    return cast(FilmUpdate, result) if partial else cast(FilmAssumptions, result)


def inverse_quality_factor(
    participation: float | None, loss_tangent: float | None
) -> float | None:
    """Return participation times loss tangent, or ``None`` for unknown inputs."""

    if participation is None or loss_tangent is None:
        return None
    if (
        isinstance(participation, bool)
        or not isinstance(participation, (int, float))
        or not math.isfinite(participation)
        or participation < 0
    ):
        raise ValueError(
            refusal(
                "participation must be finite and non-negative or None",
                found=participation,
            )
        )
    film_assumptions({"loss_tangent": loss_tangent}, partial=True)
    return float(participation) * float(loss_tangent)
