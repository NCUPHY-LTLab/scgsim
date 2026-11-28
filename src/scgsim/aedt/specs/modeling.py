"""The ``modeling`` selector of an AEDT request: ``"solid"`` or ``"thin_film"``, chosen explicitly.

The choice picks the SGB route AEDT builds the plan's model on (``thin_film`` Route A, ``solid`` Route C,
issue #67 decision 7); the physical interpretation behind it belongs to the geometry stage.
"""

from __future__ import annotations

from typing import Any, Literal

Modeling = Literal["solid", "thin_film"]


def modeling_from_value(value: Any) -> Modeling:
    """Validate one explicit solid or thin-film modeling choice."""
    if value not in {"solid", "thin_film"}:
        raise ValueError("modeling must be explicitly 'solid' or 'thin_film'")
    return value
