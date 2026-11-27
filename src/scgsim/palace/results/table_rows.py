"""The row checks every native Palace table validator shares: numeric cells, Palace's infinite Q, ordered indices."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scgsim.palace.results.resolved_records import ParsedTable
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared, found_text


def require_finite_data(
    row: Mapping[str, Any],
    *,
    allow_infinite: bool = False,
    allow_non_numeric: bool = False,
) -> None:
    """Reject missing, boolean or nonfinite numerical row values outside explicit exclusions."""
    for key, value in row.items():
        if value is None:
            raise ValueError(
                Declared(
                    f"result row has a missing entry in column {key!r}",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(dict(row)))
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if not allow_non_numeric:
                raise ValueError(
                    Declared(
                        f"result value in column {key!r}", RETURNED_RUN_FOLDER
                    ).refusal("is not numeric")
                )
            continue
        if math.isnan(value) or (not allow_infinite and not math.isfinite(value)):
            raise ValueError(
                Declared(
                    f"result value in column {key!r}", RETURNED_RUN_FOLDER
                ).refusal("is non-finite")
            )


def require_surface_q_data(row: Mapping[str, Any]) -> None:
    """Permit only native lossless-Q infinities; all participation stays finite."""

    for key, value in row.items():
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                Declared(
                    f"result value in column {key!r}", RETURNED_RUN_FOLDER
                ).refusal("is not numeric")
            )
        if math.isnan(value):
            raise ValueError(
                Declared(
                    f"result value in column {key!r}", RETURNED_RUN_FOLDER
                ).refusal("is non-finite")
            )
        if math.isinf(value) and (
            value < 0 or not (key == "Q" or key.startswith("Q_surf["))
        ):
            raise ValueError(
                Declared(
                    "only a positive solver-native Q", RETURNED_RUN_FOLDER
                ).refusal(f"may contain infinity; got {key!r}")
            )


def require_ordered_index(table: ParsedTable, column: str, expected_count: int) -> None:
    """Require the table index to be the complete ordered sequence from 1 through N."""
    values = [row.get(column) for row in table.rows]
    expected = list(range(1, expected_count + 1))
    if values != expected:
        raise ValueError(
            Declared(f"{table.path} {column!r} values", RETURNED_RUN_FOLDER).refusal(
                f"must be ordered 1..{expected_count}"
            )
        )


def require_positive_index(value: Any) -> None:
    """Require a positive integer native mode or excitation index."""
    if not isinstance(value, (int, float)) or value != int(value) or value <= 0:
        raise ValueError(
            Declared(
                "matrix row index must be a positive integer", RETURNED_RUN_FOLDER
            ).refusal(found_text(value))
        )


def validate_error_indicators(table: ParsedTable) -> None:
    """Require the native scalar AMR-indicator table shape and finite values."""
    if tuple(table.headers) != ("Norm", "Minimum", "Maximum", "Mean"):
        raise ValueError(
            Declared(
                f"{table.path} error-indicators header", RETURNED_RUN_FOLDER
            ).refusal("is malformed")
        )
    for row in table.rows:
        require_finite_data(row)
    if len(table.rows) != 1:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                "must contain exactly one error indicator row"
            )
        )
