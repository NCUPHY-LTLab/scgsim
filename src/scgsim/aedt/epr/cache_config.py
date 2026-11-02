"""Encode expression cache identities, modal contexts, and native setup requests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from scgsim.aedt.runtime.native.common import detached_data
from scgsim.aedt.specs.epr import (
    ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet,
    iter_expression_convergence,
)


def identity_sha256(value: Any) -> str:
    """Hash a canonical expression or cache identity with its historical encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def cache_intrinsics(mode_count: int, selected_mode: int) -> str:
    """Encode the complete one-hot postprocessing context for one mode."""
    assignments = ["Phase='0deg'"]
    assignments.extend(
        f"scgsim_epr_pp_mode_{index}='{'1' if index == selected_mode else '0'}'"
        for index in range(1, mode_count + 1)
    )
    return " ".join(assignments)


def expression_name(
    expressions: list[dict[str, Any]],
    *,
    purpose: str,
    selection_kind: str,
    selection_id_key: str,
    selection_id: str,
) -> str:
    """Resolve a convergence target to its unique authored expression."""
    matches = [
        item
        for item in expressions
        if item["identity"].get("purpose") == purpose
        and item["identity"].get("selection", {}).get("kind") == selection_kind
        and item["identity"].get("selection", {}).get(selection_id_key) == selection_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"compiled EPR expression binding is missing or ambiguous: {purpose!r}"
        )
    return str(matches[0]["name"])


def cache_items(
    expressions: list[dict[str, Any]],
    mode_count: int,
    selected_modes: tuple[int, ...],
    convergence: ExpressionCacheConvergence
    | ExpressionCacheConvergenceSet
    | None = None,
) -> list[dict[str, Any]]:
    """Bind native cache titles to mode-specific expression identities."""
    items: list[dict[str, Any]] = []
    for mode in selected_modes:
        intrinsics = cache_intrinsics(mode_count, mode)
        for expression in expressions:
            identity = {
                "name": expression["name"],
                "identity_sha256": expression["identity"]["sha256"],
                "intrinsics": intrinsics,
                "mode": mode,
            }
            items.append(
                {
                    "title": f"scgsim_epr_cache_m{mode}_{identity_sha256(identity)[:16]}",
                    "expression": expression["name"],
                    "intrinsics": intrinsics,
                    "is_convergence": False,
                    "mode": mode,
                    "expression_identity_sha256": expression["identity"]["sha256"],
                    "purpose": expression["identity"]["purpose"],
                    "selection": detached_data(expression["identity"]["selection"]),
                }
            )
            item = items[-1]
            if expression.get("cache_only"):
                item["cache_only"] = True
            for control in iter_expression_convergence(convergence):
                if (
                    expression.get("convergence_target") == control.target.to_payload()
                    and mode == control.mode
                ):
                    item["is_convergence"] = True
                    item["use_relative_convergence"] = control.use_relative_convergence
                    item["criterion"] = control.criterion
                    item["control_id"] = identity_sha256(control.to_payload())
                    item["requested_control"] = control.to_payload()
    if len({item["title"] for item in items}) != len(items):
        raise RuntimeError("compiled EPR cache titles are not unique")
    return items


def native_expression_cache(items: list[dict[str, Any]]) -> list[Any]:
    """Encode the native cache request with its established convergence fields."""
    result: list[Any] = ["NAME:ExpressionCache"]
    for item in items:
        result.append(
            [
                "NAME:CacheItem",
                "Title:=",
                item["title"],
                "Expression:=",
                item["expression"],
                "Intrinsics:=",
                item["intrinsics"],
                "IsConvergence:=",
                item.get("is_convergence", False),
                "UseRelativeConvergence:=",
                int(item.get("use_relative_convergence", False)),
                "MaxConvergenceDelta:=",
                item.get("criterion", 1),
                "MaxConvergeValue:=",
                str(item.get("criterion", 1)),
                "ReportType:=",
                "Fields",
                ["NAME:ExpressionContext"],
            ]
        )
    return result
