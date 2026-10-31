"""Submit EPR expressions as the HFSS expression cache, and read it back.

Owns: turning authored expressions into per-mode cache items, submitting them
with the setup, and verifying the cache persisted in the saved project.
Does not own: authoring the expressions (`expressions.py`), reading traces back
out of a finished solve (`history.py`) or typing one integral (`integrals.py`).

The cache is keyed by adaptive Pass (`UseCacheFor = Pass`). Each item selects one
mode by setting its `scgsim_epr_pp_mode_<n>` variable to 1 and the others to 0.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from scgsim.aedt.epr.expressions import _author_epr_expressions
from scgsim.aedt.runtime.native.common import detached_data, saved_setup_properties
from scgsim.aedt.runtime.native.hfss_eigenmode import submit_eigenmode_cache
from scgsim.aedt.specs.epr import (
    ExpressionCacheConvergence,
)
from scgsim.aedt.specs.hfss_prepared import HfssEprSpec

# --- Entry point ------------------------------------------------------------


def _prepare_expressions_and_cache(
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Author the expressions, then submit them as the setup's expression cache."""

    if inset_plan is None:
        raise RuntimeError("EPR inset geometry must be precomputed before Desktop")
    expressions, authoring, phase_timings = _author_epr_expressions(
        app,
        run_dir,
        spec,
        native_geometry,
        inset_plan,
    )
    started = time.perf_counter()
    cache = _submit_epr_cache(app, spec, expressions, authoring)
    phase_timings["cache_seconds"] = round(time.perf_counter() - started, 6)
    return expressions, cache, phase_timings


# --- Cache items, submission and read-back ----------------------------------


def _identity_sha256(value: Any) -> str:
    """SHA-256 of the canonical JSON form of `value`."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_intrinsics(mode_count: int, selected_mode: int) -> str:
    """Intrinsics string that selects one mode: its variable 1, the others 0."""

    assignments = ["Phase='0deg'"]
    assignments.extend(
        f"scgsim_epr_pp_mode_{index}='{'1' if index == selected_mode else '0'}'"
        for index in range(1, mode_count + 1)
    )
    return " ".join(assignments)


def _cache_items(
    expressions: list[dict[str, Any]],
    mode_count: int,
    selected_modes: tuple[int, ...],
    convergence: ExpressionCacheConvergence | None = None,
) -> list[dict[str, Any]]:
    """One cache item per (mode, expression), titled by a hash of its identity."""

    items: list[dict[str, Any]] = []
    for mode in selected_modes:
        intrinsics = _cache_intrinsics(mode_count, mode)
        for expression in expressions:
            identity = {
                "name": expression["name"],
                "identity_sha256": expression["identity"]["sha256"],
                "intrinsics": intrinsics,
                "mode": mode,
            }
            items.append(
                {
                    "title": f"scgsim_epr_cache_m{mode}_{_identity_sha256(identity)[:16]}",
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
            if (
                expression.get("cache_only")
                and convergence is not None
                and mode == convergence.mode
            ):
                item["is_convergence"] = True
                item["use_relative_convergence"] = convergence.use_relative_convergence
                item["criterion"] = convergence.criterion
    if len({item["title"] for item in items}) != len(items):
        raise RuntimeError("compiled EPR cache titles are not unique")
    return items


def _native_expression_cache(items: list[dict[str, Any]]) -> list[Any]:
    """Arguments of AEDT's `ExpressionCache` setup property for these items."""

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


def _submit_epr_cache(
    app: Any,
    spec: HfssEprSpec,
    expressions: list[dict[str, Any]],
    authoring: dict[str, Any],
) -> dict[str, Any]:
    """Write the cache items into the setup and return the cache request record."""

    request = spec.epr_request
    if request is None:
        raise RuntimeError("EPR cache submission requires an analysis request")
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    items = _cache_items(
        expressions,
        spec.run_control.num_modes,
        selected_modes,
        spec.expression_convergence,
    )
    raw_cache = _native_expression_cache(items)
    submit_eigenmode_cache(
        app,
        spec.run_control,
        expression_cache=raw_cache,
        use_cache_for=["Pass"],
    )
    return {
        "schema_version": "scgsim.aedt.epr-cache-request.v1",
        "setup_name": spec.run_control.setup_name,
        "use_cache_for": ["Pass"],
        "items": items,
        "source_assignment": authoring["source_assignment"],
        "postprocessing_variables": authoring["postprocessing_variables"],
        "native_surface_references": detached_data(
            authoring["native_surface_references"]
        ),
        "native_readback": {
            "status": "PENDING_SAVED_SETUP_READBACK",
            "reason": (
                "submitted cache settings are not saved-setup readback; "
                "saved properties are inspected after the project save"
            ),
        },
        "raw_cache_args_sha256": _identity_sha256(raw_cache),
    }


def _read_cache(
    app: Any, spec: HfssEprSpec, submitted: list[dict[str, Any]]
) -> dict[str, Any]:
    """Read the cache back from the saved setup and require it to match `submitted`."""

    raw = saved_setup_properties(app, spec.run_control.setup_name)
    use_cache_for = raw.get("UseCacheFor")
    if isinstance(use_cache_for, str):
        use_cache_for = [use_cache_for]
    expression_cache = raw.get("ExpressionCache")
    if not isinstance(expression_cache, dict):
        raise RuntimeError("saved EPR setup lacks ExpressionCache")
    raw_items = expression_cache.get("CacheItem")
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raise RuntimeError("saved EPR ExpressionCache items are invalid")
    persisted = []
    convergence_field_names = (
        "IsConvergence",
        "UseRelativeConvergence",
        "MaxConvergenceDelta",
        "MaxConvergeValue",
    )
    convergence_fields = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise RuntimeError("saved EPR cache item is invalid")
        convergence_fields.append(
            {
                "title": item.get("Title"),
                "values": {
                    name: item[name] for name in convergence_field_names if name in item
                },
                "unavailable": [
                    name for name in convergence_field_names if name not in item
                ],
            }
        )
        persisted.append(
            {
                "title": item.get("Title"),
                "expression": item.get("Expression"),
                # Unescape AEDT's saved `\'` so it compares equal to what was submitted.
                "intrinsics": str(item.get("Intrinsics", "")).replace("\\'", "'"),
            }
        )
    expected = [
        {
            "title": item["title"],
            "expression": item["expression"],
            "intrinsics": item["intrinsics"],
        }
        for item in submitted
    ]
    if use_cache_for != ["Pass"] or persisted != expected:
        raise RuntimeError("saved EPR cache persisted-subset readback mismatch")
    return {
        "use_cache_for": use_cache_for,
        "items": persisted,
        "convergence_fields": convergence_fields,
    }
