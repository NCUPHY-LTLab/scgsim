"""Read saved adaptive-pass traces and turn them into per-pass EPR rows.

Owns: querying the native frequency and expression-cache traces, validating them
against the persisted cache items, and combining them into `EprResult` rows.
Does not own: authoring the expressions (`cache.py`), unit and slot handling of one
integral (`integrals.py`), or the energy arithmetic (`analysis.py`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.epr.analysis import combine_epr_mode
from scgsim.aedt.epr.cache import cache_intrinsics
from scgsim.aedt.epr.integrals import (
    SURFACE_ANALYSIS_SCOPE,
    attach_saved_native_surface_membership,
    canonical_integral,
    empty_raw_integrals,
    integral_target,
    purpose_unit,
    store_integral,
)
from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.runtime.families.hfss_body_eigenmode import (
    read_body_eigenmode_history,
    solution_trace,
)
from scgsim.aedt.specs.hfss_prepared import HfssEprSpec


def _trace_by_pass(trace: dict[str, Any], *, final_pass: int) -> dict[int, float]:
    columns = trace["columns"]
    if "Pass" not in columns or columns[-1] != "value":
        raise RuntimeError("native adaptive trace lacks explicit Pass/value columns")
    pass_index = columns.index("Pass")
    result: dict[int, float] = {}
    for row in trace["real_rows"]:
        pass_number = float(row[pass_index])
        if not pass_number.is_integer() or pass_number <= 0:
            raise RuntimeError("native adaptive trace Pass coordinate is invalid")
        pass_value = int(pass_number)
        value = float(row[-1])
        if pass_value > final_pass:
            if math.isfinite(value):
                raise RuntimeError(
                    "native adaptive trace has a finite value beyond the completed pass"
                )
            continue
        if not math.isfinite(value):
            raise RuntimeError("native adaptive trace value is nonfinite")
        if pass_value in result:
            raise RuntimeError("native adaptive trace repeats one Pass coordinate")
        result[pass_value] = value
    return result


def _trace_for_cache_context(
    trace: dict[str, Any], item: dict[str, Any], query: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the actual one-hot rows before interpreting Pass coordinates."""

    expected = {
        name: float(value)
        for name, value in re.findall(
            r"(scgsim_epr_pp_mode_\d+)='([01])'", item["intrinsics"]
        )
    }
    columns = trace["columns"]
    present = {name for name in expected if name in columns}
    if not present:
        if "Phase" in columns and any(
            float(row[columns.index("Phase")]) != 0.0 for row in trace["real_rows"]
        ):
            return trace, {
                "status": "mismatch",
                "reason": "native phase coordinate differs from scoped query",
                "expected_postprocessing_variables": expected,
            }
        status = (
            "verified_persisted_scoped_query"
            if query.get("persisted_binding") == "exact"
            and query.get("scope") == "one_hot_one_quantity"
            and query.get("returned_expressions") == [query.get("quantity")]
            and query.get("quantity_binding")
            in {"title", "unique_expression", "native_expression_cache_title"}
            else "not_reported"
        )
        return trace, {
            "status": status,
            "expected_postprocessing_variables": expected,
            "evidence": (
                "persisted_item_and_scoped_native_query"
                if status.startswith("verified")
                else None
            ),
        }
    if present != set(expected):
        return trace, {
            "status": "partial",
            "expected_postprocessing_variables": expected,
            "reported_postprocessing_variables": sorted(present),
        }
    if len(trace["real_rows"]) != len(trace["imaginary_rows"]):
        return trace, {
            "status": "mismatch",
            "reason": "native real/imaginary coordinate counts differ",
            "expected_postprocessing_variables": expected,
        }
    selected_real: list[list[float]] = []
    selected_imaginary: list[list[float]] = []
    for real, imaginary in zip(
        trace["real_rows"], trace["imaginary_rows"], strict=False
    ):
        if real[:-1] != imaginary[:-1]:
            return trace, {
                "status": "mismatch",
                "reason": "native real/imaginary coordinates differ",
                "expected_postprocessing_variables": expected,
            }
        if "Phase" in columns and float(real[columns.index("Phase")]) != 0.0:
            continue
        if all(
            float(real[columns.index(name)]) == value
            for name, value in expected.items()
        ):
            selected_real.append(real)
            selected_imaginary.append(imaginary)
    if not selected_real:
        return trace, {
            "status": "mismatch",
            "reason": "no native rows match the persisted one-hot context",
            "expected_postprocessing_variables": expected,
        }
    return {
        **trace,
        "real_rows": selected_real,
        "imaginary_rows": selected_imaginary,
    }, {
        "status": "verified",
        "expected_postprocessing_variables": expected,
        "excluded_context_rows": len(trace["real_rows"]) - len(selected_real),
    }


def adaptive_epr_result(
    spec: HfssEprSpec,
    history: dict[str, Any],
    convergence: dict[str, Any],
    *,
    native_surface_references: list[dict[str, Any]] | None = None,
) -> EprResult:
    """Build one EPR row per (mode, adaptive pass) from the saved traces.

    A row is `complete` only when every cache item has a verified value for that
    pass and a native frequency; otherwise it stays `partial` and records why.
    """

    frequency_traces = history["frequency_traces"]
    cache = history["cache_integral_traces"]
    final_pass = convergence["final_pass"]
    if type(final_pass) is not int or final_pass <= 0:
        raise RuntimeError("native completed pass identity is invalid")
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request is None or spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    rows: list[dict[str, Any]] = []
    for mode in selected_modes:
        series = _mode_series(
            mode, frequency_traces[f"Mode({mode})"], cache, final_pass=final_pass
        )
        rows.extend(
            _pass_row(spec, series, pass_id, native_surface_references)
            for pass_id in range(1, final_pass + 1)
        )
    return EprResult(
        result_kind="adaptive_history",
        setup_name=spec.run_control.setup_name,
        rows=tuple(rows),
        provenance={
            "model_source_sha256": spec.geometry.model_sha256,
            "analysis_source_sha256": spec.geometry.source_sha256,
            "surface_analysis_scope": SURFACE_ANALYSIS_SCOPE,
            "solver_last_completed_pass": convergence["final_pass"],
            "requested_modes": list(selected_modes),
            "cache_status": cache["status"],
            "report_inventory": history["report_inventory"],
            "join_identity": [
                "setup",
                "native_pass",
                "mode",
                "cache_title",
                "intrinsics",
            ],
        },
    )


@dataclass(frozen=True, slots=True)
class _ModeSeries:
    """Everything the saved history says about one mode, keyed by adaptive pass."""

    mode: int
    frequency_trace: dict[str, Any]
    frequencies: dict[int, float]
    items: list[dict[str, Any]]
    item_values: dict[str, dict[int, float]]
    item_contexts: dict[str, dict[str, Any]]
    native_units: dict[str, Any]


def _mode_series(
    mode: int,
    frequency_trace: dict[str, Any],
    cache: dict[str, Any],
    *,
    final_pass: int,
) -> _ModeSeries:
    """Collect one mode's frequency trace and its verified cache-item traces."""

    trace_by_title = cache["traces"]
    query_by_title = cache.get("queries", {})
    frequencies = _trace_by_pass(frequency_trace, final_pass=final_pass)
    mode_items = [
        item
        for item in cache["items"]
        if item["mode"] == mode and not item.get("cache_only")
    ]
    item_values: dict[str, dict[int, float]] = {}
    item_contexts: dict[str, dict[str, Any]] = {}
    for item in mode_items:
        title = item["title"]
        if title not in trace_by_title:
            continue
        scoped_trace, context = _trace_for_cache_context(
            trace_by_title[title], item, query_by_title.get(title, {})
        )
        item_contexts[title] = context
        if not context["status"].startswith("verified"):
            continue
        try:
            item_values[title] = _trace_by_pass(scoped_trace, final_pass=final_pass)
        except RuntimeError as exc:
            item_contexts[title] = {
                **context,
                "status": "invalid_pass_axis",
                "error": str(exc),
            }
    return _ModeSeries(
        mode=mode,
        frequency_trace=frequency_trace,
        frequencies=frequencies,
        items=mode_items,
        item_values=item_values,
        item_contexts=item_contexts,
        native_units={
            title: trace.get("unit") for title, trace in trace_by_title.items()
        },
    )


def _pass_row(
    spec: HfssEprSpec,
    series: _ModeSeries,
    pass_id: int,
    native_surface_references: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build the row for one mode at one adaptive pass."""

    missing = [
        item["title"]
        for item in series.items
        if pass_id not in series.item_values.get(item["title"], {})
    ]
    invalid_context = [
        title
        for title, status in series.item_contexts.items()
        if not status["status"].startswith("verified")
    ]
    row: dict[str, Any] = {
        "mode": series.mode,
        "native_pass": pass_id,
        "status": "partial",
        "missing_cache_titles": missing,
        "unverified_cache_context_titles": invalid_context,
        "frequency_native_unit": series.frequency_trace["unit"],
    }
    raw = empty_raw_integrals()
    integral_evidence, unit_mismatches = _store_pass_integrals(raw, series, pass_id)
    row["raw_integrals"] = raw
    row["raw_integral_evidence"] = integral_evidence
    if native_surface_references:
        attach_saved_native_surface_membership(raw, row, native_surface_references)
    if unit_mismatches:
        row["unit_mismatches"] = unit_mismatches
    frequency_hz = _pass_frequency_hz(row, series, pass_id)
    if frequency_hz is None:
        return row
    raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
    if missing or invalid_context or unit_mismatches:
        row["frequency_hz"] = frequency_hz
        return row
    try:
        row.update(combine_epr_mode(spec.geometry, raw, request=spec.epr_request))
        # combine_epr_mode adds its own `native_surface_references` rows to `row`;
        # the saved face membership belongs on those too.
        if native_surface_references:
            attach_saved_native_surface_membership(raw, row, native_surface_references)
        row["status"] = "complete"
    except (TypeError, ValueError, RuntimeError) as exc:
        row["combination_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _store_pass_integrals(
    raw: dict[str, Any], series: _ModeSeries, pass_id: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Store this pass's verified integrals; return (evidence, unit mismatches)."""

    integral_evidence: list[dict[str, Any]] = []
    unit_mismatches: list[dict[str, Any]] = []
    for item in series.items:
        title = item["title"]
        values = series.item_values.get(title, {})
        if pass_id not in values:
            continue
        native_unit = series.native_units[title]
        context = series.item_contexts[title]
        target = integral_target(item["purpose"], item["selection"])
        try:
            value, unit_evidence = canonical_integral(
                {"value": values[pass_id], "unit": native_unit},
                target=target,
                purpose=item["purpose"],
            )
        except RuntimeError as exc:
            unit_mismatches.append(
                {
                    "title": title,
                    "native_unit": native_unit,
                    "expected_unit": purpose_unit(target.kind),
                    "error": str(exc),
                }
            )
            continue
        integral_evidence.append(
            {
                "title": title,
                "purpose": item["purpose"],
                "context": context,
                **unit_evidence,
            }
        )
        store_integral(
            raw,
            target=target,
            value=value,
            expression_identity={
                "name": item["expression"],
                "identity_sha256": item["expression_identity_sha256"],
                "purpose": item["purpose"],
            },
        )
    return integral_evidence, unit_mismatches


def _pass_frequency_hz(
    row: dict[str, Any], series: _ModeSeries, pass_id: int
) -> float | None:
    """Return the mode frequency in Hz at this pass, or None with the reason in `row`.

    GHz and Hz are converted directly. Any other native unit is unverified unless
    the history carries a native SI frequency table, whose value is then used.
    """

    frequency = series.frequencies.get(pass_id)
    frequency_trace = series.frequency_trace
    if frequency is None:
        row["missing_frequency"] = True
        return None
    if frequency_trace["unit"] == "GHz":
        return frequency * 1e9
    if frequency_trace["unit"] == "Hz":
        return frequency
    row["frequency_native_value"] = frequency
    row["frequency_unit_status"] = "unverified"
    si = frequency_trace.get("si_evidence") if frequency_trace["unit"] == "" else None
    if si is None:
        return None
    row["frequency_unit_evidence"] = {
        key: value
        for key, value in si.items()
        if key not in {"variations", "real_rows", "imaginary_rows"}
    }
    if si["status"] != "available":
        return None
    pass_index = si["columns"].index("Pass")
    si_rows = {
        int(real_row[pass_index]): (real_row, imaginary_row)
        for real_row, imaginary_row in zip(
            si["real_rows"], si["imaginary_rows"], strict=False
        )
    }
    if pass_id not in si_rows:
        row["frequency_unit_evidence"].update(
            status="unavailable",
            error="native SI frequency Pass is unavailable",
        )
        return None
    real_row, imaginary_row = si_rows[pass_id]
    row["frequency_unit_evidence"].update(
        real_row=real_row, imaginary_row=imaginary_row
    )
    row["frequency_unit_status"] = "native_si"
    return real_row[-1]


@dataclass(frozen=True, slots=True)
class _NativeCacheScope:
    """Where to query cached expressions: category, solution and known quantities."""

    category: str
    solution: str
    categories: list[str]
    quantities: list[str]


def adaptive_mode_history(
    app: Any,
    spec: HfssEprSpec,
    cache_items: list[dict[str, Any]],
    persisted_cache_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return raw frequency and cache traces without positional joins.

    Every cache item is queried on its own (one-hot mode, one quantity), so a
    failure is recorded on that item and the rest of the history survives.
    """

    history = read_body_eigenmode_history(app, spec)
    category = history["category"]
    solution = history["solution"]
    frequency_solutions = history["report_inventory"]["eigenmode_parameter_solutions"]
    cache_titles = [item["title"] for item in cache_items]
    cache_observation: dict[str, Any] = {
        "status": "not_requested" if not cache_titles else "unavailable",
        "requested_titles": cache_titles,
        "items": cache_items,
        "traces": {},
        "queries": {},
    }
    if cache_titles:
        try:
            if solution not in frequency_solutions:
                raise RuntimeError(
                    f"native adaptive solution is unavailable for {category}: {frequency_solutions!r}"
                )
            scope = _read_cache_scope(app, category, solution)
            cache_observation["available_quantity_categories"] = scope.categories
            cache_observation["available_quantities"] = {
                "Expression Cache": scope.quantities
            }
            persisted_by_title = {item["title"]: item for item in persisted_cache_items}
            if len(persisted_by_title) != len(persisted_cache_items):
                raise RuntimeError("persisted EPR cache titles are not unique")
            for item in cache_items:
                title = item["title"]
                query: dict[str, Any] = {
                    "persisted_item": persisted_by_title.get(title),
                    "scope": "one_hot_one_quantity",
                }
                cache_observation["queries"][title] = query
                trace = _query_cache_item(app, spec, item, query, scope)
                if trace is not None:
                    cache_observation["traces"][title] = trace
            returned_titles = sorted(cache_observation["traces"])
            cache_observation["returned_titles"] = returned_titles
            cache_observation["missing_titles"] = sorted(
                set(cache_titles) - set(returned_titles)
            )
            cache_observation["unexpected_titles"] = []
            cache_observation["status"] = (
                "complete" if len(returned_titles) == len(cache_titles) else "partial"
            )
        except Exception as exc:  # noqa: BLE001 -- partial native history is evidence.
            cache_observation["error"] = f"{type(exc).__name__}: {exc}"
    history["cache_integral_traces"] = cache_observation
    return history


def _read_cache_scope(app: Any, category: str, solution: str) -> _NativeCacheScope:
    """List the native "Expression Cache" quantities available for this solution."""

    categories = [
        str(item)
        for item in app.post.available_quantities_categories(
            category, solution=solution
        )
        or ()
    ]
    quantities = (
        [
            str(item)
            for item in app.post.available_report_quantities(
                category,
                solution=solution,
                quantities_category="Expression Cache",
            )
            or ()
        ]
        if "Expression Cache" in categories
        else []
    )
    return _NativeCacheScope(category, solution, categories, quantities)


def _query_cache_item(
    app: Any,
    spec: HfssEprSpec,
    item: dict[str, Any],
    query: dict[str, Any],
    scope: _NativeCacheScope,
) -> dict[str, Any] | None:
    """Query one cache item's native trace; record the outcome in `query`.

    Returns the trace on success, or None with `query["error"]` explaining why not.
    """

    title = item["title"]
    persisted = query["persisted_item"]
    if persisted != {
        "title": title,
        "expression": item["expression"],
        "intrinsics": item["intrinsics"],
    }:
        query["error"] = "persisted cache item expression/context differs"
        return None
    query["persisted_binding"] = "exact"
    if item["intrinsics"] != cache_intrinsics(spec.run_control.num_modes, item["mode"]):
        query["error"] = "cache item is not the complete one-hot context"
        return None
    expected_quantity = f"ExprCache({persisted['title']})"
    matches = [
        ("Expression Cache", quantity)
        for quantity in scope.quantities
        if quantity == expected_quantity
    ]
    query["inventory_matches"] = matches
    if len(matches) != 1:
        query["error"] = "native quantity binding is missing or ambiguous"
        return None
    quantity = matches[0][1]
    query["quantity"] = quantity
    query["quantity_binding"] = "native_expression_cache_title"
    # One-hot: the item's own mode variable is 1 and every other mode's is 0.
    sweeps = {"Pass": "All", "Phase": "0deg"}
    sweeps.update(
        {
            f"scgsim_epr_pp_mode_{index}": "1" if index == item["mode"] else "0"
            for index in range(1, spec.run_control.num_modes + 1)
        }
    )
    query["sweeps"] = sweeps
    try:
        cached = app.post.get_solution_data_per_variation(
            scope.category, scope.solution, [], sweeps, [quantity]
        )
        if not cached:
            raise RuntimeError("native scoped cache report is unavailable")
        returned = list(cached.expressions)
        query["returned_expressions"] = returned
        if returned != [quantity]:
            raise RuntimeError("native scoped cache quantity differs")
        cached.primary_sweep = "Pass"
        trace = solution_trace(cached, quantity)
        query["status"] = "reported"
    except Exception as exc:  # noqa: BLE001 -- retain per-item partial history.
        query["error"] = f"{type(exc).__name__}: {exc}"
        return None
    return trace
