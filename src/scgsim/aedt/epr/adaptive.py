"""Join native adaptive frequency and expression-cache traces into EPR results."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from scgsim.aedt.epr.analysis import (
    combine_epr_mode,
)
from scgsim.aedt.epr.cache_config import (
    cache_intrinsics,
)
from scgsim.aedt.epr.integrals import (
    SURFACE_ANALYSIS_SCOPE,
    attach_saved_native_surface_membership,
    canonical_integral,
    empty_raw_integrals,
    integral_target,
    purpose_unit,
    store_integral,
)
from scgsim.aedt.epr.models import (
    EprResult,
)
from scgsim.aedt.runtime.families.hfss_history import (
    read_body_eigenmode_history,
    solution_trace,
)
from scgsim.aedt.specs.hfss import HfssEprSpec
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl


def trace_by_pass(trace: dict[str, Any], *, final_pass: int) -> dict[int, float]:
    """Index valid native pass values without a positional frequency join."""
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


def trace_for_cache_context(
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
    """Combine each available mode and native pass while retaining partial rows."""
    return _AdaptiveJoin(spec, history, convergence, native_surface_references).build()


@dataclass
class _AdaptiveJoin:
    """Carry frequency, context, and integral evidence through one adaptive join."""

    spec: HfssEprSpec
    history: dict[str, Any]
    convergence: dict[str, Any]
    native_surface_references: list[dict[str, Any]] | None

    def build(self) -> EprResult:
        self._initialize()
        for mode in self.selected_modes:
            self.mode = mode
            self._join_mode()
        return self._result()

    def _initialize(self) -> None:
        self.frequency_traces = self.history["frequency_traces"]
        self.cache = self.history["cache_integral_traces"]
        self.final_pass = self.convergence["final_pass"]
        if type(self.final_pass) is not int or self.final_pass <= 0:
            raise RuntimeError("native completed pass identity is invalid")
        self.trace_by_title = self.cache["traces"]
        self.query_by_title = self.cache.get("queries", {})
        self.rows: list[dict[str, Any]] = []
        self.selected_modes = (
            tuple(
                range(1, cast(EigenmodeRunControl, self.spec.run_control).num_modes + 1)
            )
            if self.spec.epr_request is None
            or self.spec.epr_request.mode_indices is None
            else self.spec.epr_request.mode_indices
        )

    def _join_mode(self) -> None:
        self.frequency_trace = self.frequency_traces[f"Mode({self.mode})"]
        self.frequencies = trace_by_pass(
            self.frequency_trace, final_pass=self.final_pass
        )
        self.mode_items = [
            item
            for item in self.cache["items"]
            if item["mode"] == self.mode and not item.get("cache_only")
        ]
        self.item_values: dict[str, dict[int, float]] = {}
        self.item_contexts: dict[str, dict[str, Any]] = {}
        for item in self.mode_items:
            title = item["title"]
            if title not in self.trace_by_title:
                continue
            scoped_trace, context = trace_for_cache_context(
                self.trace_by_title[title], item, self.query_by_title.get(title, {})
            )
            self.item_contexts[title] = context
            if not context["status"].startswith("verified"):
                continue
            try:
                self.item_values[title] = trace_by_pass(
                    scoped_trace, final_pass=self.final_pass
                )
            except RuntimeError as exc:
                self.item_contexts[title] = {
                    **context,
                    "status": "invalid_pass_axis",
                    "error": str(exc),
                }
        for pass_id in range(1, self.final_pass + 1):
            self.pass_id = pass_id
            self.rows.append(self._pass_row())

    def _pass_row(self) -> dict[str, Any]:
        self._start_row()
        self._collect_integrals()
        return self._finish_row()

    def _start_row(self) -> None:
        self.missing = [
            item["title"]
            for item in self.mode_items
            if self.pass_id not in self.item_values.get(item["title"], {})
        ]
        self.invalid_context = [
            title
            for title, status in self.item_contexts.items()
            if not status["status"].startswith("verified")
        ]
        self.row: dict[str, Any] = {
            "mode": self.mode,
            "native_pass": self.pass_id,
            "status": "partial",
            "missing_cache_titles": self.missing,
            "unverified_cache_context_titles": self.invalid_context,
            "frequency_native_unit": self.frequency_trace["unit"],
        }
        self.raw = empty_raw_integrals()
        self.integral_evidence: list[dict[str, Any]] = []
        self.unit_mismatches: list[dict[str, Any]] = []

    def _collect_integrals(self) -> None:
        for item in self.mode_items:
            title = item["title"]
            values = self.item_values.get(title, {})
            if self.pass_id not in values:
                continue
            native_unit = self.trace_by_title[title].get("unit")
            context = self.item_contexts[title]
            target = integral_target(item["purpose"], item["selection"])
            try:
                value, unit_evidence = canonical_integral(
                    {"value": values[self.pass_id], "unit": native_unit},
                    target=target,
                    purpose=item["purpose"],
                )
            except RuntimeError as exc:
                self.unit_mismatches.append(
                    {
                        "title": title,
                        "native_unit": native_unit,
                        "expected_unit": purpose_unit(target.kind),
                        "error": str(exc),
                    }
                )
                continue
            self.integral_evidence.append(
                {
                    "title": title,
                    "purpose": item["purpose"],
                    "context": context,
                    **unit_evidence,
                }
            )
            store_integral(
                self.raw,
                target=target,
                value=value,
                expression_identity={
                    "name": item["expression"],
                    "identity_sha256": item["expression_identity_sha256"],
                    "purpose": item["purpose"],
                },
            )

    def _finish_row(self) -> dict[str, Any]:
        self.row["raw_integrals"] = self.raw
        self.row["raw_integral_evidence"] = self.integral_evidence
        if self.native_surface_references:
            attach_saved_native_surface_membership(
                self.raw, self.row, self.native_surface_references
            )
        if self.unit_mismatches:
            self.row["unit_mismatches"] = self.unit_mismatches
        frequency = self.frequencies.get(self.pass_id)
        if frequency is None:
            self.row["missing_frequency"] = True
            return self.row
        if self.frequency_trace["unit"] == "GHz":
            frequency_hz = frequency * 1e9
        elif self.frequency_trace["unit"] == "Hz":
            frequency_hz = frequency
        else:
            self.row["frequency_native_value"] = frequency
            self.row["frequency_unit_status"] = "unverified"
            si = (
                self.frequency_trace.get("si_evidence")
                if self.frequency_trace["unit"] == ""
                else None
            )
            if si is None:
                return self.row
            self.row["frequency_unit_evidence"] = {
                key: value
                for key, value in si.items()
                if key not in {"variations", "real_rows", "imaginary_rows"}
            }
            if si["status"] != "available":
                return self.row
            pass_index = si["columns"].index("Pass")
            si_rows = {
                int(real_row[pass_index]): (real_row, imaginary_row)
                for real_row, imaginary_row in zip(
                    si["real_rows"], si["imaginary_rows"], strict=False
                )
            }
            if self.pass_id not in si_rows:
                self.row["frequency_unit_evidence"].update(
                    status="unavailable",
                    error="native SI frequency Pass is unavailable",
                )
                return self.row
            real_row, imaginary_row = si_rows[self.pass_id]
            self.row["frequency_unit_evidence"].update(
                real_row=real_row, imaginary_row=imaginary_row
            )
            frequency_hz = real_row[-1]
            self.row["frequency_unit_status"] = "native_si"
        self.raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
        if self.missing or self.invalid_context or self.unit_mismatches:
            self.row["frequency_hz"] = frequency_hz
            return self.row
        try:
            self.row.update(
                combine_epr_mode(
                    self.spec.geometry, self.raw, request=self.spec.epr_request
                )
            )
            if self.native_surface_references:
                attach_saved_native_surface_membership(
                    self.raw, self.row, self.native_surface_references
                )
            self.row["status"] = "complete"
        except (TypeError, ValueError, RuntimeError) as exc:
            self.row["combination_error"] = f"{type(exc).__name__}: {exc}"
        return self.row

    def _result(self) -> EprResult:
        return EprResult(
            result_kind="adaptive_history",
            setup_name=cast(EigenmodeRunControl, self.spec.run_control).setup_name,
            rows=tuple(self.rows),
            _legacy_payload=self.spec.legacy_payload,
            provenance={
                "model_source_sha256": self.spec.geometry.model_sha256,
                "analysis_source_sha256": self.spec.geometry.source_sha256,
                "surface_analysis_scope": SURFACE_ANALYSIS_SCOPE,
                "solver_last_completed_pass": self.convergence["final_pass"],
                "requested_modes": list(self.selected_modes),
                "cache_status": self.cache["status"],
                "expression_convergence_targets": self.history.get(
                    "expression_convergence_targets", []
                ),
                "combined_expression_convergence_status": self.history.get(
                    "combined_expression_convergence_status", "unknown"
                ),
                "report_inventory": self.history["report_inventory"],
                "join_identity": [
                    "setup",
                    "native_pass",
                    "mode",
                    "cache_title",
                    "intrinsics",
                ],
            },
        )


def adaptive_mode_history(
    app: Any,
    spec: HfssEprSpec,
    cache_items: list[dict[str, Any]],
    persisted_cache_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return raw frequency and cache traces without positional joins."""
    return _AdaptiveHistory(app, spec, cache_items, persisted_cache_items).read()


@dataclass
class _AdaptiveHistory:
    """Keep native inventory and each persisted modal query with its observations."""

    app: Any
    spec: HfssEprSpec
    cache_items: list[dict[str, Any]]
    persisted_cache_items: list[dict[str, Any]]

    def read(self) -> dict[str, Any]:
        self.history = read_body_eigenmode_history(self.app, self.spec)
        self.category = self.history["category"]
        self.solution = self.history["solution"]
        self.frequency_solutions = self.history["report_inventory"][
            "eigenmode_parameter_solutions"
        ]
        self.cache_titles = [item["title"] for item in self.cache_items]
        self.cache_observation: dict[str, Any] = {
            "status": "not_requested" if not self.cache_titles else "unavailable",
            "requested_titles": self.cache_titles,
            "items": self.cache_items,
            "traces": {},
            "queries": {},
        }
        if self.cache_titles:
            try:
                self._read_query_inventory()
                for item in self.cache_items:
                    self._read_query(item)
                self._summarize_queries()
            except Exception as exc:  # noqa: BLE001 -- partial native history is evidence.
                self.cache_observation["error"] = f"{type(exc).__name__}: {exc}"
        return self._result()

    def _read_query_inventory(self) -> None:
        if self.solution not in self.frequency_solutions:
            raise RuntimeError(
                f"native adaptive solution is unavailable for {self.category}: {self.frequency_solutions!r}"
            )
        cache_categories = [
            str(item)
            for item in self.app.post.available_quantities_categories(
                self.category, solution=self.solution
            )
            or ()
        ]
        self.cache_quantities = (
            [
                str(item)
                for item in self.app.post.available_report_quantities(
                    self.category,
                    solution=self.solution,
                    quantities_category="Expression Cache",
                )
                or ()
            ]
            if "Expression Cache" in cache_categories
            else []
        )
        self.cache_observation["available_quantity_categories"] = cache_categories
        self.cache_observation["available_quantities"] = {
            "Expression Cache": self.cache_quantities
        }
        self.persisted_by_title = {
            item["title"]: item for item in self.persisted_cache_items
        }
        if len(self.persisted_by_title) != len(self.persisted_cache_items):
            raise RuntimeError("persisted EPR cache titles are not unique")

    def _read_query(self, item: dict[str, Any]) -> None:
        title = item["title"]
        query: dict[str, Any] = {
            "persisted_item": self.persisted_by_title.get(title),
            "scope": "one_hot_one_quantity",
        }
        self.cache_observation["queries"][title] = query
        persisted = self.persisted_by_title.get(title)
        if persisted != {
            "title": title,
            "expression": item["expression"],
            "intrinsics": item["intrinsics"],
        }:
            query["error"] = "persisted cache item expression/context differs"
            return
        query["persisted_binding"] = "exact"
        if item["intrinsics"] != cache_intrinsics(
            cast(EigenmodeRunControl, self.spec.run_control).num_modes, item["mode"]
        ):
            query["error"] = "cache item is not the complete one-hot context"
            return
        expected_quantity = f"ExprCache({cast(Mapping[str, Any], persisted)['title']})"
        matches = [
            ("Expression Cache", quantity)
            for quantity in self.cache_quantities
            if quantity == expected_quantity
        ]
        query["inventory_matches"] = matches
        if len(matches) != 1:
            query["error"] = "native quantity binding is missing or ambiguous"
            return
        quantity = matches[0][1]
        query["quantity"] = quantity
        query["quantity_binding"] = "native_expression_cache_title"
        sweeps = {"Pass": "All", "Phase": "0deg"}
        sweeps.update(
            {
                f"scgsim_epr_pp_mode_{index}": ("1" if index == item["mode"] else "0")
                for index in range(
                    1, cast(EigenmodeRunControl, self.spec.run_control).num_modes + 1
                )
            }
        )
        query["sweeps"] = sweeps
        try:
            cached = self.app.post.get_solution_data_per_variation(
                self.category, self.solution, [], sweeps, [quantity]
            )
            if not cached:
                raise RuntimeError("native scoped cache report is unavailable")
            returned = list(cached.expressions)
            query["returned_expressions"] = returned
            if returned != [quantity]:
                raise RuntimeError("native scoped cache quantity differs")
            cached.primary_sweep = "Pass"
            self.cache_observation["traces"][title] = solution_trace(cached, quantity)
            query["status"] = "reported"
        except Exception as exc:  # noqa: BLE001 -- retain per-item partial history.
            query["error"] = f"{type(exc).__name__}: {exc}"

    def _summarize_queries(self) -> None:
        returned_titles = sorted(self.cache_observation["traces"])
        self.cache_observation["returned_titles"] = returned_titles
        self.cache_observation["missing_titles"] = sorted(
            set(self.cache_titles) - set(returned_titles)
        )
        self.cache_observation["unexpected_titles"] = []
        self.cache_observation["status"] = (
            "complete" if len(returned_titles) == len(self.cache_titles) else "partial"
        )

    def _result(self) -> dict[str, Any]:
        self.history["cache_integral_traces"] = self.cache_observation
        self.history["expression_convergence_targets"] = [
            {
                "control_id": item["control_id"],
                "mode": item["mode"],
                "title": item["title"],
                "expression": item["expression"],
                "expression_identity_sha256": item["expression_identity_sha256"],
                "requested_control": item["requested_control"],
                "native_trace": self.cache_observation["traces"].get(item["title"]),
                "query": self.cache_observation["queries"].get(item["title"]),
                "per_pass_fields_status": "unknown",
            }
            for item in self.cache_items
            if item["is_convergence"]
        ]
        self.history["combined_expression_convergence_status"] = "unknown"
        return self.history
