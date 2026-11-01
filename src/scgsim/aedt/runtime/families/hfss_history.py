"""Own saved eigenmode traces and adaptive solution history."""

from __future__ import annotations

from typing import Any

from scgsim.aedt.runtime.families.hfss_requests import eigenmode_control
from scgsim.aedt.specs.hfss import (
    HfssEprSpec,
)


def _frequency_solution_trace(
    data: Any, expression: str, *, category: str, solution: str
) -> dict[str, Any]:
    """Retain raw frequency and optional explicit native SI evidence."""

    trace = solution_trace(data, expression)
    if trace["unit"] != "":
        return trace
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "query": {
            "category": category,
            "solution": solution,
            "expression": expression,
            "sweeps": {"Pass": "All"},
            "si_value": True,
            "real_method": "GetRealDataValues",
            "imaginary_method": "GetImagDataValues",
        },
        "unit": "Hz",
        "variations": [],
    }
    trace["si_evidence"] = evidence
    # SI evidence is optional; failure must retain the original partial trace.
    try:
        import numpy as np  # noqa: PLC0415 -- optional native unit/SI evidence loads its SDK or NumPy dependency only when queried.

        if len(data._original_data) != len(data.variations):  # noqa: SLF001 -- pinned native solution handles provide independent SI frequency reads.
            raise RuntimeError("native SI frequency variation count differs")
        real_rows: list[list[Any]] = []
        imaginary_rows: list[list[Any]] = []
        for index, native in enumerate(data._original_data):  # noqa: SLF001 -- pinned native solution handles provide independent SI frequency reads.
            variation = data.variations[index]
            observed: dict[str, Any] = {
                "index": index,
                "coordinates": dict(variation),
                "intrinsics": {
                    name: values.tolist()
                    for name, values in data.intrinsics_by_variation(index).items()
                },
            }
            evidence["variations"].append(observed)
            observed["real_values"] = list(native.GetRealDataValues(expression, True))
            observed["imaginary_values"] = list(
                native.GetImagDataValues(expression, True)
            )
            real_rows.extend(
                data._full_keys(  # noqa: SLF001 -- pinned coordinate expansion preserves the native variation-row ordering.
                    variation, np.asarray(observed["real_values"], dtype=float), index
                ).tolist()
            )
            imaginary_rows.extend(
                data._full_keys(  # noqa: SLF001 -- pinned coordinate expansion preserves the native variation-row ordering.
                    variation,
                    np.asarray(observed["imaginary_values"], dtype=float),
                    index,
                ).tolist()
            )
        coordinates = [row[:-1] for row in trace["real_rows"]]
        if not (
            coordinates
            == [row[:-1] for row in real_rows]
            == [row[:-1] for row in imaginary_rows]
            == [row[:-1] for row in trace["imaginary_rows"]]
        ):
            raise RuntimeError("native SI frequency coordinates differ from raw trace")
        evidence.update(
            status="available",
            columns=trace["columns"],
            real_rows=real_rows,
            imaginary_rows=imaginary_rows,
        )
    except Exception as exc:  # noqa: BLE001 -- retain optional native SI failure.
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return trace


def solution_trace(data: Any, expression: str) -> dict[str, Any]:
    """Detach one native trace while retaining its coordinate columns."""

    real, imaginary = data.full_matrix_real_imag
    variation_keys = tuple(data.variations[0]) if data.variations else ()
    intrinsic_keys = tuple(data.intrinsics_by_variation(0)) if data.variations else ()
    if any(
        tuple(variation) != variation_keys
        or tuple(data.intrinsics_by_variation(index)) != intrinsic_keys
        for index, variation in enumerate(data.variations)
    ):
        raise RuntimeError(
            "native adaptive trace coordinate columns differ by variation"
        )
    columns = [*variation_keys, *intrinsic_keys, "value"]
    real_rows = real[expression].tolist()
    imaginary_rows = imaginary[expression].tolist()
    if any(len(row) != len(columns) for row in (*real_rows, *imaginary_rows)):
        raise RuntimeError("native adaptive trace coordinate width differs")
    return {
        "columns": columns,
        "real_rows": real_rows,
        "imaginary_rows": imaginary_rows,
        "unit": data.units_data.get(expression),
    }


def read_body_eigenmode_history(app: Any, spec: HfssEprSpec) -> dict[str, Any]:
    """Read native adaptive frequency history independently of EPR cache analysis."""
    control = eigenmode_control(spec)
    category = "Eigenmode Parameters"
    solution = f"{control.setup_name} : AdaptivePass"
    frequency_solutions = [
        str(item) for item in app.post.available_report_solutions(category) or ()
    ]
    field_solutions = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if solution not in frequency_solutions:
        raise RuntimeError(
            f"native adaptive solution is unavailable for {category!r}: {frequency_solutions!r}"
        )
    quantities: list[str] = []
    for quantity_category in (
        app.post.available_quantities_categories(category, solution=solution) or ()
    ):
        for quantity in (
            app.post.available_report_quantities(
                category,
                solution=solution,
                quantities_category=quantity_category,
            )
            or ()
        ):
            if str(quantity).startswith("Mode(") and quantity not in quantities:
                quantities.append(str(quantity))
    expected = [f"Mode({index})" for index in range(1, control.num_modes + 1)]
    if len(quantities) != len(expected) or set(quantities) != set(expected):
        raise RuntimeError(
            f"native adaptive Eigenmode quantities differ: {quantities!r}"
        )
    data = app.post.get_solution_data_per_variation(
        category, solution, [], {"Pass": "All"}, quantities
    )
    if not data:
        raise RuntimeError("native adaptive Eigenmode history is unavailable")
    data.primary_sweep = "Pass"
    rows = {
        quantity: _frequency_solution_trace(
            data, quantity, category=category, solution=solution
        )
        for quantity in expected
    }
    return {
        "schema_version": "scgsim.aedt.epr-adaptive-history.v1",
        "category": category,
        "solution": solution,
        "primary_sweep": data.primary_sweep,
        "pass_axis": data.primary_sweep_values.tolist(),
        "pass_unit": data.units_sweeps.get("Pass"),
        "frequency_traces": rows,
        "cache_integral_traces": {
            "status": "not_requested",
            "requested_titles": [],
            "items": [],
            "traces": {},
            "queries": {},
        },
        "report_inventory": {
            "eigenmode_parameter_solutions": frequency_solutions,
            "field_solutions": field_solutions,
        },
        "frequency_unit_status": "native_reported_only",
    }
