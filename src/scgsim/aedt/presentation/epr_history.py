"""Selected-pass bars and per-pass convergence lines for an EPR result.

Owns: projecting ``EprResult`` rows into display-only bars and lines.
Does not own: drawing them (``epr.py``, ``runs.py``).
"""

from __future__ import annotations

import html
from typing import Any

from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.presentation.epr_surfaces import (
    surface_hover,
    surface_key,
    surface_label,
    surface_metadata,
    surface_tick_label,
    usable_frequency,
)


def history_view(
    result: EprResult,
    *,
    mode: int | None,
    native_pass: int | None,
    show_convergence: bool,
) -> dict[str, Any]:
    """Project an EPR result into bars for one pass and, optionally, lines per pass."""
    requested_modes = _requested_modes(result, mode)
    target_pass = _target_pass(result, native_pass)
    selected = [
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (
            result.result_kind != "adaptive_history"
            or row["native_pass"] == target_pass
        )
    ]
    unavailable = _missing_selections(requested_modes, selected, target_pass)
    bars = _selected_pass_bars(selected, unavailable)
    convergence_view = show_convergence and result.result_kind == "adaptive_history"
    lines: list[dict[str, Any]] = []
    if convergence_view:
        lines = _convergence_lines(result, requested_modes)
    return {
        "result_kind": result.result_kind,
        "mode": mode,
        "native_pass": target_pass,
        "show_convergence": show_convergence,
        "convergence_view": convergence_view,
        "bars": bars,
        "lines": lines,
        "unavailable": unavailable,
    }


def _requested_modes(result: EprResult, mode: int | None) -> list[int]:
    """Modes the run asked for, narrowed to one when the caller picks a mode."""
    requested_modes_raw = result.provenance.get(
        "requested_modes", sorted({int(row["mode"]) for row in result.rows})
    )
    known_modes = {
        int(item) for item in requested_modes_raw if type(item) is int and item > 0
    }
    if mode is not None and mode not in known_modes:
        raise ValueError(f"mode {mode} was not requested by this EPR result")
    return [
        int(item)
        for item in requested_modes_raw
        if type(item) is int and item > 0 and (mode is None or item == mode)
    ]


def _target_pass(result: EprResult, native_pass: int | None) -> int | None:
    """The pass to show: the caller's, else the solver's last completed pass."""
    if result.result_kind == "adaptive_history" and native_pass is None:
        observed_last = result.provenance.get("solver_last_completed_pass")
        if type(observed_last) is not int or observed_last <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        return observed_last
    return native_pass


def _missing_selections(
    requested_modes: list[int],
    selected: list[dict[str, Any]],
    target_pass: int | None,
) -> list[str]:
    selected_by_mode = {int(row["mode"]): row for row in selected}
    unavailable: list[str] = []
    for requested_mode in requested_modes:
        if requested_mode not in selected_by_mode:
            unavailable.append(
                f"mode {requested_mode}: requested pass {target_pass} is missing"
                if target_pass is not None
                else f"mode {requested_mode}: result row is missing"
            )
    return unavailable


def _selected_pass_bars(
    selected: list[dict[str, Any]], unavailable: list[str]
) -> list[dict[str, Any]]:
    """One bar group per complete row; partial rows add a message to ``unavailable``."""
    bars: list[dict[str, Any]] = []
    for row in selected:
        row_mode = row["mode"]
        pass_label = (
            "" if "native_pass" not in row else f", pass {row['native_pass']:g}"
        )
        if row.get("status") == "partial" or "normalization_energy_j" not in row:
            unavailable.append(
                f"mode {row_mode}{pass_label}: "
                f"{row.get('combination_error', 'incomplete native evidence')}"
            )
            continue
        labels: list[str] = []
        keys: list[str] = []
        kinds: list[str] = []
        values: list[float] = []
        for surface in row.get("surface_contributions", ()):
            labels.append(surface_tick_label(surface))
            keys.append(surface_key(surface))
            kinds.append(f"surface:{surface['interface_kind']}")
            values.append(float(surface["participation"]))
        for junction in row.get("junctions", ()):
            junction_id = junction["junction_id"]
            labels.extend(
                [
                    f"junction-L:{html.escape(str(junction_id))}",
                    f"junction-C:{html.escape(str(junction_id))}",
                ]
            )
            keys.extend([f"junction-L:{junction_id}", f"junction-C:{junction_id}"])
            kinds.extend(["junction-L", "junction-C"])
            values.extend(
                [
                    float(junction["inductive_participation"]),
                    float(junction["capacitive_participation"]),
                ]
            )
        bars.append(
            {
                "name": f"mode {row_mode}{pass_label}",
                "labels": labels,
                "keys": keys,
                "kinds": kinds,
                "values": values,
            }
        )
    return bars


def _convergence_lines(
    result: EprResult, requested_modes: list[int]
) -> list[dict[str, Any]]:
    """Per-pass lines (energy, surfaces, junctions, frequency) for each mode."""
    last_pass = result.provenance.get("solver_last_completed_pass")
    if type(last_pass) is not int or last_pass <= 0:
        raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
    pass_axis = list(range(1, last_pass + 1))
    lines: list[dict[str, Any]] = []
    for row_mode in requested_modes:
        lines.extend(_mode_lines(result, row_mode, pass_axis))
    return lines


def _mode_lines(
    result: EprResult, row_mode: int, pass_axis: list[int]
) -> list[dict[str, Any]]:
    complete = sorted(
        (
            row
            for row in result.rows
            if row["mode"] == row_mode and row["status"] == "complete"
        ),
        key=lambda item: item["native_pass"],
    )
    series: dict[str, dict[str, Any]] = {}
    frequency_by_pass: dict[int, float] = {}
    for row in result.rows:
        if row["mode"] != row_mode:
            continue
        frequency_hz = usable_frequency(row.get("frequency_hz"))
        if frequency_hz is not None:
            frequency_by_pass[int(row["native_pass"])] = frequency_hz
    for row in complete:
        native = int(row["native_pass"])
        _add_energy_point(series, row_mode, row, native)
        _add_surface_points(series, row_mode, row, native)
        _add_junction_points(series, row_mode, row, native)
    lines = [
        {
            **{key: value for key, value in item.items() if key != "values"},
            "x": pass_axis,
            "y": [item["values"].get(native) for native in pass_axis],
            "secondary_y": False,
        }
        for item in sorted(
            series.values(), key=lambda entry: entry["name"] + entry["identity"]
        )
    ]
    lines.append(
        {
            "name": f"mode {row_mode} frequency",
            "x": pass_axis,
            "y": [frequency_by_pass.get(item) for item in pass_axis],
            "kind": "frequency",
            "identity": f"mode:{row_mode}:frequency",
            "mode": row_mode,
            "secondary_y": True,
        }
    )
    return lines


def _add_energy_point(
    series: dict[str, dict[str, Any]], row_mode: int, row: dict[str, Any], native: int
) -> None:
    energy = series.setdefault(
        "energy",
        {
            "name": f"mode {row_mode} magnetic+inductive / normalization",
            "kind": "energy",
            "identity": "energy",
            "values": {},
        },
    )
    energy["values"][native] = float(row["magnetic_energy_balance_j"]) / float(
        row["normalization_energy_j"]
    )


def _add_surface_points(
    series: dict[str, dict[str, Any]], row_mode: int, row: dict[str, Any], native: int
) -> None:
    for surface in row["surface_contributions"]:
        identity = surface_key(surface)
        item = series.setdefault(
            identity,
            {
                "name": f"mode {row_mode} {surface_label(surface)}",
                "kind": "surface",
                "interface_kind": surface["interface_kind"],
                "identity": identity,
                "hover": surface_hover(surface),
                "source_metadata": surface_metadata(surface),
                "mode": row_mode,
                "values": {},
            },
        )
        item["values"][native] = float(surface["participation"])


def _add_junction_points(
    series: dict[str, dict[str, Any]], row_mode: int, row: dict[str, Any], native: int
) -> None:
    for junction in row["junctions"]:
        for kind, field in (
            ("junction-L", "inductive_participation"),
            ("junction-C", "capacitive_participation"),
        ):
            identity = f"{kind}:{junction['junction_id']}"
            item = series.setdefault(
                identity,
                {
                    "name": f"mode {row_mode} {html.escape(identity)}",
                    "kind": kind,
                    "identity": identity,
                    "mode": row_mode,
                    "values": {},
                },
            )
            item["values"][native] = float(junction[field])
