"""Detached EPR result projections shared by plots and run summaries."""

from __future__ import annotations

import html
import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.aedt.epr.models import EprResult, detached
from scgsim.presentation.notebook import (
    details,
    display_text,
    json_details,
    table,
)


def saved_solution_version_text(provenance: Mapping[str, Any]) -> str:
    """Describe the captured solution's recorded origin, including missingness."""

    source = provenance.get("saved_solution_aedt_version")
    if isinstance(source, str) and source:
        return f"Source AEDT version: {html.escape(source)}"
    return "Source AEDT version not recorded (來源版本未記錄)"


def saved_solution_versions_text(provenance: Mapping[str, Any]) -> str:
    """Keep saved origin, requested release, and connected release distinct."""

    parts = [saved_solution_version_text(provenance)]
    for key, label in (
        ("requested_aedt_version", "Requested AEDT version"),
        ("connected_aedt_version", "Connected AEDT version"),
    ):
        value = provenance.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{label}: {html.escape(value)}")
    return " · ".join(parts)


def usable_frequency(value: Any) -> float | None:
    """Return a finite positive frequency, or None for unusable input."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        frequency_hz = float(value)
    except OverflowError:
        return None
    return frequency_hz if math.isfinite(frequency_hz) and frequency_hz > 0 else None


def surface_key(surface: Any) -> str:
    """Use recorded scientific identity, never a shortened display label."""

    members = surface.get("members", surface.get("binding_ids", ()))
    return json.dumps(
        {
            "group_id": surface.get("group_id"),
            "contribution_id": surface.get("contribution_id"),
            "owner_id": surface.get("owner_id"),
            "interface_kind": surface.get("interface_kind"),
            "evaluation_kind": surface.get("evaluation_kind", "requested_margin"),
            "margin_um": surface.get("margin_um"),
            "field_side": surface.get("field_side"),
            "members": detached(members),
        },
        sort_keys=True,
    )


def _surface_label_parts(surface: Any) -> tuple[str, str, str]:
    owner = surface.get("owner_id", surface.get("contribution_id", "owner"))
    readable_owner = str(owner).replace("_", " ")
    evaluation = (
        "Unmasked"
        if surface.get("evaluation_kind") == "unmasked_baseline"
        else f"Requested {surface['margin_um']:g} µm"
    )
    return readable_owner, str(surface["interface_kind"]), evaluation


def surface_label(surface: Any) -> str:
    """Build an HTML-safe display label for a surface contribution."""
    return " · ".join(
        html.escape(part, quote=False) for part in _surface_label_parts(surface)
    )


def surface_tick_label(surface: Any) -> str:
    """Wrap category labels so long source names cannot consume the plot area."""

    raw = " · ".join(_surface_label_parts(surface))
    parts = textwrap.wrap(raw, width=28, break_long_words=True, break_on_hyphens=False)
    return "<br>".join(html.escape(part, quote=False) for part in parts)


def surface_hover(surface: Any) -> str:
    """Render source and assumption facts into an escaped hover description."""
    metadata = {
        "group_id": surface.get("group_id"),
        "owner_id": surface.get("owner_id"),
        "interface_kind": surface.get("interface_kind"),
        "evaluation_kind": surface.get("evaluation_kind"),
        "margin_um": surface.get("margin_um"),
        "field_side": surface.get("field_side"),
        "members": detached(surface.get("members", surface.get("binding_ids"))),
        "source": surface.get("source"),
        "assumptions": detached(surface.get("assumptions")),
        "original_assumptions": detached(surface.get("original_assumptions")),
        "assumption_update": detached(surface.get("assumption_update")),
        "inverse_q": surface.get("inverse_q"),
    }
    members = metadata["members"] or []
    lines = [
        ("Owner", metadata["owner_id"]),
        ("Group", metadata["group_id"]),
        ("Interface", metadata["interface_kind"]),
        ("Evaluation", metadata["evaluation_kind"]),
        ("Margin (µm)", metadata["margin_um"]),
        ("Side", metadata["field_side"]),
        ("Source", metadata["source"]),
        ("Members", len(members)),
    ]
    for index, member in enumerate(members[:3], 1):
        if isinstance(member, dict):
            identity = ", ".join(
                f"{key}={member[key]}"
                for key in (
                    "binding_id",
                    "contribution_id",
                    "source_polygon_id",
                    "field_side",
                )
                if key in member
            )
        else:
            identity = str(member)
        lines.append((f"Member {index}", identity))
    if len(members) > 3:
        lines.append(
            (
                "Additional members",
                f"{len(members) - 3}; full records in trace metadata",
            )
        )
    assumptions = metadata["assumptions"]
    if isinstance(assumptions, dict):
        lines.extend(
            (name.replace("_", " "), value) for name, value in assumptions.items()
        )
    rendered = []
    for label, value in lines:
        wrapped = textwrap.wrap(
            str(display_text(value)),
            width=58,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        rendered.append(
            f"<b>{html.escape(label, quote=False)}</b>: "
            + "<br>".join(html.escape(part, quote=False) for part in wrapped)
        )
    return "<br>".join(rendered)


def surface_metadata(surface: Any) -> dict[str, Any]:
    """Keep the full source record independently of a shortened hover display."""

    return detached(surface)


def _native_reference_value(reference: Any, *keys: str) -> str:
    for key in keys:
        if key in reference:
            value = reference[key]
            if value is None:
                return "null"
            if isinstance(value, (Mapping, list, tuple)):
                return json.dumps(detached(value), ensure_ascii=False, sort_keys=True)
            return display_text(value)
    return "not recorded"


def native_reference_table_data(
    references: Any,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Present raw native-face references without deriving participation or loss."""

    headers = (
        "Reference / contribution",
        "Native owner / object",
        "Interface / source cap / Z",
        "Source supports / incident domains",
        "Face List / faces",
        "Projection / basis / support",
        "Normal integral (V²)",
        "Tangential integral (V²)",
        "Native face area (m²)",
        "Calculator area (m²)",
    )
    rows = []
    for reference in references:
        rows.append(
            (
                "Reference: "
                + _native_reference_value(reference, "reference_id")
                + "\nContributions: "
                + _native_reference_value(reference, "contribution_ids", "binding_ids"),
                "Owner: "
                + _native_reference_value(
                    reference, "owner_semantic_id", "source_polygon_id"
                )
                + "\nObject: "
                + _native_reference_value(reference, "object_name", "native_object_id"),
                "Interfaces: "
                + _native_reference_value(reference, "interface_kinds")
                + "\nCap: "
                + _native_reference_value(reference, "source_cap")
                + "\nSource Z (µm): "
                + _native_reference_value(reference, "source_z_um"),
                "Source: "
                + _native_reference_value(reference, "source_supports")
                + "\nIncident domains: "
                + _native_reference_value(reference, "incident_domains"),
                "Selection: "
                + _native_reference_value(reference, "selection_name")
                + "\nFace List: "
                + _native_reference_value(reference, "face_list")
                + "\nFace IDs: "
                + _native_reference_value(reference, "face_ids", "native_faces"),
                "Native normal: "
                + _native_reference_value(reference, "native_normal")
                + "\nProjection: "
                + _native_reference_value(reference, "projection_normal")
                + "\nBasis: "
                + _native_reference_value(reference, "sampling_basis")
                + "\nSupport: "
                + _native_reference_value(reference, "support_kind"),
                _native_reference_value(reference, "normal_integral_v2"),
                _native_reference_value(reference, "tangential_integral_v2"),
                _native_reference_value(reference, "native_area_m2"),
                _native_reference_value(reference, "area_m2"),
            )
        )
    return headers, rows


def native_reference_html(row: Any, *, opened: bool = False) -> str:
    """Render recorded native references as separate, raw evidence."""

    if "native_surface_references" not in row:
        return ""
    references = row["native_surface_references"]
    note = (
        "<p>These raw native-owner face references are separate from the masked "
        "Sheet contributions; their supports may differ. Native face area and "
        "calculator area are separate records. An SA reference identifies its "
        "dielectric native owner separately from the existing vacuum-formula "
        "basis. No participation, loss, or equivalence is inferred.</p>"
    )
    if references is None:
        return note + "<p>The recorded native reference value is null.</p>"
    if not references:
        return note + "<p>The recorded native reference sequence is empty.</p>"
    headers, rows = native_reference_table_data(references)
    return note + details(
        "Native-face raw references",
        table(headers, rows)
        + json_details(
            "Complete native-face reference records",
            {"native_surface_references": detached(references)},
        ),
        opened=opened,
    )


def ranking_height(count: int) -> int:
    """Choose enough plot height for the requested number of ranking rows."""
    return max(460, 180 + 52 * count)


def history_view(
    result: EprResult,
    *,
    mode: int | None,
    native_pass: int | None,
    show_convergence: bool,
) -> dict[str, Any]:
    """Project selected EPR rows into bar and adaptive-history series."""
    requested_modes = _requested_modes(result, mode)
    target_pass = _target_pass(result, native_pass)
    selected_rows = _selected_history_rows(result, mode, target_pass)
    unavailable = _missing_mode_rows(requested_modes, selected_rows, target_pass)
    bars = _participation_bars(selected_rows, unavailable)
    convergence_view = show_convergence and result.result_kind == "adaptive_history"
    lines = _convergence_lines(result, requested_modes) if convergence_view else []
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
    """Select requested modes while preserving the source result's order."""
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
    """Resolve the default final pass for adaptive-history results."""
    target_pass = native_pass
    if result.result_kind == "adaptive_history" and target_pass is None:
        observed_last = result.provenance.get("solver_last_completed_pass")
        if type(observed_last) is not int or observed_last <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        target_pass = observed_last
    return target_pass


def _selected_history_rows(
    result: EprResult, mode: int | None, target_pass: int | None
) -> list[Mapping[str, Any]]:
    """Keep exact requested mode/pass filtering in its existing row order."""
    return [
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (
            result.result_kind != "adaptive_history"
            or row["native_pass"] == target_pass
        )
    ]


def _missing_mode_rows(
    requested_modes: list[int],
    selected_rows: list[Mapping[str, Any]],
    target_pass: int | None,
) -> list[str]:
    """Describe requested modes absent from the selected result rows."""
    selected_by_mode = {int(row["mode"]): row for row in selected_rows}
    unavailable: list[str] = []
    for requested_mode in requested_modes:
        if requested_mode not in selected_by_mode:
            unavailable.append(
                f"mode {requested_mode}: requested pass {target_pass} is missing"
                if target_pass is not None
                else f"mode {requested_mode}: result row is missing"
            )
    return unavailable


def _participation_bars(
    selected_rows: list[Mapping[str, Any]], unavailable: list[str]
) -> list[dict[str, Any]]:
    """Project complete result rows into contribution bars, recording partial rows."""
    bars: list[dict[str, Any]] = []
    for row in selected_rows:
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
            labels.extend(
                [
                    f"junction-L:{html.escape(str(junction['junction_id']))}",
                    f"junction-C:{html.escape(str(junction['junction_id']))}",
                ]
            )
            keys.extend(
                [
                    f"junction-L:{junction['junction_id']}",
                    f"junction-C:{junction['junction_id']}",
                ]
            )
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
    """Project adaptive history into independent energy, surface, and frequency lines."""
    last_pass = result.provenance.get("solver_last_completed_pass")
    if type(last_pass) is not int or last_pass <= 0:
        raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
    pass_axis = list(range(1, last_pass + 1))
    lines: list[dict[str, Any]] = []
    for row_mode in requested_modes:
        lines.extend(_mode_convergence_lines(result, row_mode, pass_axis))
    return lines


def _mode_convergence_lines(
    result: EprResult, row_mode: int, pass_axis: list[int]
) -> list[dict[str, Any]]:
    """Build one mode's adaptive series in stable identity order."""
    complete = sorted(
        (
            row
            for row in result.rows
            if row["mode"] == row_mode and row["status"] == "complete"
        ),
        key=lambda item: item["native_pass"],
    )
    frequency_by_pass = {
        int(row["native_pass"]): frequency
        for row in result.rows
        if row["mode"] == row_mode
        if (frequency := usable_frequency(row.get("frequency_hz"))) is not None
    }
    series = _mode_history_series(complete, row_mode)
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


def _mode_history_series(
    complete_rows: Sequence[Mapping[str, Any]], row_mode: int
) -> dict[str, dict[str, Any]]:
    """Collect energy, surface, and junction values by stable trace identity."""
    series: dict[str, dict[str, Any]] = {}
    for row in complete_rows:
        native = int(row["native_pass"])
        _record_energy_series(series, row_mode, row, native)
        _record_surface_series(series, row_mode, row, native)
        _record_junction_series(series, row_mode, row, native)
    return series


def _record_energy_series(
    series: dict[str, dict[str, Any]],
    row_mode: int,
    row: Mapping[str, Any],
    native: int,
) -> None:
    """Record the displayed magnetic-plus-inductive energy ratio."""
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


def _record_surface_series(
    series: dict[str, dict[str, Any]],
    row_mode: int,
    row: Mapping[str, Any],
    native: int,
) -> None:
    """Record each source-identified surface contribution independently."""
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


def _record_junction_series(
    series: dict[str, dict[str, Any]],
    row_mode: int,
    row: Mapping[str, Any],
    native: int,
) -> None:
    """Record inductive and capacitive junction traces independently."""
    for junction in row["junctions"]:
        for kind, field_name in (
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
            item["values"][native] = float(junction[field_name])


def show_view(
    result: EprResult, *, mode: int | None, native_pass: int | None
) -> dict[str, Any]:
    """Project one exact EPR mode/pass into detached display values."""
    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    selected = [
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (native_pass is None or row.get("native_pass") == native_pass)
    ]
    if len(selected) != 1:
        raise ValueError("show_epr requires one exact mode/pass selection")
    row = selected[0]
    has_native_references = "native_surface_references" in row
    native_references = row.get("native_surface_references")
    if row["status"] != "complete":
        frequency = usable_frequency(row.get("frequency_hz"))
        return {
            "row": row,
            "complete": False,
            "has_native_references": has_native_references,
            "native_references": native_references,
            "frequency_text": (
                f"; verified frequency {frequency:.6g} Hz"
                if frequency is not None
                else ""
            ),
        }
    surfaces = row["surface_contributions"]
    domains = row["electric_domains"]
    junctions = row["junctions"]
    baselines = [
        item for item in surfaces if item.get("evaluation_kind") == "unmasked_baseline"
    ]
    known_baselines = [
        item["inverse_q"] for item in baselines if item.get("inverse_q") is not None
    ]
    return {
        "row": row,
        "complete": True,
        "has_native_references": has_native_references,
        "native_references": native_references,
        "surfaces": surfaces,
        "domains": domains,
        "junctions": junctions,
        "bulk_participations": [
            item["energy_j"] / row["normalization_energy_j"] for item in domains
        ],
        "coverage": {
            "surface_rows": len(surfaces),
            "bindings": sum(
                len(item.get("members", item.get("binding_ids", ())))
                for item in surfaces
            ),
            "bulk_domains": len(domains),
            "junctions": len(junctions),
            "known_baseline_subtotal": (
                f"{sum(known_baselines):.6g}" if known_baselines else "unavailable"
            ),
            "unknown_baseline_losses": sum(
                item.get("inverse_q") is None for item in baselines
            ),
        },
    }
