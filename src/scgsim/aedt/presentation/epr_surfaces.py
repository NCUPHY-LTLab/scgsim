"""Shared formatting of EPR surface contributions and raw native-face references.

EPR is the energy participation ratio: the share of a mode's electric energy held in
one region, which sets that region's contribution to loss (1/Q = p * tan(delta)).

Owns: stable keys, labels, hover text and tables for the recorded surface records.
Does not own: choosing which rows to show (``epr_history.py``, ``epr.py``) or
computing participation (``scgsim.aedt.epr``).
"""

from __future__ import annotations

import html
import json
import math
import textwrap
from collections.abc import Mapping
from typing import Any

from scgsim.aedt.epr.models import detached
from scgsim.presentation.notebook import (
    details,
    display_text,
    json_details,
    table,
)


def usable_frequency(value: Any) -> float | None:
    """A positive, finite frequency in Hz, or None when the recorded value cannot be plotted."""
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


def surface_label_parts(surface: Any) -> tuple[str, str, str]:
    """Owner, interface kind and margin text that together name one surface evaluation."""
    owner = surface.get("owner_id", surface.get("contribution_id", "owner"))
    readable_owner = str(owner).replace("_", " ")
    evaluation = (
        "Unmasked"
        if surface.get("evaluation_kind") == "unmasked_baseline"
        else f"Requested {surface['margin_um']:g} µm"
    )
    return readable_owner, str(surface["interface_kind"]), evaluation


def surface_label(surface: Any) -> str:
    """Single-line HTML-escaped surface name."""
    return " · ".join(
        html.escape(part, quote=False) for part in surface_label_parts(surface)
    )


def surface_tick_label(surface: Any) -> str:
    """Wrap category labels so long source names cannot consume the plot area."""

    raw = " · ".join(surface_label_parts(surface))
    parts = textwrap.wrap(raw, width=28, break_long_words=True, break_on_hyphens=False)
    return "<br>".join(html.escape(part, quote=False) for part in parts)


def surface_hover(surface: Any) -> str:
    """Multi-line hover text with the surface metadata, a few members and the film assumptions."""
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


def native_reference_value(reference: Any, *keys: str) -> str:
    """Text for the first of the given keys present in a native reference record."""
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
                + native_reference_value(reference, "reference_id")
                + "\nContributions: "
                + native_reference_value(reference, "contribution_ids", "binding_ids"),
                "Owner: "
                + native_reference_value(
                    reference, "owner_semantic_id", "source_polygon_id"
                )
                + "\nObject: "
                + native_reference_value(reference, "object_name", "native_object_id"),
                "Interfaces: "
                + native_reference_value(reference, "interface_kinds")
                + "\nCap: "
                + native_reference_value(reference, "source_cap")
                + "\nSource Z (µm): "
                + native_reference_value(reference, "source_z_um"),
                "Source: "
                + native_reference_value(reference, "source_supports")
                + "\nIncident domains: "
                + native_reference_value(reference, "incident_domains"),
                "Selection: "
                + native_reference_value(reference, "selection_name")
                + "\nFace List: "
                + native_reference_value(reference, "face_list")
                + "\nFace IDs: "
                + native_reference_value(reference, "face_ids", "native_faces"),
                "Native normal: "
                + native_reference_value(reference, "native_normal")
                + "\nProjection: "
                + native_reference_value(reference, "projection_normal")
                + "\nBasis: "
                + native_reference_value(reference, "sampling_basis")
                + "\nSupport: "
                + native_reference_value(reference, "support_kind"),
                native_reference_value(reference, "normal_integral_v2"),
                native_reference_value(reference, "tangential_integral_v2"),
                native_reference_value(reference, "native_area_m2"),
                native_reference_value(reference, "area_m2"),
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
    """Figure height in pixels for a horizontal bar ranking of this many rows."""
    return max(460, 180 + 52 * count)
