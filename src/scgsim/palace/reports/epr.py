"""Render one detached EPR mode with surface, bulk and signed-port axes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _plain_view(value: Any) -> Any:
    """Convert immutable EPR containers into values suitable for Plotly hover metadata."""
    if isinstance(value, Mapping):
        return {str(key): _plain_view(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_view(item) for item in value]
    return value


def _show_view(result: Any, *, mode: int | None) -> dict[str, Any]:
    """Select exactly one mode and summarize known whole-surface loss and coverage."""
    selected = [row for row in result.rows if mode is None or row["mode"] == mode]
    if len(selected) != 1:
        raise ValueError("show_epr requires one exact mode")
    row = selected[0]
    surface = row["surface"]
    baselines = [
        item for item in surface if item["evaluation_kind"] == "unmasked_baseline"
    ]
    known_baselines = [
        item["inverse_q"] for item in baselines if item["inverse_q"] is not None
    ]
    return {
        "row": row,
        "coverage": {
            "surface_rows": len(surface),
            "source_surfaces": len({item["name"] for item in surface}),
            "margin_rows": sum(
                item["evaluation_kind"] == "requested_margin" for item in surface
            ),
            "bulk_domains": len(row["bulk"]),
            "ports": len(row["junction"]),
            "known_baseline_subtotal": (
                f"{sum(known_baselines):.6g}" if known_baselines else "unavailable"
            ),
            "unknown_baseline_losses": sum(
                item["inverse_q"] is None for item in baselines
            ),
        },
    }


def _surface_label(item: Mapping[str, Any]) -> str:
    """Label a surface row by its Palace index, entry name, Entities and interface, with its margin in micrometres.

    The Palace ``Dielectric`` index is unique per row, so two margins of one baseline that print alike (1000.001
    and 1000.002 um both read ``1000 µm``) stay apart; the entry name keeps the label readable.
    """
    entities = ",".join(item["entities"])
    label = (
        f"[{item['index']}] {item['name']}: {entities} / {item['interface_kind']} / "
        f"{item['evaluation_kind']}"
    )
    return f"{label} {item['margin_um']:g} µm"


def show_epr(result: Any, *, mode: int | None = None) -> Any:
    """Plot one mode with separate surface, bulk, and signed Palace port axes."""

    view = _show_view(result, mode=mode)
    row = view["row"]
    from plotly import graph_objects as go  # noqa: PLC0415 -- Defer heavy Plotly loading until a figure is requested.
    from plotly.subplots import make_subplots  # noqa: PLC0415 -- Defer heavy Plotly loading until a figure is requested.

    figure = make_subplots(
        rows=3, cols=1, subplot_titles=("Surface", "Bulk", "Palace port")
    )
    surface = row["surface"]
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in surface],
            y=[_surface_label(item) for item in surface],
            customdata=[
                json.dumps(
                    _plain_view(
                        {
                            "group_id": item["group_id"],
                            "index": item["index"],
                            "name": item["name"],
                            "rows": item["rows"],
                            "face_kinds": item["face_kinds"],
                            "nets": item["nets"],
                            "bands": item.get("bands"),
                            "lineage": item["lineage"],
                            "inverse_q": item["inverse_q"],
                            "assumptions": item["assumptions"],
                            "original_assumptions": item["original_assumptions"],
                            "recorded_native_assumptions": item.get(
                                "recorded_native_assumptions"
                            ),
                            "assumption_update": item.get("assumption_update"),
                        }
                    ),
                    sort_keys=True,
                )
                for item in surface
            ],
            hovertemplate="%{y}<br>p=%{x:.6g}<br>%{customdata}<extra></extra>",
            orientation="h",
            name="surface p",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in row["bulk"]],
            y=[item["domain_id"] for item in row["bulk"]],
            orientation="h",
            name="bulk p",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in row["junction"]],
            y=[item["port_name"] for item in row["junction"]],
            orientation="h",
            name="signed port p",
        ),
        row=3,
        col=1,
    )
    coverage = view["coverage"]
    figure.add_annotation(
        text=(
            f"Coverage: {coverage['surface_rows']} surface rows / "
            f"{coverage['source_surfaces']} source surfaces; "
            f"{coverage['bulk_domains']} bulk domains; {coverage['ports']} ports. "
            f"Known baseline 1/Q subtotal: {coverage['known_baseline_subtotal']}; "
            f"{coverage['unknown_baseline_losses']} baseline losses unknown. "
            f"{coverage['margin_rows']} margin rows are displayed separately "
            "and excluded from the whole-surface subtotal. "
            "Sidewalls uncomputed when unselected; requested margins are alternatives."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
    )
    figure.update_layout(title=f"Palace EPR mode {row['mode']}", height=900)
    return figure
