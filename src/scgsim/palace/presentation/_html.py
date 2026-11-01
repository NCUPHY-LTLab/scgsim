"""HTML fragments and number formatting for Palace run reports.

Small, display-only helpers: table cells, the number formats the report cards use,
and the mesh, surface-loss and notice sections. They take verified values and return
strings. Layout of the whole report and its display methods live in ``reports.py``.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..results.report_data import (
    SurfaceEprSeriesSnapshot,
    SurfaceMaskEprSeriesSnapshot,
)


def _surface_mask_html(snapshot: SurfaceMaskEprSeriesSnapshot) -> str:
    zeroes = sum(
        record.contribution_status == "zero_contribution" for record in snapshot.records
    )
    return (
        "<p style='opacity:0.75'>Native mask Q and energy remain available in "
        "<code>masked_snapshots</code>. The native Inset output does not retain "
        "surface area support; zero participation or energy is reported as "
        "<code>zero_contribution</code>, not as proof of excluded geometry. "
        f"This snapshot contains {zeroes} zero-contribution mask record(s).</p>"
    )


def _surface_loss_html(snapshot: SurfaceEprSeriesSnapshot, problem: str) -> str:
    if snapshot.loss_status == "available":
        q_value = _fmt(snapshot.quality_factor_total)
        detail = "Q_total uses 1 / Σ(p_i tanδ_i); individual Q values are never summed."
    elif snapshot.loss_status == "unavailable_missing":
        q_value = "unavailable"
        detail = "At least one structured surface loss tangent is unavailable."
    else:
        q_value = "unavailable / non-finite"
        detail = (
            "The configured surface-loss sum is zero, so native +inf Q is not "
            "converted to zero. Participation remains available."
        )
    cards = [
        ("Latest source", snapshot.source),
        ("Surface-loss Q_total", q_value),
    ]
    if problem == "Eigenmode":
        cards.append(
            (
                "Surface-loss T1",
                _fmt_seconds(snapshot.t1_seconds)
                if snapshot.t1_seconds is not None
                else "unavailable / non-finite",
            )
        )
    items = "".join(
        "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid "
        "var(--border,#d0d7de);border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem'>"
        f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(label)}</div>"
        f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    return (
        f"<div style='display:flex;flex-wrap:wrap'>{items}</div>"
        f"<p style='opacity:0.75'>{html.escape(detail)}</p>"
    )


def _native_positive_infinity_q_notice(
    traces: Sequence[tuple[str, Sequence[float | None]]],
) -> str | None:
    values = [value for _name, ys in traces for value in ys]
    contains_positive_infinity = any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == math.inf
        for value in values
    )
    finite_or_positive_infinity_only = all(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (math.isfinite(value) or value == math.inf)
        )
        for value in values
    )
    if not contains_positive_infinity or not finite_or_positive_infinity_only:
        return None
    return (
        "<p style='opacity:0.75'>Native-reported Q includes +inf; "
        "a finite-Q convergence view is unavailable.</p>"
    )


def _html_cell(tag: str, text: str) -> str:
    return (
        f"<{tag} style='border:1px solid var(--border,#d0d7de);"
        f"padding:0.35rem 0.65rem;text-align:left'>{html.escape(text)}</{tag}>"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    seconds = float(value)
    if seconds >= 120:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.3g} s"


def _fmt_mb(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    mb = float(value)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GiB"
    return f"{mb:.3g} MiB"


_MESH_REGION_COLUMNS = (
    "Region",
    "Dim",
    "Elements",
    "Gmsh type: count",
    "Edges",
    "Min",
    "Mean",
    "Max",
    "Unit",
)


def _mesh_summary_html(
    summary: Mapping[str, Any], *, card_html: Callable[[str, Any], str]
) -> str:
    """Keep geometric discretization distinct from native solver measurements."""
    if summary["status"] == "not_recorded":
        return "<section><h4>Geometry mesh</h4><p>Mesh observations were not recorded in this historical manifest.</p></section>"
    meshing = summary["meshing"]
    stats = meshing["statistics"]
    items = "".join(card_html(label, value) for label, value in _mesh_cards(summary))
    rows = [_mesh_region_row(region) for region in stats["physical_regions"]]
    header = "".join(_html_cell("th", label) for label in _MESH_REGION_COLUMNS)
    types = ", ".join(
        f"{record['name']}: {record['count']}" for record in stats["element_types"]
    )
    curve_context = "; ".join(
        f"{label}: {'recorded' if summary.get(key) is not None else 'not recorded'}"
        for label, key in (
            ("Source curves", "curve_source"),
            ("Native curve arrangement", "curved_arrangement"),
        )
    )
    return (
        "<section><h4>Geometry mesh</h4>"
        f'<div style="display:flex;flex-wrap:wrap">{items}</div>'
        "<p>The ND estimate uses the configured FEM order and tetrahedral primary-corner topology before boundary constraints; it is not actual solver DOF. Electrostatic H1 estimation is unavailable. Thread options are not measured thread utilization.</p>"
        "<details><summary>Physical regions and mesh discretization</summary>"
        f"<p>Gmsh {html.escape(str(meshing['gmsh_version']))}. {html.escape(types)}.</p>"
        "<p>Edge statistics count unique PRIMARY CORNER CHORDS within each region, not curved arc lengths. Point regions have no edges; missing historical statistics are not recorded.</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<p>{html.escape(curve_context)}. Full controls and provenance remain available in mesh_summary() and report data.</p>"
        "</details></section>"
    )


def _mesh_cards(summary: Mapping[str, Any]) -> list[tuple[str, Any]]:
    meshing = summary["meshing"]
    estimate = summary["dof_estimate"]
    stats, effective = meshing["statistics"], meshing["effective"]
    return [
        (
            "Geometry / FEM order",
            f"{effective['geometry_order']} / {_fmt(summary['finite_element_order'])}",
        ),
        (
            "Native nodes / elements",
            f"{stats['node_count']} / {stats['element_count']}",
        ),
        ("3D algorithm", effective["algorithm_3d"]),
        (
            "Volume / surface thread options",
            f"{_fmt(effective['volume_threads'])} / {_fmt(effective['surface_threads_2d'])}",
        ),
        ("HighOrder optimization", meshing["high_order_optimization"]["status"]),
        ("ND topological DOF estimate", _fmt(estimate.get("value"))),
    ]


def _mesh_region_row(region: Mapping[str, Any]) -> str:
    chords = region.get("corner_chord_edge_lengths")
    values = [
        region["name"],
        region["dimension"],
        region["element_count"],
        ", ".join(
            f"{code}: {count}" for code, count in region["element_types"].items()
        ),
    ]
    values.extend(
        [
            chords["count"],
            *(_fmt(chords[key]) for key in ("minimum", "mean", "maximum")),
            chords["unit"],
        ]
        if chords is not None
        else ["not recorded"] * 5
    )
    return "<tr>" + "".join(_html_cell("td", str(value)) for value in values) + "</tr>"


def card_html(label: Any, value: Any) -> str:
    """One labelled value box of the report's card rows."""
    return (
        "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid var(--border,#d0d7de);"
        "border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem;'>"
        f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(str(label))}</div>"
        f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(str(value))}</div>"
        "</div>"
    )
