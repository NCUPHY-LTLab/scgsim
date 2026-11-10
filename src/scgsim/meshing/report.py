"""The quality report: a JSON document and a short text summary of one meshing run.

Owns what a run says about itself: the recipe (name, version, parameters), the repair policy, the mesher
options and the budgets, the time budget of the model, the threshold, how the loop stopped, the counts,
and the bad elements that remain with their location, groups and feature kinds.  The counts and the
minimum quality are measured on the linear mesh, before any elevation to a higher order; the report says
so.  All timings sit under one
``timing`` key, so two runs of the same input compare equal without it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TypedDict

from .findings import QUALITY_MEASURE, BadElement
from .mesher import MesherOptions
from .recipes import SizingRecipe
from .repair import Budgets, RepairOutcome, RepairPolicy, TimeBudget

REPORT_VERSION = "scgsim.meshing.report/3"
MAX_LISTED_ELEMENTS = 200
MEASURED_ON = "linear mesh before elevation"


class QualityReport(TypedDict):
    """Shape of the report; the nested records are plain JSON mappings."""

    report_version: str
    recipe: dict[str, Any]
    repair: dict[str, Any]
    mesher: dict[str, Any]
    quality_measure: str
    threshold: float
    stop: str
    rounds: int
    budgets: dict[str, Any]
    time_budget: dict[str, Any]
    mesh: dict[str, Any] | None
    history: list[dict[str, Any]]
    remaining_bad: dict[str, Any]
    timing: dict[str, Any]


@dataclass(frozen=True)
class RunSettings:
    """What the caller chose for a run, as the report records it."""

    threshold: float
    recipe: SizingRecipe
    repair: RepairPolicy
    mesher: MesherOptions
    budgets: Budgets
    time_budget: TimeBudget  # the limit and target for this model's footprint


def recipe_record(recipe: SizingRecipe) -> dict[str, Any]:
    """A recipe as the report and the Palace mesh manifest record it: name, version, and dataclass fields."""
    return {
        "name": recipe.name,
        "version": recipe.version,
        "parameters": asdict(recipe) if is_dataclass(recipe) else {},  # type: ignore[call-overload]
    }


def build_report(
    outcome: RepairOutcome, settings: RunSettings, total_seconds: float
) -> QualityReport:
    """The report of a finished loop.  ``mesh`` is ``None`` when no round produced a mesh."""
    audit = outcome.last_audit
    return QualityReport(
        report_version=REPORT_VERSION,
        recipe=recipe_record(settings.recipe),
        repair=asdict(settings.repair),
        mesher=asdict(settings.mesher),
        quality_measure=QUALITY_MEASURE,
        threshold=settings.threshold,
        stop=outcome.stop,
        rounds=outcome.rounds,
        budgets=asdict(settings.budgets),
        time_budget=asdict(settings.time_budget),
        mesh=None
        if audit is None
        else {
            "tetrahedra": audit.tetrahedra,
            "nodes": audit.nodes,
            "min_quality": audit.min_quality,
            "min_gamma": audit.min_gamma,
            "bad_count": audit.bad_count,
            "measured_on": MEASURED_ON,
            "geometry_order": outcome.geometry_order,
            "algorithm_3d": outcome.algorithm_3d,
            "fallback": outcome.fallback,
            "sized_curves": len(outcome.curve_sizes),
            "local_sizes": len(outcome.local_sizes),
            "smallest_local_size_um": min(
                (ball.size for ball in outcome.local_sizes), default=None
            ),
        },
        history=[asdict(record) for record in outcome.history],
        remaining_bad=_remaining_bad(
            audit.bad if audit else (), audit.bad_count if audit else 0
        ),
        timing={
            "total_seconds": total_seconds,
            "target_met": total_seconds <= settings.time_budget.target_seconds,
            "round_seconds": list(outcome.round_seconds),
            "mesh_seconds": audit.mesh_seconds if audit else None,
            "audit_seconds": audit.audit_seconds if audit else None,
        },
    )


def render_text(report: QualityReport) -> str:
    """A short human summary of ``report``."""
    mesh = report["mesh"]
    lines = [
        f"scgsim.meshing {report['recipe']['version']}: {_headline(report)}",
        f"  quality measure {report['quality_measure']}, threshold {report['threshold']:g}",
    ]
    if mesh is None:
        lines.append("  no round produced a mesh")
    else:
        fallback = f", fallback after {mesh['fallback']}" if mesh["fallback"] else ""
        lines.append(
            f"  {mesh['tetrahedra']} tetrahedra, min quality {mesh['min_quality']:.4g}, "
            f"{mesh['bad_count']} below the threshold ({mesh['measured_on']}); "
            f"algorithm {mesh['algorithm_3d']}{fallback}, geometry order {mesh['geometry_order']}"
        )
        lines.append(
            f"  {mesh['sized_curves']} sized curves, {mesh['local_sizes']} local sizes, smallest "
            f"{_size(mesh['smallest_local_size_um'])}"
        )
    remaining = report["remaining_bad"]
    for location, count in remaining["by_location"].items():
        lines.append(f"  remaining bad, {location}: {count}")
    for element in remaining["listed"][:5]:
        lines.append(
            f"    element {element['element']} quality {element['quality']:.3g} "
            f"{element['location']} groups {', '.join(element['groups']) or '-'} "
            f"features {', '.join(element['feature_kinds']) or '-'}"
        )
    budget = report["time_budget"]
    lines.append(
        f"  total {report['timing']['total_seconds']:.1f} s; limit {budget['limit_seconds']:.0f} s, "
        f"target {budget['target_seconds']:.0f} s for {_footprint(budget['footprint_mm2'])}"
    )
    return "\n".join(lines) + "\n"


def write_report(report: QualityReport, json_path: Path, text_path: Path) -> None:
    """Write the report as JSON to ``json_path`` and its text summary to ``text_path``."""
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text_path.write_text(render_text(report), encoding="utf-8")


def _remaining_bad(bad: tuple[BadElement, ...], bad_count: int) -> dict[str, Any]:
    """Counts by location and the worst elements.  Elements the audit did not attribute are only counted."""
    by_location: dict[str, int] = {}
    for element in bad:
        by_location[element.location] = by_location.get(element.location, 0) + 1
    worst = sorted(bad, key=lambda e: (e.quality, e.element))[:MAX_LISTED_ELEMENTS]
    return {
        "count": bad_count,
        "by_location": dict(sorted(by_location.items())),
        "listed": [_listed(element) for element in worst],
    }


def _listed(element: BadElement) -> dict[str, Any]:
    return {
        "element": element.element,
        "quality": element.quality,
        "centroid": list(element.centroid),
        "shortest_edge_midpoint": list(element.shortest_edge_midpoint),
        "edge_min": element.edge_min,
        "edge_max": element.edge_max,
        "location": element.location,
        "curves": list(element.curves),
        "points": list(element.points),
        "sgb_ids": list(element.sgb_ids),
        "groups": list(element.groups),
        "feature_kinds": list(element.feature_kinds),
    }


def _headline(report: QualityReport) -> str:
    rounds = report["rounds"]
    plural = "" if rounds == 1 else "s"
    if report["stop"] == "clean":
        return f"clean after {rounds} refinement round{plural}"
    return f"stopped without a clean mesh ({report['stop']}) after {rounds} refinement round{plural}"


def _footprint(footprint_mm2: float | None) -> str:
    return (
        "an unread footprint" if footprint_mm2 is None else f"{footprint_mm2:.3g} mm2"
    )


def _size(size: float | None) -> str:
    return "-" if size is None else f"{size:.4g} um"
