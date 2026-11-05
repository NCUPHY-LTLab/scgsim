"""The quality report: a JSON document and a short text summary of one meshing run.

Owns what a run says about itself: the rule-set version and parameters, the threshold, how the loop
stopped, the counts, and the bad elements that remain with their location, groups and feature kinds.  All
timings sit under one ``timing`` key, so two runs of the same input compare equal without it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

from .findings import QUALITY_MEASURE, BadElement
from .policy import RuleSet
from .repair import Budgets, RepairOutcome

REPORT_VERSION = "scgsim.meshing.report/1"
MAX_LISTED_ELEMENTS = 200


class QualityReport(TypedDict):
    """Shape of the report; the nested records are plain JSON mappings."""

    report_version: str
    rules: dict[str, Any]
    quality_measure: str
    threshold: float
    stop: str
    rounds: int
    budgets: dict[str, Any]
    mesh: dict[str, Any] | None
    history: list[dict[str, Any]]
    remaining_bad: dict[str, Any]
    timing: dict[str, Any]


def build_report(
    outcome: RepairOutcome,
    *,
    threshold: float,
    ruleset: RuleSet,
    budgets: Budgets,
    total_seconds: float,
) -> QualityReport:
    """The report of a finished loop.  ``mesh`` is ``None`` when no round produced a mesh."""
    audit = outcome.last_audit
    return QualityReport(
        report_version=REPORT_VERSION,
        rules={"version": ruleset.version, "parameters": asdict(ruleset)},
        quality_measure=QUALITY_MEASURE,
        threshold=threshold,
        stop=outcome.stop,
        rounds=outcome.rounds,
        budgets=asdict(budgets),
        mesh=None
        if audit is None
        else {
            "tetrahedra": audit.tetrahedra,
            "nodes": audit.nodes,
            "min_quality": audit.min_quality,
            "min_gamma": audit.min_gamma,
            "bad_count": audit.bad_count,
            "algorithm_3d": outcome.algorithm_3d,
            "fallback": outcome.fallback,
            "refined_curves": len(outcome.curve_sizes),
            "smallest_size_um": min(outcome.curve_sizes.values(), default=None),
        },
        history=[asdict(record) for record in outcome.history],
        remaining_bad=_remaining_bad(
            audit.bad if audit else (), audit.bad_count if audit else 0
        ),
        timing={
            "total_seconds": total_seconds,
            "round_seconds": list(outcome.round_seconds),
            "mesh_seconds": audit.mesh_seconds if audit else None,
            "audit_seconds": audit.audit_seconds if audit else None,
        },
    )


def render_text(report: QualityReport) -> str:
    """A short human summary of ``report``."""
    mesh = report["mesh"]
    lines = [
        f"scgsim.meshing {report['rules']['version']}: {_headline(report)}",
        f"  quality measure {report['quality_measure']}, threshold {report['threshold']:g}",
    ]
    if mesh is None:
        lines.append("  no round produced a mesh")
    else:
        fallback = f", fallback after {mesh['fallback']}" if mesh["fallback"] else ""
        lines.append(
            f"  {mesh['tetrahedra']} tetrahedra, min quality {mesh['min_quality']:.4g}, "
            f"{mesh['bad_count']} below the threshold; algorithm {mesh['algorithm_3d']}{fallback}"
        )
        lines.append(
            f"  {mesh['refined_curves']} refined curves, smallest size "
            f"{_size(mesh['smallest_size_um'])}"
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
    lines.append(f"  total {report['timing']['total_seconds']:.1f} s")
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


def _size(size: float | None) -> str:
    return "-" if size is None else f"{size:.4g} um"
