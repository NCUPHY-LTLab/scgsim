"""The public layout check: will this component simulate?

Owns the order of the rules. Rules 1 and 2 read the component and always report. Rules 3 to 5 are the
production stages, so a layout that passes them reaches ``mesh()`` without a geometry refusal; they run
only when rule 2 has nothing to report, because a broken declaration would only make them refuse the
same record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.drc.polygons import polygon_findings
from scgsim.geometry.drc.report import (
    PHYSICAL_INTERPRETATION,
    SGB_STRUCTURE,
    SURFACE_EPR_BANDS,
    LayoutCheckReport,
    SkippedRule,
)
from scgsim.geometry.drc.semantics import entities_by_layer, semantic_findings
from scgsim.geometry.drc.stages import StageInputs, run_production_stages


def simulation_drc(  # noqa: PLR0913 - the plan's inputs plus the optional rule controls
    component: Any,
    *,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Any] | None = None,
    nets: Mapping[str, Sequence[str]] | None = None,
    inset_margins_um: Sequence[float] | None = None,
    root_id: str = "chip",
    coupon_padding_um: float = 0.0,
    vacuum_padding_um: Any = None,
    min_feature_um: float | None = None,
) -> LayoutCheckReport:
    """Check whether ``component`` reaches ``mesh()`` without a geometry refusal, and report every finding.

    ``layer_stack``, ``material_records``, ``instances`` and ``nets`` are what ``GeometryPlan`` takes:
    ``instances`` maps a registered path such as ``"chip/device"`` to the reference that places it, and
    ``nets`` maps a Net ID to its qualified Entity IDs.

    Rule 5 (Surface-EPR bands) runs when ``inset_margins_um`` is given. The other keywords carry what
    ``GeometryPlan`` needs and the signature of the check does not otherwise provide; each defaults to
    the plan's own behavior:

    * ``root_id`` is the first segment of every path in ``instances``.
    * ``coupon_padding_um`` is the plan's required padding around the component, as ``0.0`` (none).
    * ``vacuum_padding_um`` is what ``GeometryPlan.set_vacuum_region`` takes; ``None`` sets no region.
    * ``min_feature_um`` is the smallest model edge or face width you accept; ``None`` reports no
      sliver, because SCGSim sets no such limit itself.
    """
    registered = dict(instances or {})
    cells = {root_id: component} | _registered_cells(registered)
    findings = polygon_findings(component, entities_by_layer(cells))
    semantic = semantic_findings(
        component, root_id=root_id, instances=registered, layer_stack=layer_stack
    )
    findings.extend(semantic)
    if semantic:
        skipped = _skipped_stages("semantic-contract findings must be fixed first")
        return LayoutCheckReport(tuple(findings), skipped)
    outcome = run_production_stages(
        StageInputs(
            component=component,
            root_id=root_id,
            layer_stack=layer_stack,
            material_records=material_records,
            instances=registered,
            nets=nets,
            coupon_padding_um=coupon_padding_um,
            vacuum_padding_um=vacuum_padding_um,
            inset_margins_um=inset_margins_um,
            min_feature_um=min_feature_um,
        )
    )
    return LayoutCheckReport((*findings, *outcome.findings), tuple(outcome.skipped))


def _registered_cells(instances: Mapping[str, Any]) -> dict[str, Any]:
    """The cell of each registered instance that has one; rule 2 reports the others."""
    return {
        path: reference.cell
        for path, reference in instances.items()
        if getattr(reference, "cell", None) is not None
    }


def _skipped_stages(reason: str) -> tuple[SkippedRule, ...]:
    return tuple(
        SkippedRule(rule, reason)
        for rule in (PHYSICAL_INTERPRETATION, SGB_STRUCTURE, SURFACE_EPR_BANDS)
    )
