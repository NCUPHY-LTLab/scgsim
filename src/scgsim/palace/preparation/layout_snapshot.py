"""Turn a layout Component into the ``GeometryPlan`` snapshot that ``simulation_drc`` checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.geometry.source.plan import GeometryPlan, GeometryPlanSnapshot

_ROOT_ID = "chip"


def prepare_layout_snapshot(
    component: Any,
    *,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Any] | None,
) -> GeometryPlanSnapshot:
    """The prepared snapshot a user would build from ``component`` for a simulation.

    The plan's root is ``"chip"``, so ``instances`` keys read ``"chip/<name>"``.  Coupon padding is zero
    because the check is about the drawn layout, not about a simulation's surroundings.
    """
    plan = GeometryPlan(
        component,
        root_id=_ROOT_ID,
        layer_stack=layer_stack,
        material_records=material_records,
        coupon_padding_um=0.0,
    )
    for path, reference in (instances or {}).items():
        plan.register_instance(path, reference)
    return plan.prepare()
