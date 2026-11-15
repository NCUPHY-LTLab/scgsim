"""Turn a layout Component into the ``GeometryPlan`` snapshot that ``simulation_drc`` checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.source.plan import GeometryPlan, GeometryPlanSnapshot

_ROOT_ID = "chip"

# The check wraps a layout that has no vacuum of its own in this (x, y, z) padding, as ``set_vacuum_region``
# takes it (Issue #182).  None in the plane, so the vacuum ends at the
# layout's own planar extent, and 1000 um below and above, so the metal has air above it to band.
DRC_VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)


def prepare_layout_snapshot(
    component: Any,
    *,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    instances: Mapping[str, Any] | None,
    nets: Mapping[str, Sequence[str]] | None,
) -> GeometryPlanSnapshot:
    """The prepared snapshot a user would build from ``component`` for a simulation.

    The plan's root is ``"chip"``, so ``instances`` keys and ``nets`` members read ``"chip/<name>"``.
    ``nets`` is given to ``GeometryPlan.set_nets`` after the instances are registered, as a simulation's
    plan is built, so the check builds the model that simulation would build.  Coupon padding is zero
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
    if nets is not None:
        plan.set_nets(nets)
    return plan.prepare()


def snapshot_owns_vacuum(snapshot: GeometryPlanSnapshot) -> bool:
    """Whether the plan carries its own outer vacuum envelope, the fact ``set_vacuum_region`` refuses on.

    The envelope is the solution region named ``VACUUM_REGION``.  A vacuum-material region under another
    name, such as the flip-chip gap of a PDK, fills a cavity and is not the envelope.
    """
    return "VACUUM_REGION" in snapshot.stack["solution_regions"]
