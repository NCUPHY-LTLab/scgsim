"""Serialize the existing conformity records and preserve their report schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.geometry.models.construction import ConstructionPlanRecord

_COORD_TOL_UM = 1e-9


@dataclass(frozen=True)
class ConformalityObservations:
    """Ordered native or planned failures and observations for one existing check."""

    failures: list[dict[str, Any]]
    records: list[dict[str, Any]]


def conformality_report(
    plan: ConstructionPlanRecord,
    engine_gate: str,
    stage: str,
    observations: ConformalityObservations,
    *,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    """Serialize the existing conformity evidence without adding acceptance policy."""
    return {
        "schema": "sgb.engine_gate.v1",
        "engine_gate": engine_gate,
        "route": plan.route,
        "status": "fail" if observations.failures else "pass",
        "stage": stage,
        "checked_record_ids": {
            "interfaces": [record.interface_id for record in plan.interfaces],
            "surface_partitions": [
                record.partition_id for record in plan.surface_partitions
            ],
            "points": [record.point_id for record in plan.points],
            "curves": [record.curve_id for record in plan.curves],
            "surface_loops": [record.loop_id for record in plan.surface_loops],
            "surfaces": [record.surface_id for record in plan.surfaces],
            "volumes": [record.volume_id for record in plan.volumes],
            "backend_entity_tags": [
                record.source_record_id for record in plan.backend_entity_tags
            ],
        },
        "counts": dict(counts),
        "tolerances": {
            "coordinate_um": _COORD_TOL_UM,
        },
        "failures": observations.failures,
        "records": observations.records,
    }


def failure_record(
    code: str,
    record_ids: Sequence[str],
    message: str,
) -> dict[str, Any]:
    """Record one existing conformity failure in its encountered order."""
    return {
        "code": code,
        "record_ids": list(record_ids),
        "message": message,
    }
