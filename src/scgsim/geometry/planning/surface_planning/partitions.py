"""Read explicit parent-interface partition intents in authored order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from scgsim.geometry.models.common import (
    RouteLiteral,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
)
from scgsim.geometry.planning.surface_planning.construction import (
    SurfacePartitionRecord,
)


def plan_surface_partitions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral | None,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[SurfacePartitionRecord, ...]:
    """Plan parent-interface partitions before live surfaces are created.

    This function only consumes explicit `build_input.metadata["surface_partitions"]`
    records.

    Child regions are partition intent, not backend geometry. Each returned
    `SurfacePartitionRecord` must point to a child `SurfacePlanRecord` that
    will be created directly by the backend. The parent interface itself is a
    semantic aggregate and must not be cut by OCC after creation.
    """
    interface_ids = {interface.interface_id for interface in interfaces}
    raw_partitions = build_input.metadata.get("surface_partitions", ())
    if raw_partitions in (None, ()):
        return ()
    if not isinstance(raw_partitions, tuple | list):
        raise TypeError("surface_partitions metadata must be a sequence")

    records: list[SurfacePartitionRecord] = []
    for index, intent in enumerate(raw_partitions):
        if not isinstance(intent, Mapping):
            raise TypeError("surface_partitions entries must be mappings")
        if route is None:
            if intent.get("valid_routes"):
                continue
            valid_routes: tuple[RouteLiteral, ...] = ()
        else:
            valid_routes = cast(
                tuple[RouteLiteral, ...],
                tuple(
                    str(value)
                    for value in cast(
                        Sequence[object], intent.get("valid_routes", (route,))
                    )
                ),
            )
            if route not in valid_routes:
                continue
        parent_interface_id = str(intent.get("parent_interface_id", ""))
        if parent_interface_id not in interface_ids:
            raise ValueError(
                f"surface partition references unknown interface "
                f"{parent_interface_id!r}"
            )
        label = str(intent.get("label", "")).strip()
        if not label:
            raise ValueError("surface partition label must be non-empty")
        child_surface_id = str(
            intent.get("child_surface_id") or f"SURF__{parent_interface_id}__{label}"
        )
        records.append(
            SurfacePartitionRecord(
                partition_id=str(
                    intent.get("partition_id")
                    or f"PART__{parent_interface_id}__{index:04d}"
                ),
                parent_interface_id=parent_interface_id,
                child_surface_id=child_surface_id,
                label=label,
                valid_routes=valid_routes,
                metadata=dict(intent),
            )
        )
    return tuple(records)
