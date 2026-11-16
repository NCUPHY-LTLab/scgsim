"""Assign the Q3D Nets of an imported SGB model and read each back.

Each Net of the request is assigned over its conductor objects with its Q3D type, in the request's order.
Q3D on the model extracts C/G only, so no Net carries a source or sink; AC/RL is designed again in
NCUPHY-LTLab/scgsim#265.
"""

from __future__ import annotations

from typing import Any

from scgsim.aedt.specs.q3d import Q3dSpec


class Q3dNetError(RuntimeError):
    """A Net did not take as requested, or the native Net order is not the request's."""


def assign_q3d_nets(app: Any, spec: Q3dSpec) -> list[dict[str, Any]]:
    """Assign the request's Nets in their order and read each one's objects back."""
    records: list[dict[str, Any]] = []
    for net in spec.nets:
        boundary = app.assign_net(list(net.object_names), net.name, net.net_type)
        if not boundary or net.name not in app.net_names:
            raise Q3dNetError(f"Q3D net assignment failed for {net.name!r}")
        expected = sorted(int(app.modeler[name].id) for name in net.object_names)
        observed = sorted(
            int(value) for value in app.oboundary.GetExcitationAssignment(net.name)
        )
        if observed != expected:
            raise Q3dNetError(f"Q3D native net assignment differs for {net.name!r}")
        records.append(
            {
                "name": net.name,
                "net_type": net.net_type,
                "object_names": list(net.object_names),
                "native_object_ids": observed,
            }
        )
    if list(app.net_names) != [net.name for net in spec.nets]:
        raise Q3dNetError("Q3D native net order does not match the request")
    return records
