"""Check what a GeometryPlan's caller binds: the final Nets, the Entities' overlap priorities and the Surface-EPR specs."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.source.surface_epr_specs import normalize_surface_epr_specs
from scgsim.refusal import Declared


def plan_call(method: str, name: str) -> Declared:
    """An argument of a GeometryPlan method, which its caller changes."""
    call = "GeometryPlan(...)" if method == "__init__" else f"GeometryPlan.{method}"
    return Declared(name, f"the call to {call}")


def check_bindings_still_hold(
    nets: Mapping[str, Sequence[str]],
    priorities: Mapping[str, int],
    available: set[str],
) -> None:
    """Refuse a Net or priority whose Entity the plan no longer holds: its caller binds it again."""
    for net, members in nets.items():
        if not members or any(member not in available for member in members):
            raise ValueError(
                plan_call("set_nets", f"Net {net!r}").refusal(
                    "it changed after set_nets; call set_nets again"
                )
            )
    for entity_id in priorities:
        if entity_id not in available:
            raise ValueError(
                plan_call(
                    "set_entity_priorities", f"priority of Entity {entity_id!r}"
                ).refusal(
                    "it changed after set_entity_priorities; call set_entity_priorities again"
                )
            )


def validated_nets(
    nets: Mapping[str, Sequence[str]], available: set[str]
) -> dict[str, tuple[str, ...]]:
    """Each Net's members once, in order; every member an Entity of ``available``, in one Net only."""
    owners: dict[str, str] = {}
    prepared: dict[str, tuple[str, ...]] = {}
    for net, members in nets.items():
        if not isinstance(net, str) or not net:
            raise ValueError(
                plan_call("set_nets", "nets").expected("Net ID", "nonempty text", net)
            )
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            raise TypeError(
                plan_call("set_nets", f"Net {net!r}").expected(
                    "members", "a sequence of Entity IDs", members
                )
            )
        unique = tuple(dict.fromkeys(members))
        if not unique:
            raise ValueError(
                plan_call("set_nets", f"Net {net!r}").refusal(
                    "it needs at least one Entity ID"
                )
            )
        for entity_id in unique:
            if not isinstance(entity_id, str) or entity_id not in available:
                raise ValueError(
                    plan_call("set_nets", f"Net {net!r}").refusal(
                        f"it has unknown Entity ID {entity_id!r}"
                    )
                )
            prior = owners.setdefault(entity_id, net)
            if prior != net:
                raise ValueError(
                    plan_call("set_nets", f"Entity {entity_id!r}").refusal(
                        "it belongs to two Nets"
                    )
                )
        prepared[net] = unique
    return prepared


def validated_priorities(
    priorities: Mapping[str, int], available: set[str]
) -> dict[str, int]:
    """Each priority of an Entity of ``available``, which must be an integer."""
    prepared: dict[str, int] = {}
    for entity_id, value in priorities.items():
        if not isinstance(entity_id, str) or entity_id not in available:
            raise ValueError(
                plan_call("set_entity_priorities", "priorities").refusal(
                    f"priority has unknown Entity ID {entity_id!r}"
                )
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                plan_call(
                    "set_entity_priorities", f"priority of Entity {entity_id!r}"
                ).expected("priority", "an integer", value)
            )
        prepared[entity_id] = value
    return prepared


def checked_surface_epr_specs(
    specs: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """``specs`` normalized against ``materials`` by the shared Surface-EPR rules, Palace's and AEDT's.

    "No Surface EPR" has one spelling, no call or ``set_surface_epr(None)``, so an empty mapping is refused.
    """
    if isinstance(specs, Mapping) and not specs:
        raise ValueError(
            plan_call(
                "set_surface_epr", "set_surface_epr({}) requests no Surface EPR"
            ).refusal(
                "to have none, do not call plan.set_surface_epr(), or call "
                "plan.set_surface_epr(None) to turn it off again"
            )
        )
    return normalize_surface_epr_specs(specs, materials=materials)


def authored_surface_epr_specs(
    specs: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """``specs`` as the caller wrote them, checked against ``materials`` and detached from the caller.

    The plan keeps what was written, a ``material_id`` included, because ``prepare()`` normalizes it against
    the stack it prepares.  Every mapping becomes an ordinary dictionary and every sequence a list, so a
    read-only mapping (``MappingProxyType``) is held like any other.
    """
    checked_surface_epr_specs(specs, materials)
    return _detached(specs)


def _detached(value: Any) -> Any:
    """A copy of ``value`` that shares nothing with it, with plain dictionaries and lists as its containers."""
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detached(item) for item in value]
    return copy.deepcopy(value)
