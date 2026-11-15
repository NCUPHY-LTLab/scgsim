"""The Q3D reading of an SGB model: the AEDT objects of each Net, and the object a terminal Entity names.

Q3D puts a Net on whole objects (issue #67).  In a Route C model the conductors are the guard's conductor
solids.  In a Route A model they are the conductor sheets, which the import rebuilds as objects named by their
sidecar ``SHEET`` group (:func:`sheet_object_names`), and the finite bodies retained as solids.  Every reading
here comes from the sidecar and the guard, so a request is checked against the model before AEDT is touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.aedt.preparation.sgb_model import PreparedSgbModel, aedt_net_names
from scgsim.contracts.sgb_sidecar import Sidecar, read_sidecar
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard
from scgsim.geometry.sgb.output.sidecar.aedt import sanitise_aedt_name


def sheet_groups(sidecar: Sidecar) -> dict[str, Mapping[str, Any]]:
    """The Route A conductor-sheet groups of the model (interface ``SHEET``), by group name."""
    return {
        name: group
        for name, group in sorted(sidecar["group"].items())
        if group["dim"] == 2 and group.get("interface_type") == "SHEET"
    }


def sheet_object_names(sidecar: Sidecar, solid_names: set[str]) -> dict[str, str]:
    """``SHEET`` group -> the AEDT name of the object Q3D rebuilds it as: the group's sanitised name.

    A name two groups share, or one an imported solid already has, is refused rather than suffixed, as
    Nets are (the Human, 2026-11-12, issue #67).
    """
    out: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for group in sheet_groups(sidecar):
        name = sanitise_aedt_name(group)
        if name in by_name or name in solid_names:
            other = by_name.get(name, f"the solid {name!r}")
            raise ValueError(
                f"the sheet group {group!r} and {other!r} both become the AEDT name {name!r}; "
                "rename one of them"
            )
        by_name[name] = group
        out[group] = name
    return out


def q3d_conductors(model: PreparedSgbModel) -> dict[str, dict[str, str]]:
    """Net -> {AEDT object: the Entity it carries} over the model's conductor objects.

    The objects are the conductor solids of the guard and, in Route A, the rebuilt sheet objects.  A
    conductor with no Net is refused: Q3D assigns every conductor to one.
    """
    sidecar = read_sidecar(model.path("sidecar"))
    guard = read_step_guard(model.path("guard"))
    solids = {solid["solid"]: solid["aedt_name"] for solid in guard["solids"]}
    out: dict[str, dict[str, str]] = {}
    for solid_id, name in solids.items():
        volume = sidecar["volume"][solid_id]
        if volume["kind"] != "conductor":
            continue
        if volume["net"] is None:
            raise ValueError(f"conductor solid {name!r} has no Net")
        out.setdefault(volume["net"], {})[name] = volume["entity"]
    sheets = sheet_object_names(sidecar, set(solids.values()))
    for group, name in sheets.items():
        record = sidecar["group"][group]
        if record["net"] is None or len(record["entities"]) != 1:
            raise ValueError(
                f"sheet group {group!r} must carry one Net and one Entity, not "
                f"{record['net']!r} and {record['entities']}"
            )
        out.setdefault(record["net"], {})[name] = record["entities"][0]
    return dict(sorted(out.items()))


def q3d_net_objects(model: PreparedSgbModel) -> dict[str, tuple[str, ...]]:
    """AEDT Net name -> its conductor objects, sorted, in the Net order of the model (sorted Net ids)."""
    conductors = q3d_conductors(model)
    names = aedt_net_names(conductors)
    return {names[net]: tuple(sorted(objects)) for net, objects in conductors.items()}


def terminal_object(model: PreparedSgbModel, net_id: str, entity_id: str) -> str:
    """The one conductor object of Net ``net_id`` that carries Entity ``entity_id``.

    A terminal names an Entity; it must be exactly one object of its own Net.
    """
    objects = q3d_conductors(model).get(net_id, {})
    matches = [name for name, entity in objects.items() if entity == entity_id]
    if len(matches) != 1:
        raise ValueError(
            f"terminal Entity {entity_id!r} is {len(matches)} conductor objects of Net {net_id!r} "
            f"({sorted(matches)}), not one"
        )
    return matches[0]
