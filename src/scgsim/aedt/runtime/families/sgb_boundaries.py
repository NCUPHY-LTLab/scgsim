"""Plan the boundaries the AEDT SGB import assigns, from the model's sidecar and guard, before AEDT is touched.

:func:`plan_hfss_boundaries` names every HFSS boundary and binds each port treatment (a junction or a
generic ``LumpedRlc``) to all the faces of its port sheet: SGB partitions a port sheet where another face
meets it (a pad wall, a band), and each of those faces must import exactly once.  :func:`q3d_net_names`
names the Q3D Nets.  Net, junction and port-sheet IDs such as ``chip/A`` and ``chip_A`` sanitise to the
same AEDT name, so every name comes from :func:`~scgsim.geometry.sgb.sidecar.aedt.aedt_names`, which
appends ``__2``, ``__3`` ... to a later collision; the assignment and its readback then use that one name.
A model the plan refuses raises :class:`BoundaryPlanError` before the import changes the design.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.aedt.preparation.sgb_model import SgbJunction, port_group
from scgsim.aedt.rlc import rlc_native_topology
from scgsim.aedt.specs.common import LumpedRlc
from scgsim.geometry.sgb.exchange.guard import StepGuard
from scgsim.geometry.sgb.sidecar.aedt import aedt_names

Point = tuple[float, float, float]


class BoundaryPlanError(ValueError):
    """The sidecar and guard give a boundary the import cannot realize; nothing in AEDT has changed."""


@dataclass(frozen=True)
class PortRlc:
    """One Lumped RLC the import assigns: its boundary name, its sidecar port and unit direction, its branches.

    ``evidence`` is what the treatment adds to the assignment record (the junction or the requested
    ``LumpedRlc``).
    """

    name: str
    port: Mapping[str, Any]
    direction: Point
    branches: Mapping[str, Any]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class HfssBoundaryPlan:
    """Every HFSS boundary the import assigns, named and checked against the sidecar and guard before AEDT.

    ``perfect_e`` maps each Net with Route A sheet faces to its boundary name and its planned faces.  Names
    are unique among their kind (:func:`~scgsim.geometry.sgb.sidecar.aedt.aedt_names`), so two Nets or two
    treatments whose IDs sanitise alike never claim one boundary.
    """

    perfect_e: Mapping[str, tuple[str, tuple[str, ...]]]
    junctions: tuple[PortRlc, ...]
    lumped_rlcs: tuple[PortRlc, ...]


def plan_hfss_boundaries(
    sidecar: Mapping[str, Any],
    guard: StepGuard,
    junctions: Sequence[SgbJunction],
    lumped_rlcs: Sequence[LumpedRlc],
) -> HfssBoundaryPlan:
    """The HFSS boundaries of the model: Perfect E per Net, and a Lumped RLC per port treatment.

    Each port treatment takes every face of its port sheet; each of those faces must import exactly once
    (on one solid, or as a rebuilt free face), and a junction must run along its port sheet.
    """
    copies = _planned_copies(guard)
    return HfssBoundaryPlan(
        perfect_e=_perfect_e_plan(sidecar),
        junctions=_junction_rlcs(sidecar, copies, junctions),
        lumped_rlcs=_generic_rlcs(sidecar, copies, lumped_rlcs),
    )


def _planned_copies(guard: StepGuard) -> Counter[str]:
    """Planned face id -> how many copies AEDT will hold: one per solid listing it, one for a free face."""
    copies = Counter(
        face["face"] for solid in guard["solids"] for face in solid["faces"]
    )
    copies.update(face["face"] for face in guard["free_faces"])
    return copies


def _perfect_e_plan(
    sidecar: Mapping[str, Any],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    groups = sidecar["physical_groups"]
    sheets = {
        net: tuple(
            planned
            for group in net_groups
            if groups[group]["interface_type"] == "SHEET"
            for planned in groups[group]["ids"]
        )
        for net, net_groups in sorted(sidecar["unions"]["nets"].items())
    }
    sheets = {net: faces for net, faces in sheets.items() if faces}
    names = aedt_names([(net, f"perfect_e_{net}") for net in sheets])
    return {net: (names[net], faces) for net, faces in sheets.items()}


def _junction_rlcs(
    sidecar: Mapping[str, Any], copies: Counter[str], junctions: Sequence[SgbJunction]
) -> tuple[PortRlc, ...]:
    """Each junction's parallel Lumped RLC (its inductance and capacitance) on its port sheet."""
    names = aedt_names(
        [
            (item.junction.junction_id, f"junction_rlc_{item.junction.junction_id}")
            for item in junctions
        ]
    )
    planned = []
    for item in junctions:
        junction = item.junction
        port = _checked_port(sidecar, item.port_group, copies)
        direction = _unit(port["direction"])
        if math.dist(direction, _unit((*junction.direction_xy, 0.0))) > 1e-9:
            raise BoundaryPlanError(
                f"junction {junction.junction_id!r} runs along {junction.direction_xy}, its port sheet "
                f"along {port['direction']}"
            )
        branches: dict[str, Any] = {
            "RLC Type": "Parallel",
            "UseResist": False,
            "UseInduct": True,
            "Inductance": f"{junction.inductance_h}H",
            "UseCap": bool(junction.capacitance_f),
        }
        if junction.capacitance_f:
            branches["Capacitance"] = f"{junction.capacitance_f}F"
        evidence = {
            "junction_id": junction.junction_id,
            "inductance_h": junction.inductance_h,
            "capacitance_f": junction.capacitance_f,
        }
        planned.append(
            PortRlc(names[junction.junction_id], port, direction, branches, evidence)
        )
    return tuple(planned)


def _generic_rlcs(
    sidecar: Mapping[str, Any], copies: Counter[str], lumped_rlcs: Sequence[LumpedRlc]
) -> tuple[PortRlc, ...]:
    """Each ``LumpedRlc`` on the port sheet it names: exactly its enabled branches, in its topology."""
    names = aedt_names(
        [(item.port_sheet_id, f"rlc_{item.port_sheet_id}") for item in lumped_rlcs]
    )
    planned = []
    for item in lumped_rlcs:
        port = _checked_port(sidecar, port_group(item.port_sheet_id), copies)
        evidence = {"port_sheet_id": item.port_sheet_id, "requested": item.to_payload()}
        planned.append(
            PortRlc(
                names[item.port_sheet_id],
                port,
                _unit(port["direction"]),
                rlc_branches(item),
                evidence,
            )
        )
    return tuple(planned)


def rlc_branches(treatment: LumpedRlc) -> dict[str, Any]:
    """The native Lumped RLC properties of ``treatment``: its topology and exactly its enabled branches.

    ``None`` alone disables a branch, so an authored zero stays an enabled branch of value zero.
    """
    branches: dict[str, Any] = {"RLC Type": rlc_native_topology(treatment.topology)}
    for enabled, key, value, unit in (
        ("UseResist", "Resistance", treatment.resistance_ohm, "ohm"),
        ("UseInduct", "Inductance", treatment.inductance_h, "H"),
        ("UseCap", "Capacitance", treatment.capacitance_f, "F"),
    ):
        branches[enabled] = value is not None
        if value is not None:
            branches[key] = f"{value}{unit}"
    return branches


def _checked_port(
    sidecar: Mapping[str, Any], group: str, copies: Counter[str]
) -> Mapping[str, Any]:
    """The one sidecar port of ``group``; each of its faces must import exactly once."""
    ports = [p for p in sidecar["ports"] if p["group"] == group]
    if len(ports) != 1:
        raise BoundaryPlanError(
            f"the sidecar has {len(ports)} lumped ports {group!r}, not one"
        )
    port = ports[0]
    if not port["faces"]:
        raise BoundaryPlanError(f"port sheet {group!r} has no face in the sidecar")
    for planned in port["faces"]:
        if copies[planned] != 1:
            raise BoundaryPlanError(
                f"port face {planned} of {group!r} imports as {copies[planned]} copies, not one"
            )
    return port


def q3d_net_names(nets: Iterable[str]) -> dict[str, str]:
    """Net -> its AEDT Net name, unique across ``nets`` (``chip/A`` and ``chip_A`` become ``chip_A``, ``chip_A__2``)."""
    return aedt_names([(net, net) for net in sorted(nets)])


def _unit(values: Sequence[float]) -> Point:
    x, y, z = (float(value) for value in values)
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0:
        raise ValueError("a direction must not be zero")
    return (x / length, y / length, z / length)
