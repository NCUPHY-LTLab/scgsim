"""What a Palace simulation does with the plan's lumped elements, read from the snapshot's records.

The plan declares each element once (``GeometryPlan.set_lumped_elements``); here a Palace problem realizes it.
Eigenmode writes every element as a not excited Palace ``LumpedPort`` ``P<number>`` with its R, L and C, and an
excitation port as its impedance, a resistance: a load, so the Eigenmode Q is the loaded Q.  An authored sheet
that carries no element is not modeled.  Electrostatic realizes none: its capacitance matrix is the
conductors', with every lumped element open, and the sheets stay as authored.  What each problem does with an
element is one line of ``simulation_drc``'s report and one entry of ``input_summary()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scgsim.geometry import LumpedElementRecord, LumpedPort
from scgsim.geometry.source.plan import GeometryPlanSnapshot

from .ports import RequestedPort

# Palace's ``LumpedPort`` circuit key of each component field: R, L and C of one parallel circuit, in SI units.
_PALACE_CIRCUIT_KEYS = {
    "resistance_ohm": "R",
    "inductance_h": "L",
    "capacitance_f": "C",
}
_SI_UNITS = {"R": "ohm", "L": "H", "C": "F"}


def eigenmode_ports(plan: GeometryPlanSnapshot) -> list[RequestedPort]:
    """Every lumped element of ``plan`` as the Palace port it is, in its number's order: element n is ``P<n>``."""
    return [
        RequestedPort(
            name=record.name,
            layer=record.target_layer,
            circuit=_palace_circuit(record),
        )
        for record in plan.lumped_elements
    ]


def eigenmode_report(plan: GeometryPlanSnapshot) -> tuple[str, ...]:
    """One line per element of ``plan`` and per authored sheet that carries none, as Eigenmode realizes them."""
    lines = [
        f"P{record.number} {record.name}: {eigenmode_realization(record)}"
        for record in plan.lumped_elements
    ]
    lines.extend(
        f"{name}: no lumped element on the plan; not modeled"
        for name in _undeclared_sheets(plan)
    )
    return tuple(lines)


def electrostatic_report(plan: GeometryPlanSnapshot) -> tuple[str, ...]:
    """One line per element of ``plan``: Electrostatic realizes none of them."""
    return tuple(
        f"{record.describe()}: {electrostatic_realization(record)}"
        for record in plan.lumped_elements
    )


def eigenmode_realization(record: LumpedElementRecord) -> str:
    """What Eigenmode makes of ``record``: a not excited Palace ``LumpedPort`` with its circuit keys."""
    keys = ", ".join(
        f"{key} {value!r} {_SI_UNITS[key]}"
        for key, value in _palace_circuit(record).items()
    )
    if record.is_port:
        return f"LumpedPort, {keys}: the excitation port as its impedance, a load (the Q is the loaded Q)"
    return f"LumpedPort, {keys}"


def electrostatic_realization(record: LumpedElementRecord) -> str:  # noqa: ARG001 -- every record is left open alike
    """What Electrostatic makes of ``record``: nothing; the conductors' capacitance has the element open."""
    return "not realized: the capacitance matrix is the conductors', with every lumped element open"


def summarized_elements(
    plan: GeometryPlanSnapshot, realization: Callable[[LumpedElementRecord], str]
) -> list[dict[str, Any]]:
    """The records of ``plan`` as ``input_summary()`` shows them, each with what the problem makes of it."""
    return [
        {**record.to_payload(), "realization": realization(record)}
        for record in plan.lumped_elements
    ]


def _palace_circuit(record: LumpedElementRecord) -> dict[str, float]:
    """The Palace ``LumpedPort`` circuit keys of ``record``: its R, L, C, or a port's impedance as R."""
    declaration = record.declaration
    if isinstance(declaration, LumpedPort):
        return {"R": declaration.impedance_ohm}
    return {
        _PALACE_CIRCUIT_KEYS[field]: value
        for field, value in declaration.circuit.items()
    }


def _undeclared_sheets(plan: GeometryPlanSnapshot) -> list[str]:
    """The authored sheets of ``plan`` that carry no lumped element, in the order the layout authors them."""
    declared = {record.name for record in plan.lumped_elements}
    authored = plan.stack["metadata"].get("port_sheet_source_layers", ())
    return [record["name"] for record in authored if record["name"] not in declared]


__all__ = [
    "eigenmode_ports",
    "eigenmode_realization",
    "eigenmode_report",
    "electrostatic_realization",
    "electrostatic_report",
    "summarized_elements",
]
