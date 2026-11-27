"""What a Palace simulation does with the plan's lumped elements, read from the snapshot's records.

The plan declares each element once (``GeometryPlan.set_lumped_elements``); here a Palace problem realizes it.
Eigenmode writes every element as a not excited Palace ``LumpedPort`` ``P<number>`` with its R, L and C, and an
excitation port as its impedance, a resistance: a load, so the Eigenmode Q is the loaded Q.  Driven writes the
same rows, and excites each port by itself, numbered as the port is, so port *i* is row and column *i* of the
S-matrix; a plan with no port has nothing to drive and is refused.  An authored sheet that carries no element
is not modeled.  Electrostatic realizes none: its capacitance matrix is the conductors', with every lumped
element open, and the sheets stay as authored.  What each problem does with an element is one line of
``simulation_drc``'s report and one entry of ``input_summary()``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scgsim.geometry import LumpedElementRecord, LumpedPort
from scgsim.geometry.source.plan import GeometryPlanSnapshot
from scgsim.refusal import Declared

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


def driven_ports(plan: GeometryPlanSnapshot) -> list[RequestedPort]:
    """Every lumped element of ``plan`` as the Palace port Driven makes it, in its number's order: element n is ``P<n>``.

    Each port is excited by itself, its ``Direction`` from its signal conductor to its reference; every other
    element is written as Eigenmode writes it, not excited.

    Raises:
        ValueError: ``plan`` declares no ``LumpedPort``.
    """
    if not any(record.is_port for record in plan.lumped_elements):
        raise ValueError(
            Declared(
                "the plan of a DrivenSim",
                "the call to GeometryPlan.set_lumped_elements",
            ).refusal(
                "declares no LumpedPort, so a Driven run has nothing to excite: declare each port as "
                "LumpedPort(sheet, reference_net=..., impedance_ohm=...), in the S-matrix's order"
            )
        )
    return [
        RequestedPort(
            name=record.name,
            layer=record.target_layer,
            circuit=_palace_circuit(record),
            excited=record.is_port,
            # Palace's sign convention: a lumped port's Direction points from its signal conductor to its
            # reference, as a wave port's voltage path does.  The authored direction points from ``start`` to
            # ``end``, so a sheet drawn from the reference to the signal is reversed.  The sign of a port is
            # the sign of its S_ij (i != j).
            reverses_direction=record.is_port and record.signal == record.end,
        )
        for record in plan.lumped_elements
    ]


def eigenmode_report(plan: GeometryPlanSnapshot) -> tuple[str, ...]:
    """One line per element of ``plan`` and per authored sheet that carries none, as Eigenmode realizes them."""
    return _report(plan, eigenmode_realization)


def driven_report(plan: GeometryPlanSnapshot) -> tuple[str, ...]:
    """One line per element of ``plan`` and per authored sheet that carries none, as Driven realizes them."""
    return _report(plan, driven_realization)


def electrostatic_report(plan: GeometryPlanSnapshot) -> tuple[str, ...]:
    """One line per element of ``plan``: Electrostatic realizes none of them."""
    return tuple(
        f"{record.describe()}: {electrostatic_realization(record)}"
        for record in plan.lumped_elements
    )


def eigenmode_realization(record: LumpedElementRecord) -> str:
    """What Eigenmode makes of ``record``: a not excited Palace ``LumpedPort`` with its circuit keys."""
    keys = _circuit_text(record)
    if record.is_port:
        return f"LumpedPort, {keys}: the excitation port as its impedance, a load (the Q is the loaded Q)"
    return f"LumpedPort, {keys}"


def driven_realization(record: LumpedElementRecord) -> str:
    """What Driven makes of ``record``: a port excited by itself, or a not excited ``LumpedPort`` as in Eigenmode."""
    keys = _circuit_text(record)
    if record.is_port:
        return (
            f"LumpedPort, {keys}, excitation {record.number}: port {record.number} of the S-matrix, "
            f"driven from {record.signal.source_entity_id} to {record.reference.source_entity_id}"
        )
    return f"LumpedPort, {keys}, not excited"


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


def _report(
    plan: GeometryPlanSnapshot, realization: Callable[[LumpedElementRecord], str]
) -> tuple[str, ...]:
    """One line per element of ``plan`` as ``realization`` makes it, then one per authored sheet that carries none."""
    lines = [
        f"P{record.number} {record.name}: {realization(record)}"
        for record in plan.lumped_elements
    ]
    lines.extend(
        f"{name}: no lumped element on the plan; not modeled"
        for name in _undeclared_sheets(plan)
    )
    return tuple(lines)


def _circuit_text(record: LumpedElementRecord) -> str:
    """The Palace circuit keys of ``record`` as a report prints them (``R 50.0 ohm``)."""
    return ", ".join(
        f"{key} {value!r} {_SI_UNITS[key]}"
        for key, value in _palace_circuit(record).items()
    )


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
    "driven_ports",
    "driven_realization",
    "driven_report",
    "eigenmode_ports",
    "eigenmode_realization",
    "eigenmode_report",
    "electrostatic_realization",
    "electrostatic_report",
    "summarized_elements",
]
