"""What HFSS does with the plan's lumped elements: the AEDT records each one becomes, read from the snapshot.

The plan declares each element once (``GeometryPlan.set_lumped_elements``); the snapshot carries its records.
HFSS models only the sheets that carry an element, each cut to its central rectangle with the conductors' end
pieces as Palace cuts its ports (the partition id is the sheet's ``port_sheet_id``, terminal A the conductor at
the start of the authored direction), so both solvers model the same sheet.  Then, by calculation:

- **HFSS Eigenmode** (:func:`hfss_eigenmode_inputs`): an element with an inductance and no resistance is an EPR
  junction (a ``PlanarJunction`` named by its sheet, a Lumped RLC of its L and C); any other element is a
  parallel ``LumpedRlc`` of its R, L and C; an excitation port is a parallel ``LumpedRlc`` of its impedance, a
  load, so the Q is the loaded Q.
- **HFSS Driven Terminal** (:func:`hfss_driven_inputs`): each excitation port is a lumped Terminal port, its
  number the port index, its Terminal on the signal conductor; an element is refused, by name, since the
  import assigns Lumped RLCs outside a Driven design only and none was checked in AEDT.

Q3D reads no lumped element: ``prepare_q3d_from_geometry`` builds the plan's sheets as authored.  Each result
says, for every element and every authored sheet it leaves out, what HFSS makes of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from scgsim.aedt.preparation.planar_junctions import planar_junction_from_port
from scgsim.aedt.specs.common import LumpedRlc, LumpedTerminalPort
from scgsim.aedt.specs.epr import PlanarJunction
from scgsim.geometry import (
    GeometryBuildInput,
    GeometryPlanSnapshot,
    LumpedElementRecord,
    LumpedPort,
    RectangularJunctionSpec,
    with_rectangular_junctions,
)
from scgsim.geometry.models.junctions import junction_occurrence_path
from scgsim.geometry.sgb.output.sidecar.aedt import sanitise_aedt_name
from scgsim.refusal import Declared

_SET_LUMPED_ELEMENTS = "the call to GeometryPlan.set_lumped_elements"
_NOT_MODELED = "no lumped element on the plan; not modeled"


@dataclass(frozen=True)
class HfssLumpedInputs:
    """The AEDT model inputs of the plan's lumped elements for one HFSS calculation.

    ``build_input`` is the snapshot's input with only the sheets that carry an element, each bound for its
    partition; ``junctions``, ``lumped_rlcs`` and ``lumped_ports`` are what ``prepare_sgb_model`` puts on them.
    ``realizations`` says what HFSS makes of each element, by element name in number order, and ``left_out``
    names the authored sheets that carry none.
    """

    build_input: GeometryBuildInput
    junctions: tuple[PlanarJunction, ...] = ()
    lumped_rlcs: tuple[LumpedRlc, ...] = ()
    lumped_ports: tuple[LumpedTerminalPort, ...] = ()
    realizations: Mapping[str, str] = field(default_factory=dict)
    left_out: tuple[str, ...] = ()

    @property
    def report(self) -> tuple[str, ...]:
        """One line per element and per authored sheet left out, as ``simulation_drc``'s report prints them."""
        return (
            *(f"{name}: {text}" for name, text in self.realizations.items()),
            *(f"{name}: {_NOT_MODELED}" for name in self.left_out),
        )


def hfss_eigenmode_inputs(snapshot: GeometryPlanSnapshot) -> HfssLumpedInputs:
    """The plan's lumped elements as HFSS Eigenmode realizes them: EPR junctions and parallel Lumped RLCs."""
    build_input = modeled_input(snapshot)
    junctions: list[PlanarJunction] = []
    rlcs: list[LumpedRlc] = []
    realizations: dict[str, str] = {}
    for record in snapshot.lumped_elements:
        declaration = record.declaration
        if isinstance(declaration, LumpedPort):
            rlcs.append(
                LumpedRlc(
                    record.port_sheet_id,
                    "parallel",
                    resistance_ohm=declaration.impedance_ohm,
                )
            )
            realizations[record.name] = (
                f"Lumped RLC, parallel R {declaration.impedance_ohm!r} ohm: the excitation port as its "
                "impedance, a load (the Q is the loaded Q)"
            )
        elif (
            declaration.inductance_h is not None and declaration.resistance_ohm is None
        ):
            # An inductance with no resistance: what HFSS EPR reports as a junction.
            junctions.append(
                _junction(
                    build_input,
                    record,
                    declaration.inductance_h,
                    declaration.capacitance_f,
                )
            )
            capacitance_f = declaration.capacitance_f or 0.0
            realizations[record.name] = (
                f"EPR junction {record.name!r}, Lumped RLC L {declaration.inductance_h!r} H, "
                f"C {capacitance_f!r} F"
            )
        else:
            rlcs.append(
                LumpedRlc(record.port_sheet_id, "parallel", **declaration.circuit)
            )
            realizations[record.name] = (
                f"Lumped RLC, parallel {declaration.values_text()}; HFSS EPR reports a junction "
                "only for an element without a resistance"
            )
    return HfssLumpedInputs(
        build_input=build_input,
        junctions=tuple(junctions),
        lumped_rlcs=tuple(rlcs),
        realizations=realizations,
        left_out=_left_out(snapshot),
    )


def hfss_driven_inputs(snapshot: GeometryPlanSnapshot) -> HfssLumpedInputs:
    """The plan's excitation ports as HFSS Driven Terminal's lumped Terminal ports; an element is refused.

    Raises:
        ValueError: the plan declares a lumped element that is not a port, does not declare exactly two ports,
            or two port names become one AEDT name.
    """
    records = snapshot.lumped_elements
    for record in records:
        if not record.is_port:
            raise ValueError(
                Declared(
                    f"{record.describe()}, its sheet {record.place()}",
                    _SET_LUMPED_ELEMENTS,
                ).refusal(
                    "HFSS Driven Terminal does not realize a lumped element yet: its import assigns Lumped "
                    "RLCs outside a Driven design only, and none was checked in AEDT; leave it out of the "
                    "plan of this calculation"
                )
            )
    if len(records) != 2:
        raise ValueError(
            Declared("the plan's excitation ports", _SET_LUMPED_ELEMENTS).refusal(
                f"HFSS Driven Terminal V1 takes two lumped ports; the plan declares {len(records)}: "
                f"{[record.name for record in records]}"
            )
        )
    names = {record.name: sanitise_aedt_name(record.name) for record in records}
    if len(set(names.values())) != len(names):
        raise ValueError(
            Declared("the plan's excitation ports", _SET_LUMPED_ELEMENTS).refusal(
                f"{list(names)} both become the AEDT port name {next(iter(names.values()))!r}; "
                "rename one sheet"
            )
        )
    return HfssLumpedInputs(
        build_input=modeled_input(snapshot),
        lumped_ports=tuple(
            _terminal_port(record, names[record.name]) for record in records
        ),
        realizations={
            record.name: (
                f"lumped Terminal port {record.number} {names[record.name]!r}, Terminal on "
                f"{record.signal.source_entity_id}, reference {record.reference.source_entity_id}"
            )
            for record in records
        },
        left_out=_left_out(snapshot),
    )


def modeled_input(snapshot: GeometryPlanSnapshot) -> GeometryBuildInput:
    """The snapshot's input with only the sheets that carry a lumped element, each bound for its partition."""
    records = {record.port_sheet_id: record for record in snapshot.lumped_elements}
    build_input = snapshot.geometry_input
    sheets = tuple(
        region
        for region in build_input.port_sheet_regions
        if region.port_sheet_id in records
    )
    kept = replace(build_input, port_sheet_regions=sheets)
    if not sheets:
        return kept
    return with_rectangular_junctions(
        kept,
        tuple(
            RectangularJunctionSpec(
                junction_id=region.port_sheet_id,
                port_sheet_id=region.port_sheet_id,
                source_polygon_id=region.source_polygon_id,
                source_occurrence_path=junction_occurrence_path(region.metadata),
                terminal_a=records[region.port_sheet_id].start,
                terminal_b=records[region.port_sheet_id].end,
            )
            for region in sheets
        ),
    )


def _junction(
    build_input: GeometryBuildInput,
    record: LumpedElementRecord,
    inductance_h: float,
    capacitance_f: float | None,
) -> PlanarJunction:
    """The EPR junction of ``record``, named by its sheet, its terminals the start and end conductors.

    An element without a capacitance is a pure inductor, HFSS's Lumped RLC with C = 0.
    """
    return planar_junction_from_port(
        build_input,
        port_name=record.name,
        junction_id=record.name,
        terminal_a_net=record.start.net_id,
        terminal_b_net=record.end.net_id,
        inductance_h=inductance_h,
        capacitance_f=0.0 if capacitance_f is None else capacitance_f,
    )


def _terminal_port(record: LumpedElementRecord, name: str) -> LumpedTerminalPort:
    """The lumped Terminal port of an excitation port: its number, its signal and reference conductors."""
    declaration = record.declaration
    if not isinstance(declaration, LumpedPort):
        raise TypeError(f"{record.describe()} is not an excitation port")
    return LumpedTerminalPort(
        index=record.number,
        name=name,
        port_sheet_id=record.port_sheet_id,
        signal_entity_ids=(record.signal.source_entity_id,),
        reference_entity_ids=(record.reference.source_entity_id,),
        impedance_ohm=declaration.impedance_ohm,
    )


def _left_out(snapshot: GeometryPlanSnapshot) -> tuple[str, ...]:
    """The authored sheets that carry no lumped element, in the order the layout authors them."""
    declared = {record.name for record in snapshot.lumped_elements}
    authored = snapshot.stack["metadata"].get("port_sheet_source_layers", ())
    return tuple(item["name"] for item in authored if item["name"] not in declared)


__all__ = [
    "HfssLumpedInputs",
    "hfss_driven_inputs",
    "hfss_eigenmode_inputs",
    "modeled_input",
]
