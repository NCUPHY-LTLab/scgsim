"""From a convergence declaration and the config being written to the plan the run folder carries.

``convergence_plan`` checks each criterion against what the config holds (the terminals, the number of modes,
the Surface-EPR rows and margins, the Eigenmode EPR selection) and names, for every number it reads, the
result table, the row and the columns.  This is the one place that decides which columns make up a quantity:
"MA, whole surfaces" is the ``p_surf`` column of every whole-surface ``Dielectric`` entry of kind ``MA``, and
"MA beyond 1 um" that of every margin entry of kind ``MA`` at 1 um, read from the index entries the same
config writes.  The check (``scgsim.contracts.palace_convergence``) reads only the plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scgsim.contracts.palace_convergence import (
    ConvergencePlan,
    PlannedCriterion,
    PlannedValue,
    Quantity,
    TableRow,
)
from scgsim.contracts.palace_run_folder import (
    EIGENMODE,
    palace_problem,
    table_name,
)
from scgsim.refusal import Declared

from .convergence import (
    Capacitance,
    ConvergenceDeclaration,
    Criterion,
    ErrorIndicator,
    ModeFrequency,
    SurfaceEpr,
    element_label,
)
from .index import IndexEntry, SurfaceEprEntry, TerminalEntry

_FREQUENCY_COLUMN = "Re{f} (GHz)"
_ERROR_INDICATOR_COLUMN = "Norm"

_UNITS: dict[type, str] = {
    Capacitance: "F",
    ModeFrequency: "GHz",
    SurfaceEpr: "1",
    ErrorIndicator: "1",
}
_QUANTITY_NAMES: dict[type, Quantity] = {
    Capacitance: "capacitance",
    ModeFrequency: "mode_frequency",
    SurfaceEpr: "surface_epr",
    ErrorIndicator: "error_indicator",
}


@dataclass(frozen=True)
class PlanContext:
    """What the config being written holds that a criterion is checked against.

    ``entries`` are its index entries.  ``num_modes`` is the Eigenmode mode count (None on Electrostatic);
    ``surface_interfaces`` the Eigenmode EPR selection of interface kinds (None: all of them).
    """

    problem: str
    entries: Sequence[IndexEntry]
    num_modes: int | None = None
    surface_interfaces: tuple[str, ...] | None = None

    @property
    def terminals(self) -> dict[str, int]:
        """The Electrostatic terminals by name, each with its row and column number in ``terminal-C.csv``."""
        return {
            entry.name: entry.index
            for entry in self.entries
            if isinstance(entry, TerminalEntry)
        }


def convergence_plan(
    declaration: ConvergenceDeclaration, context: PlanContext
) -> ConvergencePlan:
    """Check every criterion against the config and plan the numbers it reads from each pass.

    Raises:
        ValueError: a criterion names a terminal, mode, interface kind or margin the config does not have.
    """
    return ConvergencePlan(
        problem=context.problem,
        criteria=tuple(_criterion(item, context) for item in declaration.criteria),
        minimum_converged_passes=declaration.minimum_converged_passes,
    )


def _criterion(criterion: Criterion, context: PlanContext) -> PlannedCriterion:
    quantity = criterion.quantity
    if isinstance(quantity, Capacitance):
        values = _capacitance_values(quantity, context)
    elif isinstance(quantity, ModeFrequency):
        values = _frequency_values(quantity, context)
    elif isinstance(quantity, SurfaceEpr):
        values = _surface_values(quantity, context)
    else:
        values = (
            PlannedValue(
                quantity.label,
                table_name("error-indicators"),
                None,
                (_ERROR_INDICATOR_COLUMN,),
            ),
        )
    return PlannedCriterion(
        label=criterion.label,
        quantity=_QUANTITY_NAMES[type(quantity)],
        test=criterion.test,
        threshold=criterion.threshold,
        unit=_UNITS[type(quantity)],
        values=values,
    )


def _capacitance_values(
    quantity: Capacitance, context: PlanContext
) -> tuple[PlannedValue, ...]:
    """Element ``C[a][b]`` is row ``i = index(a)``, column ``C[i][index(b)] (F)`` of ``terminal-C.csv``."""
    terminals = context.terminals
    names = list(terminals)
    pairs = _capacitance_pairs(quantity, names)
    unknown = sorted({name for pair in pairs for name in pair} - set(names))
    if unknown:
        raise ValueError(
            Declared(
                f"{quantity.label} criterion",
                "the calls to set_convergence and add_terminal",
            ).refusal(
                f"names terminals {unknown} the simulation does not have; it has {names}"
            )
        )
    if not pairs:
        raise ValueError(
            Declared(
                f"{quantity.label} criterion",
                "the calls to set_convergence and add_terminal",
            ).refusal(
                f"selects no element of the {len(names)}-terminal matrix {names}: choose another selection"
            )
        )
    return tuple(
        PlannedValue(
            element_label(a, b),
            table_name("terminal-C"),
            TableRow("i", terminals[a]),
            (f"C[i][{terminals[b]}] (F)",),
        )
        for a, b in pairs
    )


def _capacitance_pairs(
    quantity: Capacitance, names: list[str]
) -> list[tuple[str, str]]:
    """The ordered terminal pairs a capacitance selection names, terminals in the order they were added."""
    count = len(names)
    if quantity.selection in ("element", "elements"):
        return list(quantity.pairs)
    if quantity.selection == "diagonal":
        return [(name, name) for name in names]
    if quantity.selection == "sub_diagonal":
        offset = quantity.offset or 1
        return [(names[i], names[i + offset]) for i in range(count - offset)]
    off_diagonal = [
        (names[i], names[j]) for i in range(count) for j in range(i + 1, count)
    ]
    if quantity.selection == "off_diagonal":
        return off_diagonal
    return [(name, name) for name in names] + off_diagonal


def _frequency_values(
    quantity: ModeFrequency, context: PlanContext
) -> tuple[PlannedValue, ...]:
    """The frequency of mode ``m`` is row ``m``, column ``Re{f} (GHz)`` of ``eig.csv``."""
    modes = _checked_modes(quantity.modes, quantity.label, context)
    return tuple(
        PlannedValue(
            f"frequency, mode {mode}",
            table_name("eig"),
            TableRow("m", mode),
            (_FREQUENCY_COLUMN,),
        )
        for mode in modes
    )


def _surface_values(
    quantity: SurfaceEpr, context: PlanContext
) -> tuple[PlannedValue, ...]:
    """One sum of ``p_surf`` columns per mode or excitation row of ``surface-Q.csv``."""
    columns = _surface_columns(quantity, context)
    rows = _surface_rows(quantity, context)
    index_column = palace_problem(context.problem).series_index
    return tuple(
        PlannedValue(
            f"{quantity.surface_label}, {row_label}",
            table_name("surface-Q"),
            TableRow(index_column, row),
            columns,
        )
        for row_label, row in rows
    )


def _surface_columns(quantity: SurfaceEpr, context: PlanContext) -> tuple[str, ...]:
    """The ``p_surf`` columns of the kind's whole-surface entries, or of its margin entries at the margin."""
    where = "the calls to set_convergence and set_surface_epr"
    kind = quantity.interface_kind
    if (
        context.surface_interfaces is not None
        and kind not in context.surface_interfaces
    ):
        raise ValueError(
            Declared(
                f"{quantity.label} criterion",
                "the calls to set_convergence and set_epr_request",
            ).refusal(
                f"reads {kind}, which set_epr_request(surface_interfaces=...) leaves out of the "
                f"run's EPR ({list(context.surface_interfaces)}): add {kind} there or drop the criterion"
            )
        )
    entries = [
        entry
        for entry in context.entries
        if isinstance(entry, SurfaceEprEntry) and entry.rows[0].interface_type == kind
    ]
    if not entries:
        raise ValueError(
            Declared(f"{quantity.label} criterion", where).refusal(
                f"reads {kind} participation, and Surface EPR is not requested for {kind}: call "
                f"set_surface_epr() with a {kind} spec"
            )
        )
    if quantity.margin_um is None:
        chosen = [entry for entry in entries if entry.inset is None]
    else:
        chosen = [
            entry
            for entry in entries
            if entry.inset is not None and entry.inset.margin_um == quantity.margin_um
        ]
        if not chosen:
            margins = sorted(
                {entry.inset.margin_um for entry in entries if entry.inset is not None}
            )
            raise ValueError(
                Declared(f"{quantity.label} criterion", where).refusal(
                    f"reads the {kind} rows at margin {quantity.margin_um:g} um, which the {kind} spec does "
                    + (
                        f"not declare; its margins are {margins} um"
                        if margins
                        else "not declare; it declares no margin"
                    )
                )
            )
    return tuple(f"p_surf[{entry.index}]" for entry in chosen)


def _surface_rows(quantity: SurfaceEpr, context: PlanContext) -> list[tuple[str, int]]:
    """The mode or excitation rows a Surface-EPR quantity selects."""
    if context.problem == EIGENMODE.name:
        return [
            (f"mode {mode}", mode)
            for mode in _checked_modes(quantity.modes, quantity.label, context)
        ]
    terminals = context.terminals
    selected = (
        list(terminals) if quantity.terminals is None else list(quantity.terminals)
    )
    unknown = sorted(set(selected) - set(terminals))
    if unknown:
        raise ValueError(
            Declared(
                f"{quantity.label} criterion",
                "the calls to set_convergence and add_terminal",
            ).refusal(
                f"selects terminals {unknown} the simulation does not have; it has {list(terminals)}"
            )
        )
    return [(f"excitation {name}", terminals[name]) for name in selected]


def _checked_modes(
    modes: tuple[int, ...] | None, label: str, context: PlanContext
) -> tuple[int, ...]:
    """The selected modes, every computed mode for None; a mode beyond ``num_modes`` is refused."""
    count = context.num_modes or 0
    if modes is None:
        return tuple(range(1, count + 1))
    beyond = [mode for mode in modes if mode > count]
    if beyond:
        raise ValueError(
            Declared(
                f"{label} criterion", "the calls to set_convergence and set_eigenmode"
            ).refusal(
                f"selects modes {beyond}, and the run computes {count}: select modes 1 to {count}, or raise "
                "set_eigenmode(num_modes=...)"
            )
        )
    return modes


__all__ = ["PlanContext", "convergence_plan"]
