"""The convergence a user declares with ``set_convergence``: quantities, the criteria on them, and their checks.

A criterion is a quantity and a threshold.  The quantities are the Maxwell capacitance matrix
(``Capacitance``, Electrostatic), the frequency of the modes (``ModeFrequency``, Eigenmode), the Surface-EPR
participation of one interface kind (``SurfaceEpr``, both problems; ``MA`` metal-air, ``MS``
metal-substrate, ``SA`` substrate-air) and Palace's error indicator (``ErrorIndicator``, both problems)::

    sim.set_convergence(
        Capacitance.diagonal().within(relative=1e-3),
        SurfaceEpr.each("MA", "MS", "SA").within(relative=2e-2),
        ErrorIndicator().below(1e-2),
        minimum_converged_passes=2,
    )

``within`` bounds the change from the previous pass, ``below`` bounds the value; no threshold has a default.
A quantity with several numbers holds when every one does, and the run has converged when every criterion
holds in the same pass for ``minimum_converged_passes`` passes in a row (``scgsim.contracts.palace_convergence``
decides that).  ``ConvergenceDeclaration.declared`` checks what ``set_convergence`` can know: each quantity fits
the problem, and the thresholds and the pass count are valid.  ``check_numerical`` checks the declaration
against the numerical controls when ``write_config()`` runs; ``convergence_plan`` checks it against the config
and turns it into the plan the run folder carries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

from scgsim.contracts.palace_convergence import Test
from scgsim.contracts.palace_run_folder import EIGENMODE, ELECTROSTATIC
from scgsim.refusal import Declared, found_text

from .index import SURFACE_INTERFACES
from .numerical import NumericalControls

_WHERE_SET_CONVERGENCE = "the call to set_convergence"
_WHERE_NUMERICAL_OR_CONVERGENCE = "the calls to set_numerical and set_convergence"

Problem = Literal["Eigenmode", "Electrostatic"]


# ---- Quantities --------------------------------------------------------------------------------------------

CapacitanceSelection = Literal[
    "element", "diagonal", "sub_diagonal", "off_diagonal", "matrix", "elements"
]


@dataclass(frozen=True)
class Capacitance:
    """Elements of the Maxwell capacitance matrix of an Electrostatic run, named by terminal, in farads.

    Build one with a class method: ``element(a, b)``, ``diagonal()``, ``sub_diagonal(offset=1)`` (the
    elements of terminals ``offset`` apart in the order the terminals were added; a terminal added again
    under its name moves to the end), ``off_diagonal()`` (each pair of different terminals once),
    ``matrix()`` (the diagonal and each pair once) or ``elements([(a, b), ...])``.  Element ``C[a][b]`` is
    the charge on terminal ``a`` with terminal ``b`` at 1 V, row ``a`` and column ``b`` of
    ``terminal-C.csv``.  Terminal names are checked when ``write_config()`` runs.
    """

    selection: CapacitanceSelection
    pairs: tuple[tuple[str, str], ...] = ()
    offset: int | None = None

    @classmethod
    def element(cls, a: str, b: str) -> Capacitance:
        """The one element ``C[a][b]``; ``a == b`` is a diagonal element."""
        return cls("element", (_terminal_pair((a, b), "Capacitance.element"),))

    @classmethod
    def diagonal(cls) -> Capacitance:
        """Every diagonal element ``C[t][t]``."""
        return cls("diagonal")

    @classmethod
    def sub_diagonal(cls, offset: int = 1) -> Capacitance:
        """Every element ``C[t_i][t_(i+offset)]``, terminals counted in the order they were added.

        Raises:
            ValueError: ``offset`` is not an integer of at least 1.
        """
        if type(offset) is not int or offset < 1:
            raise ValueError(
                Declared(
                    "Capacitance.sub_diagonal offset",
                    "the call to Capacitance.sub_diagonal",
                ).expected("offset", "an integer of at least 1", offset)
            )
        return cls("sub_diagonal", offset=offset)

    @classmethod
    def off_diagonal(cls) -> Capacitance:
        """Each element ``C[a][b]`` of two different terminals, once per pair, ``a`` added before ``b``."""
        return cls("off_diagonal")

    @classmethod
    def matrix(cls) -> Capacitance:
        """The diagonal and each pair of different terminals once: the whole symmetric matrix."""
        return cls("matrix")

    @classmethod
    def elements(cls, pairs: Sequence[tuple[str, str]]) -> Capacitance:
        """The named elements ``C[a][b]``, each pair once.

        Raises:
            ValueError: ``pairs`` is empty or names a pair twice, or a name is not non-empty text.
        """
        where = "the call to Capacitance.elements"
        if (
            isinstance(pairs, (str, bytes))
            or not isinstance(pairs, Sequence)
            or not pairs
        ):
            raise ValueError(
                Declared("Capacitance.elements", where).refusal(
                    f"takes a non-empty list of (a, b) terminal pairs; {found_text(pairs)}"
                )
            )
        checked = tuple(_terminal_pair(pair, "Capacitance.elements") for pair in pairs)
        if len(set(checked)) != len(checked):
            raise ValueError(
                Declared("Capacitance.elements", where).refusal(
                    f"names a pair more than once; {found_text(list(checked))}"
                )
            )
        return cls("elements", checked)

    @property
    def label(self) -> str:
        """How the report names the quantity."""
        if self.selection == "element":
            return element_label(*self.pairs[0])
        if self.selection == "elements":
            return "C " + ", ".join(element_label(a, b) for a, b in self.pairs)
        if self.selection == "sub_diagonal":
            return f"C sub-diagonal (offset {self.offset})"
        return f"C {self.selection.replace('_', '-')}"

    def within(
        self, *, relative: float | None = None, absolute_f: float | None = None
    ) -> Criterion:
        """The criterion that each element changes from the previous pass by at most the threshold.

        ``relative`` bounds ``|C_k - C_(k-1)| / |C_k|``; ``absolute_f`` bounds ``|C_k - C_(k-1)|`` in farads.
        Give exactly one.  When ``C_k`` is zero the relative change is undefined and the criterion does not
        hold, unless ``C_(k-1)`` is zero too.

        Raises:
            ValueError: neither or both thresholds are given, or the threshold is not positive and finite.
        """
        return _change_criterion(self, relative, ("absolute_f", absolute_f))


def element_label(a: str, b: str) -> str:
    """How the report names element ``C[a][b]``."""
    return f"C[{a}][{b}]"


@dataclass(frozen=True)
class ModeFrequency:
    """The frequency of the selected modes of an Eigenmode run, in GHz (``Re{f}`` of ``eig.csv``).

    ``modes`` are mode numbers from 1, at most the run's ``num_modes`` (checked when ``write_config()``
    runs); None selects every computed mode.

    Raises on build:
        ValueError: ``modes`` is empty, repeats a mode, or holds something other than an integer of at least 1.
    """

    modes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Hold ``modes`` as a tuple of distinct mode numbers."""
        object.__setattr__(
            self, "modes", _modes(self.modes, "ModeFrequency(modes=...)")
        )

    @property
    def label(self) -> str:
        """How the report names the quantity."""
        return f"frequency, {_selected('mode', self.modes)}"

    def within(
        self, *, relative: float | None = None, absolute_ghz: float | None = None
    ) -> Criterion:
        """The criterion that each frequency changes from the previous pass by at most the threshold.

        ``relative`` bounds ``|f_k - f_(k-1)| / |f_k|``; ``absolute_ghz`` bounds ``|f_k - f_(k-1)|`` in GHz.
        Give exactly one.

        Raises:
            ValueError: neither or both thresholds are given, or the threshold is not positive and finite.
        """
        return _change_criterion(self, relative, ("absolute_ghz", absolute_ghz))


@dataclass(frozen=True)
class SurfaceEpr:
    """The Surface-EPR participation of one interface kind, one number per mode or per excitation.

    The participation is the sum over the kind's whole-surface rows, or, with ``margin_um``, over the
    kind's margin rows at that margin: the part of its surfaces beyond ``margin_um`` from the gap edge.  The
    two are never added together, because a margin row is a part of its whole-surface row.  On Eigenmode
    there is one number per mode (``modes``, None for every mode); on Electrostatic one per excitation, the
    terminal held at 1 V (``terminals``, None for every terminal).  The kind must have Surface EPR requested
    and ``margin_um`` must be a margin its spec declares; both are checked when ``write_config()`` runs.
    ``SurfaceEpr.each("MA", "MS", "SA")`` declares the same criterion for several kinds at once.

    Raises on build:
        ValueError: the kind is not ``MA``, ``MS`` or ``SA``, ``margin_um`` is not positive, a selection is
            empty or repeats an item, or both ``modes`` and ``terminals`` are given.
    """

    interface_kind: str
    margin_um: float | None = None
    modes: tuple[int, ...] | None = None
    terminals: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Check the kind, the margin and the selection, and hold the selection as a tuple."""
        where = "the call to SurfaceEpr"
        if self.interface_kind not in SURFACE_INTERFACES:
            raise ValueError(
                Declared("SurfaceEpr interface_kind", where).expected(
                    "interface_kind",
                    f"one of {SURFACE_INTERFACES}",
                    self.interface_kind,
                )
            )
        if self.margin_um is not None:
            if not _positive_finite(self.margin_um):
                raise ValueError(
                    Declared("SurfaceEpr margin_um", where).expected(
                        "margin_um",
                        "a positive number of micrometres, or None",
                        self.margin_um,
                    )
                )
            object.__setattr__(self, "margin_um", float(self.margin_um))
        if self.modes is not None and self.terminals is not None:
            raise ValueError(
                Declared("SurfaceEpr", where).refusal(
                    "selects modes (Eigenmode) and terminals (Electrostatic) at once; give the one of the "
                    "simulation it is declared on"
                )
            )
        object.__setattr__(self, "modes", _modes(self.modes, "SurfaceEpr(modes=...)"))
        object.__setattr__(
            self, "terminals", _terminals(self.terminals, "SurfaceEpr(terminals=...)")
        )

    @classmethod
    def each(
        cls,
        *interface_kinds: str,
        margin_um: float | None = None,
        modes: Sequence[int] | None = None,
        terminals: Sequence[str] | None = None,
    ) -> SurfaceEprKinds:
        """The same quantity for each of ``interface_kinds``, to be given one threshold with ``within``.

        Raises:
            ValueError: no kind is named, a kind is named twice, or a quantity is invalid.
        """
        if not interface_kinds or len(set(interface_kinds)) != len(interface_kinds):
            raise ValueError(
                Declared("SurfaceEpr.each", "the call to SurfaceEpr.each").refusal(
                    f"names each interface kind once, at least one of {SURFACE_INTERFACES}; "
                    f"{found_text(interface_kinds)}"
                )
            )
        return SurfaceEprKinds(
            tuple(
                cls(
                    kind,
                    margin_um,
                    None if modes is None else tuple(modes),
                    None if terminals is None else tuple(terminals),
                )
                for kind in interface_kinds
            )
        )

    @property
    def surface_label(self) -> str:
        """The kind and the surfaces summed, as each number's label starts: ``MA participation beyond 1 um``."""
        beyond = "" if self.margin_um is None else f" beyond {self.margin_um:g} um"
        return f"{self.interface_kind} participation{beyond}"

    @property
    def label(self) -> str:
        """How the report names the quantity: ``surface_label`` and the selected modes or excitations."""
        selection = (
            ""
            if self.modes is None and self.terminals is None
            else ", "
            + (
                _selected("mode", self.modes)
                if self.modes is not None
                else _selected("excitation", self.terminals)
            )
        )
        return f"{self.surface_label}{selection}"

    def within(
        self, *, relative: float | None = None, absolute: float | None = None
    ) -> Criterion:
        """The criterion that each participation changes from the previous pass by at most the threshold.

        ``relative`` bounds ``|p_k - p_(k-1)| / |p_k|``; ``absolute`` bounds ``|p_k - p_(k-1)|`` (a
        participation has no unit).  Give exactly one.

        Raises:
            ValueError: neither or both thresholds are given, or the threshold is not positive and finite.
        """
        return _change_criterion(self, relative, ("absolute", absolute))


@dataclass(frozen=True)
class SurfaceEprKinds:
    """``SurfaceEpr.each(...)``: one Surface-EPR quantity per interface kind, waiting for one threshold."""

    quantities: tuple[SurfaceEpr, ...]

    def within(
        self, *, relative: float | None = None, absolute: float | None = None
    ) -> CriterionGroup:
        """One criterion per kind, each with this threshold; see ``SurfaceEpr.within``."""
        return CriterionGroup(
            tuple(
                quantity.within(relative=relative, absolute=absolute)
                for quantity in self.quantities
            )
        )


@dataclass(frozen=True)
class ErrorIndicator:
    """The norm of Palace's AMR error indicator (``Norm`` of ``error-indicators.csv``), unitless, on both problems."""

    @property
    def label(self) -> str:
        """How the report names the quantity."""
        return "error indicator (Norm)"

    def below(self, value: float) -> Criterion:
        """The criterion that the indicator is at most ``value``: the only criterion on a value, not a change.

        Raises:
            ValueError: ``value`` is not positive and finite.
        """
        return Criterion(self, "at_most", _threshold(self, "value", value, "below"))


ConvergenceQuantity = Capacitance | ModeFrequency | SurfaceEpr | ErrorIndicator


# ---- Criteria and the declaration ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    """One quantity and its threshold: ``test`` says what the threshold bounds (``palace_convergence.Test``)."""

    quantity: ConvergenceQuantity
    test: Test
    threshold: float

    @property
    def label(self) -> str:
        """How the report names the criterion: its quantity's label."""
        return self.quantity.label


@dataclass(frozen=True)
class CriterionGroup:
    """Several criteria declared by one expression, ``SurfaceEpr.each(...).within(...)``."""

    criteria: tuple[Criterion, ...]


# How each problem's controls lose a stated amr_tolerance: Electrostatic's set_numerical resets an omitted
# control to its default, Eigenmode's keeps the current value, so only new controls drop it there.
_WITHOUT_TOLERANCE = {
    ELECTROSTATIC.name: "call set_numerical() without amr_tolerance",
    EIGENMODE.name: (
        "build the simulation with numerical controls that leave amr_tolerance out, since "
        "EigenmodeSim.set_numerical keeps an omitted amr_tolerance"
    ),
}

# The quantities each problem takes: the tables Palace writes for it.
_QUANTITIES: dict[str, tuple[type, ...]] = {
    ELECTROSTATIC.name: (Capacitance, SurfaceEpr, ErrorIndicator),
    EIGENMODE.name: (ModeFrequency, SurfaceEpr, ErrorIndicator),
}


@dataclass(frozen=True)
class ConvergenceDeclaration:
    """What ``set_convergence`` stores: the problem, the criteria in the order given, and the passes in a row."""

    problem: Problem
    criteria: tuple[Criterion, ...]
    minimum_converged_passes: int

    @classmethod
    def declared(
        cls,
        criteria: Sequence[Criterion | CriterionGroup],
        *,
        minimum_converged_passes: Any,
        problem: Problem,
        simulation: str,
    ) -> ConvergenceDeclaration:
        """Check ``set_convergence``'s arguments for a ``problem`` simulation and hold them.

        Raises:
            ValueError: no criterion is given, an argument is not a criterion, a quantity does not fit the
                problem, or ``minimum_converged_passes`` is not an integer of at least 1.
        """
        if not criteria:
            raise ValueError(
                Declared(
                    f"{simulation}.set_convergence()", _WHERE_SET_CONVERGENCE
                ).refusal(
                    "names no criterion: pass one or more, such as ErrorIndicator().below(...), or call "
                    "set_convergence(None) to remove the declaration"
                )
            )
        flat = tuple(
            item
            for argument in criteria
            for item in (
                argument.criteria
                if isinstance(argument, CriterionGroup)
                else (argument,)
            )
        )
        for criterion in flat:
            _check_fit(criterion, problem, simulation)
        if type(minimum_converged_passes) is not int or minimum_converged_passes < 1:
            raise ValueError(
                Declared("minimum_converged_passes", _WHERE_SET_CONVERGENCE).expected(
                    "minimum_converged_passes",
                    "an integer of at least 1",
                    minimum_converged_passes,
                )
            )
        return cls(problem, flat, minimum_converged_passes)

    @property
    def passes_needed(self) -> int:
        """The fewest solves in which the declaration can hold: one more than the passes in a row for a change."""
        measures_change = any(item.test != "at_most" for item in self.criteria)
        return self.minimum_converged_passes + (1 if measures_change else 0)

    def check_numerical(
        self, numerical: NumericalControls, *, amr_tolerance_stated: bool
    ) -> None:
        """Refuse numerical controls under which the declaration cannot decide when the run converged.

        With a declaration, the criteria decide and Palace's ``Tol`` must never end the run, so a stated
        ``amr_tolerance`` is refused with the ``ErrorIndicator`` criterion that replaces it.  The check reads
        every pass, so ``save_adapt_iterations=False`` is refused.  A run of ``amr_max_passes`` refinements
        solves ``amr_max_passes + 1`` times, which must reach ``passes_needed``.

        Raises:
            ValueError: one of those controls is refused.
        """
        if amr_tolerance_stated:
            raise ValueError(
                Declared("amr_tolerance", _WHERE_NUMERICAL_OR_CONVERGENCE).refusal(
                    "is set together with set_convergence(), which decides when the run has converged: "
                    f"{_WITHOUT_TOLERANCE[self.problem]}, and add the criterion "
                    f"ErrorIndicator().below({numerical.amr_tolerance!r}) to set_convergence()"
                )
            )
        if numerical.save_adapt_iterations is False:
            raise ValueError(
                Declared(
                    "save_adapt_iterations=False", _WHERE_NUMERICAL_OR_CONVERGENCE
                ).refusal(
                    "keeps Palace from saving each pass, which set_convergence() reads: call "
                    "set_numerical() without save_adapt_iterations, or with True"
                )
            )
        solves = numerical.amr_max_passes
        if isinstance(solves, int) and solves + 1 < self.passes_needed:
            raise ValueError(
                Declared(
                    f"amr_max_passes={solves}", _WHERE_NUMERICAL_OR_CONVERGENCE
                ).refusal(
                    f"allows {solves + 1} solve(s), and the declaration needs at least {self.passes_needed} "
                    f"to hold ({self.minimum_converged_passes} passes in a row"
                    + (
                        ", each compared with the pass before it"
                        if self.passes_needed > self.minimum_converged_passes
                        else ""
                    )
                    + f"): call set_numerical(amr_max_passes={self.passes_needed - 1}) or more, or lower "
                    "minimum_converged_passes"
                )
            )


def _check_fit(criterion: Any, problem: Problem, simulation: str) -> None:
    """Refuse an argument that is not a criterion, or a quantity the problem's tables do not hold."""
    taken = _QUANTITIES[problem]
    names = ", ".join(kind.__name__ for kind in taken)
    if not isinstance(criterion, Criterion):
        raise ValueError(
            Declared(f"{simulation}.set_convergence()", _WHERE_SET_CONVERGENCE).refusal(
                f"takes criteria such as {taken[0].__name__}....within(...) or ErrorIndicator().below(...); "
                f"{found_text(type(criterion).__name__)}"
            )
        )
    quantity = criterion.quantity
    if not isinstance(quantity, taken):
        raise ValueError(
            Declared(
                f"{type(quantity).__name__} criterion", _WHERE_SET_CONVERGENCE
            ).refusal(f"does not fit {simulation}, which takes {names}")
        )
    if isinstance(quantity, SurfaceEpr):
        wrong = (
            "terminals"
            if problem == EIGENMODE.name and quantity.terminals is not None
            else "modes"
            if problem == ELECTROSTATIC.name and quantity.modes is not None
            else None
        )
        if wrong is not None:
            right = "modes" if wrong == "terminals" else "terminals"
            raise ValueError(
                Declared(f"SurfaceEpr {wrong}=...", _WHERE_SET_CONVERGENCE).refusal(
                    f"does not fit {simulation}, whose rows are {right}: select with {right}=..."
                )
            )


def _change_criterion(
    quantity: Capacitance | ModeFrequency | SurfaceEpr,
    relative: float | None,
    absolute: tuple[str, float | None],
) -> Criterion:
    """The change criterion of the one threshold given, relative or absolute in the quantity's unit."""
    absolute_name, absolute_value = absolute
    if (relative is None) == (absolute_value is None):
        raise ValueError(
            Declared(
                f"{quantity.label} criterion",
                f"the call to {type(quantity).__name__}.within",
            ).refusal(
                f"takes exactly one threshold, relative=... or {absolute_name}=...; "
                f"{found_text({'relative': relative, absolute_name: absolute_value})}"
            )
        )
    if relative is not None:
        return Criterion(
            quantity,
            "relative_change",
            _threshold(quantity, "relative", relative, "within"),
        )
    return Criterion(
        quantity,
        "absolute_change",
        _threshold(quantity, absolute_name, absolute_value, "within"),
    )


def _threshold(
    quantity: ConvergenceQuantity, name: str, value: Any, call: str
) -> float:
    """``value`` as a float threshold; one that is not positive and finite is refused."""
    if not _positive_finite(value):
        raise ValueError(
            Declared(
                f"{quantity.label} criterion",
                f"the call to {type(quantity).__name__}.{call}",
            ).expected(name, "a positive finite number", value)
        )
    return float(value)


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _terminal_pair(pair: Any, call: str) -> tuple[str, str]:
    """A pair of terminal names; anything else is refused."""
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(name, str) and name for name in pair)
    ):
        raise ValueError(
            Declared(f"{call} terminal pair", f"the call to {call}").expected(
                "a pair", "(a, b) of two non-empty terminal names", pair
            )
        )
    return pair[0], pair[1]


def _modes(modes: Any, call: str) -> tuple[int, ...] | None:
    """Distinct mode numbers from 1, as a tuple, or None for every mode."""
    if modes is None:
        return None
    values = (
        tuple(modes)
        if isinstance(modes, Sequence) and not isinstance(modes, str)
        else None
    )
    if (
        not values
        or any(type(mode) is not int or mode < 1 for mode in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            Declared(call, f"the call to {call}").expected(
                "modes",
                "None or a non-empty list of distinct mode numbers from 1",
                modes,
            )
        )
    return values


def _terminals(terminals: Any, call: str) -> tuple[str, ...] | None:
    """Distinct terminal names, as a tuple, or None for every terminal."""
    if terminals is None:
        return None
    values = (
        tuple(terminals)
        if isinstance(terminals, Sequence) and not isinstance(terminals, str)
        else None
    )
    if (
        not values
        or any(not isinstance(name, str) or not name for name in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            Declared(call, f"the call to {call}").expected(
                "terminals",
                "None or a non-empty list of distinct terminal names",
                terminals,
            )
        )
    return values


def _selected(item: str, selection: tuple[Any, ...] | None) -> str:
    """``every mode``, ``mode 1`` or ``modes 1, 2``: a selection as a label says it."""
    if selection is None:
        return f"every {item}"
    names = ", ".join(str(value) for value in selection)
    return f"{item} {names}" if len(selection) == 1 else f"{item}s {names}"


__all__ = [
    "Capacitance",
    "ConvergenceDeclaration",
    "ConvergenceQuantity",
    "Criterion",
    "CriterionGroup",
    "ErrorIndicator",
    "ModeFrequency",
    "SurfaceEpr",
    "SurfaceEprKinds",
    "element_label",
]
