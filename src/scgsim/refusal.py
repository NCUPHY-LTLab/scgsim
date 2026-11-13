"""How SCGSim words a refusal: the one module every stage imports.

Every refusal has one of two shapes. "{object}: {problem}. Change it in {where}." is an input
someone has to change: ``refused`` words it, ``Declared`` words it for an object that carries its own
name and source, and ``found_text`` states the values a check compared. "internal: {invariant}. ..."
is an invariant SCGSim broke on an input it accepted: ``internal`` words it. Standard library only,
so every stage, ``semantics`` and ``meshing`` included, can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INTERNAL = "internal:"
ENVIRONMENT = "the Python environment SCGSim runs in"
"""Where a missing optional dependency or tool is set right."""
# A shown value is cut at this many characters, so a refusal of a whole record stays one line.
_SHOWN_CHARACTERS = 120


def shown(value: object) -> str:
    """A found value as a refusal prints it."""
    text = repr(value)
    if len(text) > _SHOWN_CHARACTERS:
        return text[: _SHOWN_CHARACTERS - 3] + "..."
    return text


def found_text(found: Any, expected: Any = ...) -> str:
    """The values a check compared: what it found and, when it knows it, what it expected."""
    if expected is ...:
        return f"found {shown(found)}"
    return f"found {shown(found)}, expected {shown(expected)}"


def refused(subject: str, problem: str, where: str) -> str:
    """The refusal of ``problem`` on ``subject``, which is changed in ``where``."""
    return f"{subject}: {problem}. Change it in {where}."


def internal(invariant: str) -> str:
    """The message of an invariant SCGSim broke on an input it accepted."""
    return (
        f"{INTERNAL} {invariant}. Nothing in the input is to blame: "
        "report it to SCGSim with the input that produced it."
    )


@dataclass(frozen=True)
class Declared:
    """A declared object a check reads: its name, and where it is declared, which says who changes it.

    ``name`` reads as a noun phrase (``"stack layer 'chip/pad'"``); ``source`` completes
    "Change it in ..." (``"the stack file"``, ``"the call to GeometryPlan.set_nets"``).
    """

    name: str
    source: str

    def refusal(self, problem: str) -> str:
        """The refusal of ``problem`` on this object."""
        return refused(self.name, problem, self.source)

    def expected(self, field: str, expected: str, *found: object) -> str:
        """The refusal of ``field`` of this object, which must be ``expected``; ``found`` is what it is."""
        found_text = f", not {shown(found[0])}" if found else ""
        return self.refusal(f"{field} must be {expected}{found_text}")

    def part(self, name: str) -> Declared:
        """A named part of this object, declared in the same place."""
        return Declared(f"{self.name} {name}", self.source)
