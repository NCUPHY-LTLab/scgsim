"""How the geometry stages word a refusal: what is wrong, on which declared object, and who changes it.

A refusal of the user's input names the declared object (``Declared``): what it is, where it
is declared, and therefore who changes it. A check states only the field it read, what it
expected and what it found; the layer that knows the object builds the ``Declared`` it
passes down. A drawn polygon is refused in the same way through ``GridScope.refusal``,
which adds where in the layout the problem lies.

A message that opens with ``internal:`` names an invariant SCGSim itself broke on an input
it accepted: nothing in the input is to blame, so it asks for a report. ``internal`` is the
one definition of that wording; every geometry stage, SGB included, builds such a message
with it.
"""

from __future__ import annotations

from dataclasses import dataclass

INTERNAL = "internal:"
# A shown value is cut at this many characters, so a refusal of a whole record stays one line.
_SHOWN_CHARACTERS = 120


def internal(invariant: str) -> str:
    """The message of an invariant SCGSim broke on an input it accepted."""
    return (
        f"{INTERNAL} {invariant}. Nothing in the input is to blame: "
        "report it to SCGSim with the input that produced it."
    )


def shown(value: object) -> str:
    """A found value as a refusal prints it."""
    text = repr(value)
    if len(text) > _SHOWN_CHARACTERS:
        return text[: _SHOWN_CHARACTERS - 3] + "..."
    return text


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
        return f"{self.name}: {problem}. Change it in {self.source}."

    def expected(self, field: str, expected: str, *found: object) -> str:
        """The refusal of ``field`` of this object, which must be ``expected``; ``found`` is what it is."""
        found_text = f", not {shown(found[0])}" if found else ""
        return self.refusal(f"{field} must be {expected}{found_text}")

    def part(self, name: str) -> Declared:
        """A named part of this object, declared in the same place."""
        return Declared(f"{self.name} {name}", self.source)
