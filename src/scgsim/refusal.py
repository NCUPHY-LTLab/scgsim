"""The value part of a refusal ("found X, expected Y"), and the refusal sentences for modules below geometry.

Every refusal in SCGSim has one of two shapes, worded by ``scgsim.geometry.models.refusals``:
"{object}: {problem}. Change it in {where}." for an input someone has to change, and "internal:
{invariant}. ..." for an invariant SCGSim broke. ``palace``, ``visualization`` and ``presentation``
build both with that module and state the values they found with ``found_text`` from here.
``semantics`` and ``meshing`` may not import geometry (``.importlinter``), so they word the same two
sentences with ``refused`` and ``internal`` from here.  Standard library only, so every stage can
import it.
"""

from typing import Any

INTERNAL = "internal:"
# A shown value is cut at this many characters, so a refusal of a whole record stays one line.
_SHOWN_CHARACTERS = 120


def found_text(found: Any, expected: Any = ...) -> str:
    """The values a check compared: what it found and, when it knows it, what it expected."""
    if expected is ...:
        return f"found {_shown(found)}"
    return f"found {_shown(found)}, expected {_shown(expected)}"


def refused(subject: str, problem: str, where: str) -> str:
    """The refusal of ``problem`` on ``subject``, which is changed in ``where``."""
    return f"{subject}: {problem}. Change it in {where}."


def internal(invariant: str) -> str:
    """The message of an invariant SCGSim broke on an input it accepted."""
    return (
        f"{INTERNAL} {invariant}. Nothing in the input is to blame: "
        "report it to SCGSim with the input that produced it."
    )


def _shown(value: object) -> str:
    text = repr(value)
    if len(text) > _SHOWN_CHARACTERS:
        return text[: _SHOWN_CHARACTERS - 3] + "..."
    return text
