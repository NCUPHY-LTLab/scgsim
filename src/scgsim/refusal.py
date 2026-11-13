"""The one text form of a refusal that names a value: what is wrong, the value found, and the value expected.

Every stage builds such a message here, so a reader finds the same three parts in the same order wherever a
check refuses a record, a setting or a file.  Standard library only, so every stage can import it.
"""

from typing import Any


def refusal(what: str, *, found: Any, expected: Any = ...) -> str:
    """The text of a refusal: what is wrong, the value found and, when the check knows it, the value expected."""
    if expected is ...:
        return f"{what}: found {found!r}."
    return f"{what}: found {found!r}, expected {expected!r}."
