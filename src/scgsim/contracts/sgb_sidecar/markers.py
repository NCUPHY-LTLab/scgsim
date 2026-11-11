"""The markers the sidecar records attach to their types, each with the JSON Schema keyword it stands for.

Owns the small vocabulary :mod:`.records` needs beyond plain Python types: a description, the numeric and
length bounds the format states, a string pattern, and the mark of a record that accepts keys beyond its
fields.  A marker goes inside ``Annotated[...]`` (``Annotated[float, ExclusiveMinimum(0)]``) and only
:mod:`.schema` interprets it, through :meth:`keywords`; Python's type checker sees the plain type.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Doc:
    """A description of the annotated value."""

    text: str

    def keywords(self) -> dict[str, Any]:
        """``description``."""
        return {"description": self.text}


@dataclass(frozen=True)
class Minimum:
    """The smallest number allowed."""

    value: int

    def keywords(self) -> dict[str, Any]:
        """``minimum``."""
        return {"minimum": self.value}


@dataclass(frozen=True)
class ExclusiveMinimum:
    """A number the value must exceed (``ExclusiveMinimum(0)``: strictly positive)."""

    value: int

    def keywords(self) -> dict[str, Any]:
        """``exclusiveMinimum``."""
        return {"exclusiveMinimum": self.value}


@dataclass(frozen=True)
class Pattern:
    """A regular expression the string must contain a match of (anchor it with ``^`` and ``$``)."""

    regex: str

    def keywords(self) -> dict[str, Any]:
        """``pattern``."""
        return {"pattern": self.regex}


@dataclass(frozen=True)
class Length:
    """The fewest and the most items of a list (``Length(2, 2)``: exactly two)."""

    min_items: int | None = None
    max_items: int | None = None

    def keywords(self) -> dict[str, Any]:
        """``minItems`` and ``maxItems``, each when given."""
        bounds = {"minItems": self.min_items, "maxItems": self.max_items}
        return {name: value for name, value in bounds.items() if value is not None}


@dataclass(frozen=True)
class MinProperties:
    """The fewest keys of an object."""

    count: int

    def keywords(self) -> dict[str, Any]:
        """``minProperties``."""
        return {"minProperties": self.count}


Marker = Doc | Minimum | ExclusiveMinimum | Pattern | Length | MinProperties

_EXTRA_KEYS = "__sidecar_extra_keys__"


def open_record[Record](extra: object = Any) -> Callable[[Record], Record]:
    """Mark a ``TypedDict`` as accepting keys beyond its fields, each of type ``extra`` (any value by default).

    Every record without this mark is closed: a key it does not declare makes the sidecar invalid.
    """

    def mark(record: Record) -> Record:
        setattr(record, _EXTRA_KEYS, extra)
        return record

    return mark


def extra_keys(record: type) -> tuple[bool, object]:
    """Whether ``record`` itself is marked open (a subclass is not), and the type of its extra keys' values."""
    marked = vars(record)
    return _EXTRA_KEYS in marked, marked.get(_EXTRA_KEYS)
