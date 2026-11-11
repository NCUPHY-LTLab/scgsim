"""Checking a decoded JSON value against the schema derived from the sidecar records.

Owns the subset of JSON Schema draft 2020-12 that :mod:`.schema` writes, with the standard's meaning: an
``integer`` may be written ``2.0``, a boolean is neither an integer nor a number, ``pattern`` searches the
string, a keyword of objects, arrays, strings or numbers applies only to a value of that type, and a
``oneOf`` holds when exactly one branch does.  Every number is finite, as a JSON number is, wherever it is.  :func:`require_known_keywords` refuses a schema with any other
keyword, so the schema never states more than this check enforces.  The first mismatch raises
:class:`.errors.SidecarError` with its place in the document.
"""

import math
import re
from typing import Any

from .errors import SidecarError

Path = tuple[str | int, ...]

KEYWORDS = frozenset(
    {
        *("$ref", "oneOf", "type", "enum", "const"),
        *(
            "properties",
            "required",
            "additionalProperties",
            "propertyNames",
            "minProperties",
        ),
        *("items", "minItems", "maxItems", "minimum", "exclusiveMinimum", "pattern"),
    }
)
ANNOTATIONS = frozenset({"$schema", "$id", "title", "description", "$defs"})


def require_structure(value: object, schema: dict[str, Any]) -> None:
    """Raise :class:`SidecarError` at the first place ``value`` does not match the root ``schema``."""
    try:
        _require_finite_numbers(value, ())
        _Checker(schema.get("$defs", {})).check(value, schema, ())
    except _Mismatch as mismatch:
        raise SidecarError(
            f"sidecar {_place(mismatch.path)}: {mismatch.reason}"
        ) from None


def require_known_keywords(schema: object) -> None:
    """Raise ``ValueError`` when ``schema``, or a schema inside it, uses a keyword this module does not check."""
    if isinstance(schema, list):
        for branch in schema:
            require_known_keywords(branch)
        return
    if not isinstance(schema, dict):
        return
    unknown = set(schema) - KEYWORDS - ANNOTATIONS
    if unknown:
        raise ValueError(
            f"the sidecar schema uses {sorted(unknown)}, which the shared reader does not check"
        )
    for keyword in ("$defs", "properties"):
        for subschema in schema.get(keyword, {}).values():
            require_known_keywords(subschema)
    for keyword in ("oneOf", "items", "additionalProperties", "propertyNames"):
        require_known_keywords(schema.get(keyword))


def _require_finite_numbers(value: Any, path: Path) -> None:
    """Every number of ``value`` is finite, as a JSON number is, the values no schema describes included.

    Python's JSON reader decodes a number beyond the float range, such as ``1e400``, as an infinity, and a
    document built in memory can hold an infinity or a NaN.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise _Mismatch(path, f"{value!r} is not a finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, (*path, index))


class _Mismatch(Exception):
    """Where the value stopped matching, why, and whether a ``const`` refused it (a record kind it is not)."""

    def __init__(self, path: Path, reason: str, by_const: bool = False) -> None:
        super().__init__(reason)
        self.path = path
        self.reason = reason
        self.by_const = by_const


class _Checker:
    """Checks values against schemas that refer to one set of ``$defs``."""

    def __init__(self, defs: dict[str, Any]) -> None:
        self.defs = defs

    def check(self, value: Any, schema: dict[str, Any], path: Path) -> None:
        """Raise :class:`_Mismatch` unless ``value`` matches every keyword of ``schema``."""
        if "$ref" in schema:
            self.check(value, self.defs[schema["$ref"].removeprefix("#/$defs/")], path)
        if "oneOf" in schema:
            self._one_of(value, schema["oneOf"], path)
        if "type" in schema:
            _require_type(value, schema["type"], path)
        if "enum" in schema and not any(_json_equal(value, v) for v in schema["enum"]):
            raise _Mismatch(path, f"{_shown(value)} is not one of {schema['enum']}")
        if "const" in schema and not _json_equal(value, schema["const"]):
            raise _Mismatch(
                path, f"{_shown(value)} is not {schema['const']!r}", by_const=True
            )
        if isinstance(value, dict):
            self._object(value, schema, path)
        elif isinstance(value, list):
            self._array(value, schema, path)
        elif isinstance(value, str):
            _string(value, schema, path)
        elif _is_number(value):
            _number(value, schema, path)

    def _one_of(self, value: Any, branches: list[dict[str, Any]], path: Path) -> None:
        failed = [
            m
            for m in (self._mismatch(value, b, path) for b in branches)
            if m is not None
        ]
        matched = len(branches) - len(failed)
        if matched == 1:
            return
        if matched > 1:
            raise _Mismatch(
                path, f"matches {matched} alternatives, where exactly one must match"
            )
        # The writer meant an alternative whose kind constant matched, and of those the one that got furthest
        # into the value; when several remain, name them all.
        meant = [m for m in failed if not m.by_const] or failed
        depth = max(len(m.path) for m in meant)
        deepest = [m for m in meant if len(m.path) == depth]
        raise _Mismatch(
            deepest[0].path, " / ".join(dict.fromkeys(m.reason for m in deepest))
        )

    def _mismatch(
        self, value: Any, schema: dict[str, Any], path: Path
    ) -> _Mismatch | None:
        try:
            self.check(value, schema, path)
        except _Mismatch as mismatch:
            return mismatch
        return None

    def _object(
        self, value: dict[str, Any], schema: dict[str, Any], path: Path
    ) -> None:
        for name in schema.get("required", ()):
            if name not in value:
                raise _Mismatch(path, f"lacks the field {name!r}")
        if len(value) < schema.get("minProperties", 0):
            raise _Mismatch(
                path, f"has {len(value)} keys, fewer than {schema['minProperties']}"
            )
        fields = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        names = schema.get("propertyNames")
        for name, item in value.items():
            if (
                names is not None
                and (m := self._mismatch(name, names, path)) is not None
            ):
                raise _Mismatch(path, f"the key {m.reason}")
            if name in fields:
                self.check(item, fields[name], (*path, name))
            elif extra is False:
                raise _Mismatch(path, f"has the unknown field {name!r}")
            elif isinstance(extra, dict):
                self.check(item, extra, (*path, name))

    def _array(self, value: list[Any], schema: dict[str, Any], path: Path) -> None:
        if len(value) < schema.get("minItems", 0):
            raise _Mismatch(
                path, f"has {len(value)} items, fewer than {schema['minItems']}"
            )
        if len(value) > schema.get("maxItems", math.inf):
            raise _Mismatch(
                path, f"has {len(value)} items, more than {schema['maxItems']}"
            )
        if "items" in schema:
            for index, item in enumerate(value):
                self.check(item, schema["items"], (*path, index))


def _require_type(value: Any, names: str | list[str], path: Path) -> None:
    allowed = [names] if isinstance(names, str) else names
    if not any(_has_type(value, name) for name in allowed):
        raise _Mismatch(path, f"{_shown(value)} is not {' or '.join(allowed)}")


def _has_type(value: Any, name: str) -> bool:
    if name == "integer":
        return _is_number(value) and (isinstance(value, int) or value.is_integer())
    if name == "number":
        return _is_number(value)
    return isinstance(value, _JSON_TYPES[name])


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _string(value: str, schema: dict[str, Any], path: Path) -> None:
    if "pattern" in schema and re.search(schema["pattern"], value) is None:
        raise _Mismatch(path, f"{value!r} does not match {schema['pattern']!r}")


def _number(value: float, schema: dict[str, Any], path: Path) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise _Mismatch(path, f"{value!r} is less than {schema['minimum']}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise _Mismatch(
            path, f"{value!r} is not greater than {schema['exclusiveMinimum']}"
        )


def _json_equal(value: Any, expected: Any) -> bool:
    """JSON equality of scalars: ``2 == 2.0``, but ``true`` is not ``1``."""
    if isinstance(value, bool) or isinstance(expected, bool):
        return value is expected
    if _is_number(value) and _is_number(expected):
        return value == expected
    return type(value) is type(expected) and value == expected


def _shown(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else f"{text[:57]}..."


def _place(path: Path) -> str:
    """``physical_groups['MA::pad__top'].band.index`` from the path of keys and indices."""
    parts = [
        f"[{p}]"
        if isinstance(p, int)
        else (f".{p}" if p.isidentifier() else f"[{p!r}]")
        for p in path
    ]
    return "".join(parts).removeprefix(".") or "document"
