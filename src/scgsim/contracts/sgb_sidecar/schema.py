"""The JSON Schema of the sidecar, derived from :mod:`.records`, and the digest that pins it.

Owns the translation of the record definitions to JSON Schema draft 2020-12:

* a ``TypedDict`` is one ``$defs`` entry named after its class: an object whose ``properties`` are its fields
  and whose ``required`` are its required fields, closed by ``additionalProperties: false`` unless it is an
  ``open_record``; its docstring is the ``description``.  ``Sidecar`` itself is the root;
* ``str``, ``int``, ``float``, ``bool`` and ``None`` are ``string``, ``integer``, ``number``, ``boolean`` and
  ``null``; ``Any`` is the empty schema (any value);
* ``Literal`` is a ``const`` for one value and an ``enum`` for several;
* ``list[T]`` is an array of ``T``; ``dict[K, V]`` an object whose values are ``V``, and whose keys are the
  values of ``K`` when ``K`` is a ``Literal``, or ids of a table when ``K`` carries a ``Ref``;
* ``T | None`` adds ``null`` to the type of ``T`` or to its enum, and is a ``oneOf`` with ``null`` for a record;
  a union of records is a ``oneOf``;
* a marker inside ``Annotated`` adds its keyword (:mod:`.markers`).

The file ``scgsim/geometry/sgb/output/sidecar/sidecar.schema.json`` is :func:`schema_text` of this schema, written by
``scripts/generate_sidecar_schema.py``.
"""

import hashlib
import inspect
import json
import types
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from .markers import Marker, extra_keys
from .records import Sidecar
from .version import SIDECAR_SCHEMA

DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = f"https://scgsim.invalid/schemas/{SIDECAR_SCHEMA}.json"

_SCALAR_TYPES: dict[object, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}
# Keywords that hold for a value of any type: ``null`` cannot simply join the ``type`` of a schema that has one.
_TYPE_INDEPENDENT = ("enum", "const", "oneOf", "$ref")


def sidecar_schema() -> dict[str, Any]:
    """The JSON Schema of the sidecar, derived afresh from :class:`.records.Sidecar`."""
    deriver = _Deriver()
    root = deriver.record_body(Sidecar)
    return {
        "$schema": DIALECT,
        "$id": SCHEMA_ID,
        "title": SIDECAR_SCHEMA,
        **root,
        "$defs": dict(sorted(deriver.defs.items())),
    }


def schema_text(schema: dict[str, Any]) -> str:
    """The schema as its file holds it: keys in derivation order, indented by two spaces, a final newline.

    An object or list that holds no other object or list stays on one line, so a field reads as one line.
    """
    return _formatted(schema, 0) + "\n"


def schema_digest(schema: dict[str, Any]) -> str:
    """``sha256:<hex>`` of the schema's canonical JSON (keys sorted, no whitespace, ASCII)."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


class _Deriver:
    """Translates annotations to schemas, collecting one ``$defs`` entry for every record it meets."""

    def __init__(self) -> None:
        self.defs: dict[str, dict[str, Any]] = {}
        self._records: dict[str, object] = {}

    def record_body(self, record: Any) -> dict[str, Any]:
        """The object schema of a ``TypedDict``: its fields, the required ones, and whether it takes more keys."""
        fields = get_type_hints(record, include_extras=True)
        body: dict[str, Any] = {}
        if record.__doc__:
            body["description"] = _description(record.__doc__)
        body["type"] = "object"
        is_open, extra = extra_keys(record)
        if not is_open:
            body["additionalProperties"] = False
        elif extra is not Any:
            body["additionalProperties"] = self.schema_of(extra)
        required = [name for name in fields if name in record.__required_keys__]
        if required:
            body["required"] = required
        if fields:
            body["properties"] = {
                name: self.schema_of(annotation) for name, annotation in fields.items()
            }
        return body

    def schema_of(self, annotation: Any) -> dict[str, Any]:
        """The schema of one annotation."""
        origin = get_origin(annotation)
        if origin is Annotated:
            return self._annotated(annotation)
        if origin is Union or origin is types.UnionType:
            return self._union(get_args(annotation))
        if origin is Literal:
            return _literal(get_args(annotation))
        if origin is list:
            (item,) = get_args(annotation)
            return {"type": "array", "items": self.schema_of(item)}
        if origin is dict:
            key, value = get_args(annotation)
            return self._mapping(key, value)
        if origin is Required or origin is NotRequired:
            raise TypeError(
                "a postponed Required/NotRequired is invisible at run time; declare the optional fields "
                "in a total=False subclass instead"
            )
        if is_typeddict(annotation):
            return self._record_ref(annotation)
        if annotation is Any:
            return {}
        if annotation in _SCALAR_TYPES:
            return {"type": _SCALAR_TYPES[annotation]}
        raise TypeError(
            f"the sidecar definition uses {annotation!r}, which has no schema here"
        )

    def _annotated(self, annotation: Any) -> dict[str, Any]:
        base, *markers = get_args(annotation)
        schema = self.schema_of(base)
        for marker in markers:
            if not isinstance(marker, Marker):
                raise TypeError(
                    f"{marker!r} is not a sidecar marker (scgsim.contracts.sgb_sidecar.markers)"
                )
            schema = _with_keywords(schema, marker.keywords())
        return schema

    def _union(self, members: tuple[Any, ...]) -> dict[str, Any]:
        options = [member for member in members if member is not type(None)]
        nullable = len(options) < len(members)
        if all(get_origin(option) is Literal for option in options):
            values = [value for option in options for value in get_args(option)]
            return {"enum": [*values, None] if nullable else values}
        if len(options) == 1:
            schema = self.schema_of(options[0])
            return _or_null(schema) if nullable else schema
        if nullable:
            raise TypeError(
                f"a union of several types and None ({members!r}) has no schema here"
            )
        return {"oneOf": [self.schema_of(option) for option in options]}

    def _mapping(self, key: Any, value: Any) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "object"}
        if value is not Any:
            schema["additionalProperties"] = self.schema_of(value)
        if get_origin(key) is Literal:
            schema["propertyNames"] = {"enum": list(get_args(key))}
        elif get_origin(key) is Annotated and get_args(key)[0] is str:
            # Keys that are ids of a table: the marker's annotation, which every string key passes.
            schema["propertyNames"] = self._annotated(key)
            schema["propertyNames"].pop("type")
        elif key is not str:
            raise TypeError(f"a JSON object's keys are strings, not {key!r}")
        return schema

    def _record_ref(self, record: Any) -> dict[str, Any]:
        name = record.__name__
        if self._records.setdefault(name, record) is not record:
            raise TypeError(f"two sidecar records are named {name}")
        if name not in self.defs:
            self.defs[
                name
            ] = {}  # reserved first, so a record that refers to itself ends here
            self.defs[name] = self.record_body(record)
        return {"$ref": f"#/$defs/{name}"}


def _formatted(value: Any, depth: int) -> str:
    if _fits_one_line(value):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    indent = "  " * (depth + 1)
    if isinstance(value, dict):
        lines = [
            f"{indent}{json.dumps(k)}: {_formatted(v, depth + 1)}"
            for k, v in value.items()
        ]
        opening, closing = "{", "}"
    else:
        lines = [f"{indent}{_formatted(v, depth + 1)}" for v in value]
        opening, closing = "[", "]"
    return opening + "\n" + ",\n".join(lines) + "\n" + "  " * depth + closing


def _fits_one_line(value: Any) -> bool:
    """A scalar, a list of scalars, or an object whose values are those (``{"enum": ["A", "B"]}``)."""
    if isinstance(value, dict):
        return all(_is_scalar_or_scalar_list(v) for v in value.values())
    return _is_scalar_or_scalar_list(value)


def _is_scalar_or_scalar_list(value: Any) -> bool:
    if isinstance(value, list):
        return not any(isinstance(item, dict | list) for item in value)
    return not isinstance(value, dict)


def _description(docstring: str) -> str:
    """A docstring as one line per paragraph: the wrapping of the source is not part of the text."""
    paragraphs = inspect.cleandoc(docstring).split("\n\n")
    return "\n\n".join(" ".join(paragraph.split("\n")) for paragraph in paragraphs)


def _literal(values: tuple[Any, ...]) -> dict[str, Any]:
    return {"const": values[0]} if len(values) == 1 else {"enum": list(values)}


def _or_null(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` or null: null joins the enum or the type when it can, else a ``oneOf`` with null."""
    if "enum" in schema:
        return {**schema, "enum": [*schema["enum"], None]}
    if isinstance(schema.get("type"), str) and not any(
        k in schema for k in _TYPE_INDEPENDENT
    ):
        # Every other keyword this module writes holds only for values of its own type, so null passes them.
        return {**schema, "type": [schema["type"], "null"]}
    return {"oneOf": [schema, {"type": "null"}]}


def _with_keywords(schema: dict[str, Any], keywords: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with ``keywords`` added, its description first so the file reads top-down."""
    merged = {**schema, **keywords}
    if "description" in merged:
        merged = {"description": merged.pop("description"), **merged}
    return merged
