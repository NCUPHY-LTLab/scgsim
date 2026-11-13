"""The references of a sidecar, checked after its structure: every id names a record that exists, of the kind
it needs, and the binding names exactly the written shapes.

Owns two kinds of rule.  *Typed references* come from the definition itself: a field whose type carries a
``Ref`` (:mod:`.markers`) holds the id of a record of that table, or for a group of that dimension, and
:func:`require_references` walks every record with the types of :mod:`.records` and checks each one, keys of
an object included.  *Table rules* are the facts that tie tables together:

* ``binding.ids_by_tag`` names each planned id once per dimension, exactly the vertex, edge and face tables and
  the emitted volumes, as many as ``summary.counts`` says were written; ``counts.volumes_total`` is the number of
  planned volumes;
* a group's tag is used once per dimension, and every group has a member (a face or emitted volume names it);
* a support ring lists exactly the primitives that name it;
* a face loop marks the direction of each of its edges, and a feature names each edge once;
* a face is on the outer boundary exactly when one of its side volumes is null, and the domain it bounds is
  the Entity of the other.

Not checked, because the sidecar keeps no table of them: the Entities a caller excluded from a type-level
request, which are the request as given.
"""

import functools
import types
from collections.abc import Callable, Iterator
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from .errors import SidecarError
from .markers import Ref
from .records import Sidecar

_DIMENSIONS = ("vertex", "edge", "face", "volume")
_EMITTED_TABLES = ("vertex", "edge", "face")
_SHOWN = 5  # unresolved references named in the error; the rest are counted
_KEY = "<key>"  # the place of an object's keys in a path

Path = tuple[str | int, ...]
Problems = list[str]
# A compiled check of one annotation: the value, where it is, the tables, and the list problems are added to.
_Check = Callable[[Any, Path, "_Tables", Problems], None]


def require_references(sidecar: Sidecar) -> None:
    """Raise :class:`SidecarError` naming the first references of ``sidecar`` that do not resolve."""
    tables = _Tables(sidecar)
    problems: Problems = []
    check = _compiled(Sidecar)
    if check is not None:
        check(sidecar, (), tables, problems)
    problems += [
        *_binding(sidecar),
        *_groups(sidecar),
        *_rings(sidecar),
        *_loops_and_features(sidecar),
        *_outer_boundaries(sidecar),
    ]
    if problems:
        more = f" (and {len(problems) - _SHOWN} more)" if len(problems) > _SHOWN else ""
        raise SidecarError(
            f"sidecar references do not resolve: {'; '.join(problems[:_SHOWN])}{more}"
        )


class _Tables:
    """The tables of one sidecar that ids resolve against, and the dimension of each group."""

    def __init__(self, sidecar: Sidecar) -> None:
        self.sidecar: Any = sidecar
        self.dimension = {
            name: group["dim"] for name, group in sidecar["group"].items()
        }

    def problem(self, ref: Ref, value: str, path: Path) -> str | None:
        """Why ``value`` at ``path`` does not resolve as ``ref`` says, or ``None`` when it does."""
        if value not in self.sidecar[ref.table]:
            return f"{_place(path)} names {value!r}, which {ref.table} lacks"
        if ref.dimension is not None and self.dimension[value] != ref.dimension:
            return f"{_place(path)} names the group {value!r}, which is not a {_DIMENSIONS[ref.dimension]} group"
        return None


# --------------------------------------------------------------------------- typed references


@functools.cache
def _compiled(annotation: Any) -> _Check | None:
    """The check of every reference a value of ``annotation`` holds, or ``None`` when it can hold none."""
    origin = get_origin(annotation)
    if origin is Annotated:
        base, *markers = get_args(annotation)
        refs = [m for m in markers if isinstance(m, Ref)]
        return _reference(refs[0]) if refs else _compiled(base)
    if origin is Union or origin is types.UnionType:
        return _union(get_args(annotation))
    if origin is list:
        return _items(_compiled(get_args(annotation)[0]))
    if origin is dict:
        key, value = get_args(annotation)
        return _entries(_compiled(key), _compiled(value))
    if is_typeddict(annotation):
        return _record(annotation)
    return None


def _reference(ref: Ref) -> _Check:
    def check(value: Any, path: Path, tables: _Tables, problems: Problems) -> None:
        if value is not None and (problem := tables.problem(ref, value, path)):
            problems.append(problem)

    return check


def _union(members: tuple[Any, ...]) -> _Check | None:
    """A union of scalars, ``None`` or one type; or of records, each value checked as the one record it is."""
    options = [m for m in members if m is not type(None)]
    if len(options) == 1:
        return _unless_null(_compiled(options[0]))
    if not all(is_typeddict(o) for o in options):
        return None  # a union of scalar types holds no reference
    branches = [(_Shape.of(o), _compiled(o)) for o in options]
    if all(check is None for _, check in branches):
        return None

    def check(value: Any, path: Path, tables: _Tables, problems: Problems) -> None:
        if not isinstance(value, dict):
            return
        for shape, branch in branches:
            if shape.matches(value):
                if branch is not None:
                    branch(value, path, tables, problems)
                return

    return check


def _unless_null(inner: _Check | None) -> _Check | None:
    """``inner``, for a value that may be null instead (a null holds no reference)."""
    if inner is None:
        return None

    def check(value: Any, path: Path, tables: _Tables, problems: Problems) -> None:
        if value is not None:
            inner(value, path, tables, problems)

    return check


class _Shape:
    """What tells one closed record of a union from the others: its constant fields (a ``kind`` or ``dim``),
    the fields it requires and the fields it allows.  The structure check already proved exactly one matches."""

    def __init__(
        self,
        constants: dict[str, Any],
        required: frozenset[str],
        allowed: frozenset[str],
    ) -> None:
        self.constants = constants
        self.required = required
        self.allowed = allowed

    @classmethod
    def of(cls, record: Any) -> "_Shape":
        """The shape of the ``TypedDict`` ``record``."""
        hints = get_type_hints(record, include_extras=True)
        constants = {}
        for name, annotation in hints.items():
            base = (
                get_args(annotation)[0]
                if get_origin(annotation) is Annotated
                else annotation
            )
            if get_origin(base) is Literal and len(get_args(base)) == 1:
                constants[name] = get_args(base)[0]
        return cls(constants, frozenset(record.__required_keys__), frozenset(hints))

    def matches(self, value: dict[str, Any]) -> bool:
        """Whether ``value`` is this record."""
        keys = value.keys()
        return (
            self.required <= keys
            and keys <= self.allowed
            and all(value.get(k) == v for k, v in self.constants.items())
        )


def _items(item: _Check | None) -> _Check | None:
    if item is None:
        return None

    def check(value: Any, path: Path, tables: _Tables, problems: Problems) -> None:
        for index, entry in enumerate(value):
            item(entry, (*path, index), tables, problems)

    return check


def _entries(key: _Check | None, item: _Check | None) -> _Check | None:
    if key is None and item is None:
        return None

    def check(value: Any, path: Path, tables: _Tables, problems: Problems) -> None:
        for name, entry in value.items():
            if key is not None:
                key(name, (*path, _KEY), tables, problems)
            if item is not None:
                item(entry, (*path, name), tables, problems)

    return check


def _record(record: Any) -> _Check | None:
    fields = [
        (name, field_check)
        for name, annotation in get_type_hints(record, include_extras=True).items()
        if (field_check := _compiled(annotation)) is not None
    ]
    if not fields:
        return None

    def check(value: Any, path: Path, tables: _Tables, problems: Problems) -> None:
        for name, field in fields:
            if name in value:
                field(value[name], (*path, name), tables, problems)

    return check


# --------------------------------------------------------------------------- table rules


def _binding(sidecar: Sidecar) -> Iterator[str]:
    """The binding names each written shape once, exactly the shape tables, as many as ``counts`` says."""
    by_tag = sidecar["binding"]["ids_by_tag"]
    counts = sidecar["summary"]["counts"]
    volumes = sidecar["volume"]
    if counts["volumes_total"] != len(volumes):
        yield f"summary.counts.volumes_total is {counts['volumes_total']}, volume has {len(volumes)}"
    written = (counts["vertices"], counts["edges"], counts["faces"], counts["solids"])
    emitted = {v for v, volume in volumes.items() if volume["emitted"]}
    tables = (*(set(sidecar[t]) for t in _EMITTED_TABLES), emitted)
    for dimension, (count, table) in enumerate(zip(written, tables, strict=True)):
        ids = by_tag[str(dimension)]
        if len(set(ids)) != len(ids):
            yield f"binding.ids_by_tag['{dimension}'] names a planned id more than once"
        if count != len(ids):
            yield f"summary.counts has {count} of dimension {dimension}, binding.ids_by_tag['{dimension}'] {len(ids)}"
        if set(ids) != table:
            what = (
                "the emitted volumes"
                if dimension == 3
                else f"the {_DIMENSIONS[dimension]} table"
            )
            yield f"binding.ids_by_tag['{dimension}'] does not name exactly {what}"


def _groups(sidecar: Sidecar) -> Iterator[str]:
    """Each group's tag once per dimension, and every group with a member of its dimension."""
    named: dict[tuple[int, int], str] = {}
    for name, group in sidecar["group"].items():
        key = (group["dim"], group["tag"])
        if key in named:
            yield f"group {name!r} has the tag {group['tag']} of group {named[key]!r}"
        named.setdefault(key, name)
    members = {f["group"] for f in sidecar["face"].values()} | {
        v["group"] for v in sidecar["volume"].values() if v["emitted"]
    }
    for name in sidecar["group"]:
        if name not in members:
            yield f"group {name!r} has no member"
    for volume_id, volume in sidecar["volume"].items():
        if volume["group"] is not None and not volume["emitted"]:
            yield f"volume[{volume_id!r}] is not emitted but names the group {volume['group']!r}"


def _rings(sidecar: Sidecar) -> Iterator[str]:
    """A support ring lists exactly the primitives that name it, each once."""
    listed: dict[str, str] = {}
    for ring_id, ring in sidecar["support_ring"].items():
        for primitive in ring["primitives"]:
            if primitive in listed:
                yield f"support_primitive {primitive!r} is listed by two rings"
            listed[primitive] = ring_id
    for primitive_id, primitive in sidecar["support_primitive"].items():
        if listed.get(primitive_id) != primitive["ring"]:
            yield f"support_primitive {primitive_id!r} names the ring {primitive['ring']!r}, which does not list it"


def _outer_boundaries(sidecar: Sidecar) -> Iterator[str]:
    """``outer_boundary`` exactly on the faces with one side volume null, naming the Entity of the other side."""
    volumes = sidecar["volume"]
    for face_id, face in sidecar["face"].items():
        sides = (face["negative_volume"], face["positive_volume"])
        outer = face["outer_boundary"]
        if (outer is not None) != (sides.count(None) == 1):
            yield f"face[{face_id!r}] has outer_boundary {outer!r} with side volumes {list(sides)}"
            continue
        inside = sides[0] or sides[1]
        if outer is not None and inside in volumes:
            entity = volumes[inside]["entity"]
            if outer["domain"] != entity:
                yield f"face[{face_id!r}].outer_boundary names the domain {outer['domain']!r}, its volume is {entity!r}"


def _loops_and_features(sidecar: Sidecar) -> Iterator[str]:
    """Every loop marks the direction of each edge; every mesh feature names each edge once."""
    for face_id, face in sidecar["face"].items():
        for position, loop in enumerate(face["loops"]):
            if len(loop["edges"]) != len(loop["reversed"]):
                yield f"face[{face_id!r}].loops[{position}] has {len(loop['edges'])} edges and {len(loop['reversed'])} directions"
    for feature_id, feature in sidecar["mesh_feature"].items():
        if len(set(feature["edges"])) != len(feature["edges"]):
            yield f"mesh_feature[{feature_id!r}] names an edge more than once"


def _place(path: Path) -> str:
    """``face['f:1'].loops[0].edges[2]`` from the path of keys and indices (``membership key``: its keys)."""
    parts = []
    for part in path:
        if part == _KEY:
            parts.append(" key")
        elif isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}" if part.isidentifier() else f"[{part!r}]")
    return "".join(parts).removeprefix(".") or "document"
