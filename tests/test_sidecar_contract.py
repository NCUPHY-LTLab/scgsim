"""The sidecar contract ``scgsim.contracts.sgb_sidecar`` (#170, slice A): one definition gives the schema file, the
pinned digest beside the version, and the shared reader, and they describe ``scgsim.sgb.sidecar.v2`` as written."""

from __future__ import annotations

import copy
import json
import re
from importlib import resources

import jsonschema
import pytest
from sgb_emit_cases import CASES

from scgsim.contracts.sgb_sidecar import (
    SCHEMA_DIGEST,
    SIDECAR_SCHEMA,
    SidecarError,
    check_sidecar,
    read_sidecar,
    schema_digest,
    sidecar_schema,
)
from scgsim.contracts.sgb_sidecar.schema import schema_text
from scgsim.contracts.sgb_sidecar.structure import (
    require_known_keywords,
    require_structure,
)
from scgsim.geometry.sgb.emission.plan import plan_brep
from scgsim.geometry.sgb.sidecar.build import SidecarContext, build_sidecar
from scgsim.geometry.sgb.sidecar.files import SCHEMA_FILE, write_sidecar
from scgsim.geometry.sgb.slivers.report import sliver_report


@pytest.fixture(scope="module")
def sidecars(emit_models):
    """The sidecar of every emit case as a reader decodes it from its file."""
    return {
        name: json.loads(
            json.dumps(
                build_sidecar(
                    model,
                    plan_brep(model.topology, model.routed),
                    SidecarContext(sliver_report(model)["summary"]),
                )
            )
        )
        for name, model in emit_models.items()
    }


# --------------------------------------------------------------------------- schema, file and version


def test_the_schema_file_is_the_schema_derived_from_the_records():
    """``sidecar.schema.json`` is the derived schema; after a change to the records run scripts/generate_sidecar_schema.py."""
    committed = (
        resources.files("scgsim.geometry.sgb.sidecar")
        .joinpath(SCHEMA_FILE)
        .read_text(encoding="utf-8")
    )
    assert committed == schema_text(sidecar_schema())


def test_the_definition_does_not_change_without_a_new_version():
    """The digest pinned beside the version is the digest of the derived schema.

    When this fails the definition changed: raise ``SIDECAR_SCHEMA`` in ``scgsim/contracts/sgb_sidecar/version.py``,
    move every reader to the new version in the same change, and pin the digest this test prints.
    """
    derived = schema_digest(sidecar_schema())
    assert derived == SCHEMA_DIGEST, (
        f"the sidecar definition changed; raise the version and pin {derived}"
    )


def test_the_derived_schema_is_a_draft_2020_12_schema_the_reader_enforces_in_full():
    """A valid draft 2020-12 schema, of the one version, with no keyword the shared reader leaves unchecked."""
    schema = sidecar_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    require_known_keywords(schema)
    assert schema["title"] == SIDECAR_SCHEMA
    assert schema["properties"]["schema"] == {"const": SIDECAR_SCHEMA}


def test_a_schema_keyword_the_reader_does_not_check_is_refused():
    """The reader cannot silently pass what the schema would state: an unknown keyword stops it."""
    with pytest.raises(ValueError, match="uniqueItems"):
        require_known_keywords(
            {
                "type": "object",
                "properties": {"x": {"type": "array", "uniqueItems": True}},
            }
        )


# --------------------------------------------------------------------------- the shared reader accepts what SGB writes


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_emitted_sidecar_passes_the_shared_reader_unchanged(sidecars, name):
    """Version, structure and references of every emit case hold; the document comes back as it went in."""
    document = sidecars[name]
    before = copy.deepcopy(document)
    assert check_sidecar(document) is document
    assert document == before


def test_read_sidecar_returns_the_written_file_as_decoded(tmp_path, sidecars):
    """The file SGB writes reads back equal to the dictionary it was written from."""
    path = tmp_path / "pad_b.sidecar.json"
    write_sidecar(path, sidecars["pad_b"])
    assert read_sidecar(path) == sidecars["pad_b"]


# --------------------------------------------------------------------------- what the reader refuses, and how it says so


def test_another_version_is_refused_by_name(sidecars):
    """A sidecar of another version is named as such, before its structure is looked at."""
    with pytest.raises(
        SidecarError, match=r"'scgsim\.sgb\.sidecar\.v1' is not supported.*sidecar\.v2"
    ):
        check_sidecar({"schema": "scgsim.sgb.sidecar.v1"})
    with pytest.raises(SidecarError, match="a sidecar is a JSON object, not list"):
        check_sidecar([sidecars["pad_b"]])


@pytest.mark.parametrize("text", ["{", '{"grid_um": NaN}', '{"grid_um": Infinity}'])
def test_a_file_that_is_not_json_is_refused(tmp_path, text):
    """Python's JSON reader accepts NaN and Infinity; a sidecar never holds them."""
    path = tmp_path / "broken.sidecar.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SidecarError, match="broken.sidecar.json is not JSON"):
        read_sidecar(path)


def test_a_number_beyond_the_float_range_is_refused_where_it_is(tmp_path, sidecars):
    """JSON allows ``1e400``, which Python decodes as an infinity; no sidecar number is infinite."""
    path = tmp_path / "huge.sidecar.json"
    marked = json.dumps(_with(sidecars["pad_b"], ("grid_um",), "HUGE"))
    path.write_text(marked.replace('"HUGE"', "1e400"), encoding="utf-8")
    with pytest.raises(
        SidecarError, match=r"sidecar grid_um: inf is not a finite number"
    ):
        read_sidecar(path)


def test_a_non_finite_number_is_refused_where_no_schema_describes_it(sidecars):
    """A face request's ``support`` is free-form, and still holds only finite numbers."""
    good = sidecars["pad_b_bands"]
    support = next(
        path
        for path in _paths(good)
        if path[-1] == "support" and "z_um" in _at(good, path)
    )
    with pytest.raises(
        SidecarError, match=r"support\.z_um: nan is not a finite number"
    ):
        check_sidecar(_with(good, (*support, "z_um"), float("nan")))


def _at(document, path):
    for key in path:
        document = document[key]
    return document


def _with(document, path, value):
    """A copy of ``document`` with the value at ``path`` replaced, or removed when ``value`` is ``_REMOVED``."""
    changed = copy.deepcopy(document)
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    if value is _REMOVED:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
    return changed


_REMOVED = object()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("mesh_features",),
            _REMOVED,
            r"sidecar document: lacks the field 'mesh_features'",
        ),
        (("mesh_hints",), [], r"sidecar document: has the unknown field 'mesh_hints'"),
        (("route",), "D", r"sidecar route: 'D' is not one of \['A', 'B', 'C'\]"),
        (("grid_um",), 0, r"sidecar grid_um: 0 is not greater than 0"),
        (
            ("model", "identity_digest"),
            "model:xyz",
            r"sidecar model\.identity_digest: 'model:xyz' does not match",
        ),
        (
            ("binding", "ids_by_tag", "3"),
            "vol",
            r"sidecar binding\.ids_by_tag\['3'\]: 'vol' is not array",
        ),
        (("counts", "edges"), -1, r"sidecar counts\.edges: -1 is less than 0"),
        (("counts", "later"), 2.5, r"sidecar counts\.later: 2\.5 is not integer"),
        (("files", "mesh"), "x.msh", r"sidecar files: the key 'mesh' is not one of"),
        (
            ("expected", "vertices_um", 0),
            [0.0, 0.0],
            r"sidecar expected\.vertices_um\[0\]: has 2 items, fewer than 3",
        ),
        (
            ("mesh_features", 0, "kind"),
            "curved_edge",
            r"sidecar mesh_features\[0\]\.kind: 'curved_edge' is not",
        ),
        (
            ("mesh_features", 0, "facts", "column_count"),
            3,
            r"mesh_features\[0\]\.facts: has the unknown field",
        ),
        (
            ("lineage", "membership", "f:x"),
            {"pad": -0.5},
            r"sidecar lineage\.membership\['f:x'\]\.pad: -0\.5 is less",
        ),
    ],
)
def test_a_structure_that_does_not_hold_is_refused_where_it_fails(
    sidecars, path, value, message
):
    """The first mismatch is named by its place in the document and what was expected there."""
    with pytest.raises(SidecarError, match=message):
        check_sidecar(_with(sidecars["pad_b_bands"], path, value))


def _first_group(sidecar, dim):
    return next(
        name
        for name, group in sorted(sidecar["physical_groups"].items())
        if group["dim"] == dim
    )


def test_a_reference_that_does_not_resolve_is_refused(sidecars):
    """Planned ids, native tags, groups of the right dimension, volumes, carriers and the binding's sizes are
    checked across sections."""
    good = sidecars["pad_b_bands"]
    surface = _first_group(good, 2)
    volume = _first_group(good, 3)
    other_tag = next(
        group["tag"]
        for name, group in sorted(good["physical_groups"].items())
        if group["dim"] == 2 and name != surface
    )
    emitted = next(name for name, v in sorted(good["volumes"].items()) if v["emitted"])
    bound_face = good["binding"]["ids_by_tag"]["2"][0]
    edges = good["binding"]["ids_by_tag"]["1"]
    carrier_edge = next(iter(good["lineage"]["edges"]))
    for path, value, message in (
        (
            ("physical_groups", surface, "ids", 0),
            "f:nowhere",
            r"native tag \d+ is not face 'f:nowhere'",
        ),
        (
            ("mesh_features", 0, "native_tags", 0),
            len(edges) + 1,
            r"mesh_features\[0\] \(\w+\): native tag",
        ),
        (
            ("embedded_faces",),
            [{"face": "f:nowhere", "host_volume": "v", "group": None}],
            "names face 'f:nowhere'",
        ),
        (
            ("faces", good["binding"]["ids_by_tag"]["2"][0], "group"),
            "MA::nowhere",
            "names the group 'MA::nowhere'",
        ),
        (
            ("lineage", "edges", carrier_edge),
            ["c:nowhere"],
            "names the carrier 'c:nowhere'",
        ),
        (("counts", "edges"), len(edges) + 1, "counts has"),
        (
            ("expected", "face_area_um2"),
            [1.0],
            r"expected\.face_area_um2 has 1 entries",
        ),
        (
            ("unions", "nets", "GND"),
            ["MA::nowhere"],
            r"unions\.nets\['GND'\] names the group",
        ),
        (
            ("surface_epr_rows", next(iter(good["surface_epr_rows"])), "group"),
            volume,
            f"names the group {volume!r}, which is not a face group",
        ),
        (
            ("volumes", next(iter(good["volumes"])), "group"),
            surface,
            f"names the group {surface!r}, which is not a volume group",
        ),
        (
            ("faces", bound_face),
            _REMOVED,
            rf"binding\.ids_by_tag\['2'\] names '{re.escape(bound_face)}', which faces lacks",
        ),
        (
            ("physical_groups", surface, "adjacent_volumes"),
            ["v:nowhere"],
            f"physical group '{re.escape(surface)}' names volume 'v:nowhere'",
        ),
        (
            ("volumes", emitted, "emitted"),
            False,
            rf"volumes\['{re.escape(emitted)}'\] is not emitted, which binding",
        ),
        (
            ("physical_groups", surface, "tag"),
            other_tag,
            f"has the tag {other_tag} of physical group",
        ),
    ):
        with pytest.raises(
            SidecarError, match=f"references do not resolve: .*{message}"
        ):
            check_sidecar(_with(good, path, value))


# --------------------------------------------------------------------------- the reader means what JSON Schema means


KEYWORD_CASES = [
    ({"type": "integer"}, [1, 2.0, 2.5, True, "1", None]),
    ({"type": "number"}, [1, 2.5, True, "1", None]),
    (
        {"type": ["string", "null"], "pattern": "^a[0-9]$"},
        ["a1", "a12", "xa1", None, 3],
    ),
    ({"enum": ["above", None]}, ["above", None, "below", 0]),
    ({"const": 2}, [2, 2.0, True, "2"]),
    ({"type": "number", "minimum": 0}, [0, -0.0, -1, "x"]),
    ({"type": "number", "exclusiveMinimum": 0}, [0, 0.1, -1, "x"]),
    (
        {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
            "maxItems": 2,
        },
        [[], [1], [1, 2, 3], [0], "1"],
    ),
    (
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
        },
        [{"a": "x"}, {}, {"a": 1}, {"a": "x", "b": 1}, [], None],
    ),
    (
        {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "minProperties": 1,
        },
        [{}, {"x": 1}, {"x": "1"}],
    ),
    (
        {"type": "object", "propertyNames": {"enum": ["xao", "brep"]}},
        [{"xao": "a"}, {"stl": "a"}, {}],
    ),
    (
        {"type": "object", "properties": {"a": {"type": "string"}}},
        [{"a": "x", "b": 1}, {"a": 1}, {}],
    ),
    ({"oneOf": [{"type": "string"}, {"type": "null"}]}, ["x", None, 1]),
    ({"oneOf": [{"type": "number"}, {"type": "integer"}]}, [1, 1.5, "x"]),
]


@pytest.mark.parametrize(("schema", "values"), KEYWORD_CASES)
def test_the_reader_checks_each_keyword_as_json_schema_does(schema, values):
    """For every keyword the derived schema uses, the shared reader and jsonschema give the same verdict."""
    validator = jsonschema.Draft202012Validator(schema)
    for value in values:
        assert _reader_accepts(value, schema) == validator.is_valid(value), (
            schema,
            value,
        )


def test_the_reader_and_the_schema_agree_on_every_field_of_an_emitted_sidecar(sidecars):
    """At every field position of an emitted sidecar, each probe, the value's removal and an extra key or item get
    the same verdict from the shared reader as from jsonschema with the derived schema."""
    schema = sidecar_schema()
    validator = jsonschema.Draft202012Validator(schema)
    document = _sample(sidecars["pad_b_bands"], schema, schema["$defs"])
    assert validator.is_valid(document)
    paths = list(_paths(document))
    assert len(paths) > 200
    verdicts = set()
    for path in paths:
        for mutated in _mutations(document, path):
            verdict = validator.is_valid(mutated)
            verdicts.add(verdict)
            assert _reader_accepts(mutated, schema) == verdict, path
    assert verdicts == {True, False}


def _reader_accepts(value, schema):
    try:
        require_structure(value, schema)
    except SidecarError:
        return False
    return True


def _sample(value, node, defs):
    """A copy of ``value`` with one list item and one map entry for each alternative they hold, so every field
    position keeps one value and the copy keeps the structure of the original."""
    while "$ref" in node:
        node = defs[node["$ref"].removeprefix("#/$defs/")]
    if "oneOf" in node:
        node = next(
            b
            for b in node["oneOf"]
            if jsonschema.Draft202012Validator({**b, "$defs": defs}).is_valid(value)
        )
        return _sample(value, node, defs)
    if isinstance(value, list):
        items = (
            value
            if node.get("minItems", 0) > 1
            else _one_per_alternative(value, node.get("items", {}), defs)
        )
        return [_sample(item, node.get("items", {}), defs) for item in items]
    if not isinstance(value, dict) or "additionalProperties" not in node:
        return copy.deepcopy(value)
    fields, extra = node.get("properties", {}), node["additionalProperties"]
    entries = {key: item for key, item in value.items() if key not in fields}
    if isinstance(extra, dict):
        kept = _one_per_alternative(
            list(entries.items()), extra, defs, key=lambda entry: entry[1]
        )
        entries = dict(kept)
    sampled = {
        key: _sample(value[key], fields[key], defs) for key in value if key in fields
    }
    return sampled | {key: _sample(item, extra, defs) for key, item in entries.items()}


def _one_per_alternative(items, node, defs, key=lambda item: item):
    """The first item that matches each ``oneOf`` alternative of ``node`` (the first item when it has none)."""
    while "$ref" in node:
        node = defs[node["$ref"].removeprefix("#/$defs/")]
    branches = node.get("oneOf", [node])
    kept = {}
    for item in items:
        for index, branch in enumerate(branches):
            if index not in kept and jsonschema.Draft202012Validator(
                {**branch, "$defs": defs}
            ).is_valid(key(item)):
                kept[index] = item
        if len(kept) == len(branches):
            break
    return list(kept.values())


def _paths(value, path=()):
    """The path of every value inside ``value``."""
    children = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, list)
        else ()
    )
    for key, item in children:
        yield (*path, key)
        yield from _paths(item, (*path, key))


_PROBES = (None, True, 0, -1, 0.5, "x", [], {}, [None], {"x": None})


def _mutations(document, path):
    """Each probe in place of the value at ``path``, the value removed, and an extra key or item beside it.

    The document is changed in place and restored after each yield.
    """
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    key = path[-1]
    original = parent[key]
    for probe in _PROBES:
        parent[key] = probe
        yield document
    parent[key] = original
    if isinstance(parent, dict):
        del parent[key]
        yield document
        parent[key] = original
    if isinstance(original, dict):
        original["stowaway"] = None
        yield document
        del original["stowaway"]
    if isinstance(original, list):
        original.append(None)
        yield document
        original.pop()
