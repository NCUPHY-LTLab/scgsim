"""The AEDT runtime source is a generated import closure checked by shape."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scgsim.aedt.results.provenance import (
    ENTRY_MODULES,
    SOURCE_SCHEMA,
    record_runtime_source,
    validate_runtime_source,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_ROOT,
    current_runtime_source_paths,
    runtime_import_closure,
)


def _write_tree(source_root: Path, sources: dict[str, str]) -> None:
    for relative, source in sources.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _aedt_tree(entry_source: str) -> dict[str, str]:
    """A minimal source tree with one AEDT module and both visualization roots."""
    return {
        "scgsim/__init__.py": "",
        "scgsim/aedt/__init__.py": "",
        "scgsim/aedt/entry.py": entry_source,
        "scgsim/shared/__init__.py": "",
        "scgsim/shared/added.py": "",
        "scgsim/unused.py": "",
        "scgsim/visualization/__init__.py": "",
        "scgsim/visualization/loaders/__init__.py": "",
        "scgsim/visualization/loaders/aedt.py": "",
        "scgsim/visualization/loaders/aedt_receipt.py": "",
    }


def test_current_list_is_a_sorted_unique_closure_with_the_entry_modules():
    """The list on this checkout is sorted, unique, contained and complete."""
    paths = current_runtime_source_paths()
    assert list(paths) == sorted(set(paths))
    assert all(path.startswith("src/scgsim/") for path in paths)
    assert set(ENTRY_MODULES) <= set(paths)
    aedt_modules = {
        f"src/{path.relative_to(SOURCE_ROOT).as_posix()}"
        for path in (SOURCE_ROOT / "scgsim/aedt").rglob("*.py")
    }
    assert aedt_modules <= set(paths)


def test_the_closure_holds_no_palace_or_meshing_file():
    """AEDT provenance names AEDT's own code: a Palace-only change must not alter it.

    The import-linter independence contract covers ``scgsim.aedt`` and ``scgsim.palace`` only, so an import
    of Palace from a module both use (the visualization loaders, ``semantics``, ``presentation``, the
    contracts) passes it; this checks the closure itself.
    """
    backend = [
        path
        for path in current_runtime_source_paths()
        if path.startswith(("src/scgsim/palace/", "src/scgsim/meshing/"))
    ]
    assert backend == []


def test_adding_an_import_to_an_aedt_module_adds_its_target(tmp_path):
    """The list follows the code: no roster has to be edited."""
    source_root = tmp_path / "src"
    _write_tree(source_root, _aedt_tree(""))
    before = current_runtime_source_paths(source_root)
    assert "src/scgsim/shared/added.py" not in before
    assert "src/scgsim/unused.py" not in before

    _write_tree(source_root, {"scgsim/aedt/entry.py": "from ..shared import added\n"})
    after = current_runtime_source_paths(source_root)
    assert set(after) - set(before) == {
        "src/scgsim/shared/__init__.py",
        "src/scgsim/shared/added.py",
    }


def test_deleting_a_file_outside_the_closure_changes_nothing(tmp_path):
    """A deletion elsewhere in src no longer breaks AEDT preparation."""
    source_root = tmp_path / "src"
    _write_tree(source_root, _aedt_tree(""))
    before = current_runtime_source_paths(source_root)
    (source_root / "scgsim/unused.py").unlink()
    assert current_runtime_source_paths(source_root) == before


def test_import_closure_follows_package_relative_absolute_child_and_lazy_imports(
    tmp_path,
):
    """Resolve direct, relative, package-child, initializer, and function imports."""
    source_root = tmp_path / "src"
    sources = {
        "scgsim/__init__.py": "",
        "scgsim/aedt/__init__.py": "from . import initialized_child\n",
        "scgsim/aedt/initialized_child.py": "",
        "scgsim/aedt/entry.py": (
            "from . import relative_child\n"
            "from scgsim.shared import imported_child\n"
            "import scgsim.shared.absolute_child\n"
            "def lazy_step():\n"
            "    from ..shared.lazy_child import LazyValue\n"
        ),
        "scgsim/aedt/relative_child.py": "",
        "scgsim/shared/__init__.py": "",
        "scgsim/shared/imported_child.py": "",
        "scgsim/shared/absolute_child.py": "",
        "scgsim/shared/lazy_child.py": "",
    }
    _write_tree(source_root, sources)

    paths = runtime_import_closure(source_root, ("src/scgsim/aedt/entry.py",))
    assert paths == tuple(f"src/{path}" for path in sorted(sources))


@pytest.mark.parametrize("stage", ["prepared", "actual"])
def test_record_of_the_current_source_is_accepted(stage):
    """Both stages record the computed list under the current schema."""
    record = record_runtime_source(stage)
    validate_runtime_source(record, stage=stage)
    assert record["schema_version"] == SOURCE_SCHEMA
    assert SOURCE_SCHEMA == "scgsim.aedt.runtime-source.v30"
    assert record["stage"] == stage
    assert (
        tuple(item["path"] for item in record["modules"])
        == current_runtime_source_paths()
    )


def _resealed(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute the digest so a test isolates the one property it breaks."""
    encoded = json.dumps(
        record["modules"], ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    record["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def _module(path: str) -> dict[str, str]:
    module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return {"module": module, "path": path, "sha256": "0" * 64}


def _with_module(record: dict[str, Any], path: str) -> dict[str, Any]:
    record["modules"] = sorted(
        [*record["modules"], _module(path)], key=lambda item: item["path"]
    )
    return _resealed(record)


def _outside_src(record: dict[str, Any]) -> dict[str, Any]:
    return _with_module(record, "tests/helper.py")


def _escaping_src(record: dict[str, Any]) -> dict[str, Any]:
    return _with_module(record, "src/scgsim/../../outside.py")


def _unsorted(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"].reverse()
    return _resealed(record)


def _duplicated(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"].insert(0, copy.deepcopy(record["modules"][0]))
    return _resealed(record)


def _without_entry(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"] = [
        item for item in record["modules"] if item["path"] != ENTRY_MODULES[1]
    ]
    return _resealed(record)


def _wrong_digest(record: dict[str, Any]) -> dict[str, Any]:
    record["modules"][0]["sha256"] = "f" * 64
    return record


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_outside_src, "module manifest is invalid"),
        (_escaping_src, "module manifest is invalid"),
        (_unsorted, "paths are not sorted and unique"),
        (_duplicated, "paths are not sorted and unique"),
        (_without_entry, "lacks the AEDT entry modules"),
        (_wrong_digest, "content digest is invalid"),
    ],
)
def test_a_damaged_record_is_refused(damage, message):
    """Each shape property and the digest are checked on their own."""
    record = damage(record_runtime_source("actual"))
    with pytest.raises(RuntimeError, match=message):
        validate_runtime_source(record, stage="actual")


def test_a_record_of_the_other_stage_is_refused():
    """A prepared record is not accepted where execution's record belongs."""
    with pytest.raises(RuntimeError, match="actual runtime source provenance"):
        validate_runtime_source(record_runtime_source("prepared"), stage="actual")


@pytest.mark.parametrize("schema", ["v1", "v12", "v25", "v29"])
def test_an_older_schema_is_refused_with_how_to_read_it(schema):
    """Old result folders are read with the commit that wrote them."""
    record = record_runtime_source("prepared")
    record["schema_version"] = f"scgsim.aedt.runtime-source.{schema}"
    with pytest.raises(
        RuntimeError,
        match=(
            rf"runtime-source\.{schema} was written by an earlier SCGSim.*"
            "read this result with the SCGSim commit that wrote it"
        ),
    ):
        validate_runtime_source(record, stage="prepared")


@pytest.mark.parametrize("schema", ["scgsim.aedt.runtime-source.v31", "other", None])
def test_an_unknown_schema_is_refused(schema):
    """A schema that is neither current nor older is invalid, not 'earlier'."""
    record = record_runtime_source("prepared")
    record["schema_version"] = schema
    with pytest.raises(
        RuntimeError, match="prepared runtime source provenance is invalid"
    ):
        validate_runtime_source(record, stage="prepared")
