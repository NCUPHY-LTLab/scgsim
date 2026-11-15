"""The code map's stage tables are the package folders' docstrings and their translations, and the generator
refuses a folder it cannot state.

``docs/code-map.qmd`` and its Chinese copy include the tables ``scripts/generate_code_map.py`` writes; a
docstring edited without regenerating them or without translating it again fails here, as does a new
package folder the generator does not list.
"""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = runpy.run_path(str(ROOT / "scripts" / "generate_code_map.py"))


def test_the_committed_tables_are_the_docstrings():
    """Each edition's include file is what the generator writes from the current docstrings."""
    for relative, content in GENERATOR["render"](ROOT).items():
        assert (ROOT / relative).read_text(encoding="utf-8") == content, (
            f"{relative} is stale; run python scripts/generate_code_map.py"
        )


@pytest.fixture
def package_copy(tmp_path):
    """The package's ``__init__.py`` files and the translation file, which is all the generator reads."""
    translation = tmp_path / "scripts" / "code_map_zh_tw.toml"
    translation.parent.mkdir()
    shutil.copy2(ROOT / "scripts" / "code_map_zh_tw.toml", translation)
    for init in (ROOT / "src" / "scgsim").rglob("__init__.py"):
        destination = tmp_path / init.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(init, destination)
    return tmp_path


def test_a_folder_the_generator_does_not_list_is_refused(package_copy):
    """A new package folder must be placed in the flow before the map can be written."""
    new = package_copy / "src/scgsim/palace/extra"
    new.mkdir()
    (new / "__init__.py").write_text('"""An unplaced stage."""\n')
    with pytest.raises(
        GENERATOR["CodeMapError"],
        match="src/scgsim/palace/extra: a package folder AREAS does not list",
    ):
        GENERATOR["render"](package_copy)


def test_a_stage_without_its_next_line_is_refused(package_copy):
    """A stage's row needs all three lines; the refusal names the folder and the missing line."""
    init = package_copy / "src/scgsim/palace/mesh/__init__.py"
    init.write_text(init.read_text().replace("Next: ``config``.", ""))
    with pytest.raises(
        GENERATOR["CodeMapError"],
        match=r"src/scgsim/palace/mesh/__init__.py: a stage docstring lacks its Next: line",
    ):
        GENERATOR["render"](package_copy)


def test_a_docstring_changed_without_its_translation_is_refused(package_copy):
    """The translation records the English it translates, so a changed sentence must be translated again."""
    init = package_copy / "src/scgsim/palace/reports/__init__.py"
    init.write_text(
        init.read_text().replace(
            "The result a user holds and the notebook views of it.",
            "Views of a resolved result.",
        )
    )
    with pytest.raises(
        GENERATOR["CodeMapError"],
        match=r'src/scgsim/palace/reports \(scripts/code_map_zh_tw.toml \["palace/reports"\]\): the docstring\'s summary',
    ):
        GENERATOR["render"](package_copy)


def test_a_translation_for_a_folder_that_is_gone_is_refused(package_copy):
    """An entry left behind for a removed or renamed folder is named, so the file holds no dead text."""
    translation = package_copy / "scripts" / "code_map_zh_tw.toml"
    translation.write_text(
        translation.read_text()
        + '\n["palace/gone"]\nsummary.en = "x"\nsummary.zh = "x"\n'
    )
    with pytest.raises(
        GENERATOR["CodeMapError"],
        match=r'entry \["palace/gone"\] names no listed folder',
    ):
        GENERATOR["render"](package_copy)


def test_a_folder_without_a_translation_is_refused(package_copy):
    """Every folder's row has Traditional Chinese text, so a folder with no entry is named."""
    translation = package_copy / "scripts" / "code_map_zh_tw.toml"
    text = translation.read_text()
    start = text.index('["palace/reports"]')
    translation.write_text(text[:start] + text[text.index("\n\n", start) :])
    with pytest.raises(
        GENERATOR["CodeMapError"],
        match=r'src/scgsim/palace/reports \(scripts/code_map_zh_tw.toml \["palace/reports"\]\): no translation entry',
    ):
        GENERATOR["render"](package_copy)
