"""The code map's stage tables are the package folders' docstrings, and the generator refuses a folder it cannot state.

``docs/code-map.qmd`` includes the tables ``scripts/generate_code_map.py`` writes; a docstring edited without
regenerating them fails here, as does a new package folder the generator does not list.
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
    """The package's ``__init__.py`` files alone, which is all the generator reads."""
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
