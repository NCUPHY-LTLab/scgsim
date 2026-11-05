"""The meshing module's package boundary: it reads the sidecar JSON and the XAO, nothing else of ours."""

from __future__ import annotations

import ast
import configparser
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

import scgsim.meshing

REPOSITORY = Path(__file__).resolve().parents[1]
FORBIDDEN = ("scgsim.geometry", "scgsim.aedt", "scgsim.palace")
CONTRACT = "importlinter:contract:meshing-reads-contracts-only"
PARENT_PROCESS_MODULES = (
    "entry",
    "errors",
    "features",
    "sizing",
    "mesher",
    "recipes",
    "recipes.feature",
    "recipes.global_size",
    "recipes.uniform_boundary",
    "findings",
    "job",
    "driver",
    "repair",
    "report",
)


def meshing_sources() -> list[Path]:
    """Every module file of the meshing package, the recipes included."""
    return sorted(Path(scgsim.meshing.__file__).parent.rglob("*.py"))


def module_imported_from(source: Path, node: ast.ImportFrom) -> str:
    """Absolute name of the module an ``import from`` names, a relative one resolved against the file's package."""
    if not node.level:
        return node.module or ""
    root = Path(scgsim.meshing.__file__).parent
    package = ["scgsim", "meshing", *source.relative_to(root).parts[:-1]]
    parent = package[: len(package) - node.level + 1]
    return ".".join([*parent, node.module] if node.module else parent)


def imported_modules(source: Path) -> set[str]:
    """Absolute names of every module a file imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = module_imported_from(source, node)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("source", meshing_sources(), ids=lambda p: p.name)
def test_no_meshing_module_imports_sgb_aedt_or_palace(source: Path):
    """No meshing module imports sgb aedt or palace."""
    offending = {
        name
        for name in imported_modules(source)
        if any(name == f or name.startswith(f"{f}.") for f in FORBIDDEN)
    }
    assert not offending


@pytest.mark.parametrize(
    "source",
    sorted((Path(scgsim.meshing.__file__).parent / "recipes").glob("[a-z]*.py")),
    ids=lambda p: p.name,
)
def test_a_recipe_imports_no_other_recipe(source: Path):
    """A recipe imports no other recipe: each strategy stands alone beside the shared sizing vocabulary."""
    siblings = {
        f"scgsim.meshing.recipes.{other.stem}"
        for other in source.parent.glob("[a-z]*.py")
        if other != source
    }
    assert not imported_modules(source) & siblings


def test_the_contract_is_declared_in_the_import_linter_configuration():
    """The contract is declared in the import linter configuration."""
    config = configparser.ConfigParser()
    config.read(REPOSITORY / ".importlinter", encoding="utf-8")
    contract = config[CONTRACT]
    assert contract["type"] == "forbidden"
    assert contract["source_modules"].split() == ["scgsim.meshing"]
    assert contract["forbidden_modules"].split() == list(FORBIDDEN)


def test_import_linter_keeps_the_contract():
    """Import linter keeps the contract."""
    executable = Path(sysconfig.get_path("scripts")) / "lint-imports"
    if not executable.is_file():
        pytest.skip("import-linter is not installed (dependency group: quality)")
    done = subprocess.run(
        [str(executable), "--contract", "meshing-reads-contracts-only"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_importing_the_package_and_its_parent_side_does_not_load_gmsh():
    """Importing the package and its parent side does not load gmsh."""
    code = (
        "import sys, scgsim.meshing\n"
        + "".join(f"import scgsim.meshing.{name}\n" for name in PARENT_PROCESS_MODULES)
        + "assert 'gmsh' not in sys.modules"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
