"""Check static API discovery and non-writing freshness validation."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[1]
GENERATOR = STAGE / "scripts/generate_knowledge_reference.py"
NAMESPACES = (
    "aedt",
    "aedt.specs",
    "geometry",
    "palace",
    "semantics",
    "visualization",
)


@pytest.fixture
def source_tree(tmp_path):
    """Create source files whose imports fail if generation executes them."""
    root = tmp_path / "producer"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(GENERATOR, root / "scripts/generate_knowledge_reference.py")

    public_documents = sorted(
        [
            "README.md",
            "docs/reference/index.qmd",
            "docs/reference/api/analyze-epr-usage.md",
            "zh-TW/README.md",
            "zh-TW/docs/reference/api/analyze-epr-usage.md",
        ]
    )
    for relative in public_documents:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"Source: {relative}\n", encoding="utf-8")
    (root / "docs/knowledge-sources.json").write_text(
        json.dumps(
            {"public_documents": public_documents, "api_groups": _api_groups()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for namespace in NAMESPACES:
        destination = root / "src/scgsim" / namespace.replace(".", "/") / "__init__.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "raise RuntimeError('facade must not be imported')\n"
            "__all__ = ['PublicName']\n",
            encoding="utf-8",
        )

    workflow = root / "src/scgsim/aedt/epr/workflow.py"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "raise RuntimeError('workflow must not be imported')\n"
        "def analyze_epr(*, saved_solution: SavedSolution, output_dir: str | Path, "
        "geometry_workers: int | None = None) -> Any: ...\n",
        encoding="utf-8",
    )
    for relative, content in {
        "docs/reference/api/analyze-epr-usage.md": "## Required parameters\n\nMinimal call.\n",
        "zh-TW/docs/reference/api/analyze-epr-usage.md": "## 必要參數\n\n最小呼叫。\n",
    }.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return root


def test_render_reads_ast_and_authored_text_without_importing_facades(source_tree):
    """Render exports and current signature directly from Python syntax."""
    module = _load_generator(GENERATOR, "knowledge_generator")

    outputs = module.render(source_tree)

    assert set(outputs) == {
        "docs/reference/api-index.qmd",
        "docs/reference/api/analyze-epr.qmd",
        "zh-TW/docs/reference/api-index.qmd",
        "zh-TW/docs/reference/api/analyze-epr.qmd",
    }
    index = outputs["docs/reference/api-index.qmd"]
    assert "## Task: `scgsim.geometry` {#scgsim.geometry}" in index
    assert "`PublicName`" in index
    detail = outputs["docs/reference/api/analyze-epr.qmd"]
    assert "Minimal call." in detail
    assert "saved_solution: SavedSolution" in detail
    assert "geometry_workers: int | None = None" in detail
    assert "-> Any" in detail


def test_a_declared_name_no_group_lists_fails_generation(source_tree):
    """The index lists every declared name under a task, so a name added to a facade without a group fails."""
    facade = source_tree / "src/scgsim/geometry/__init__.py"
    facade.write_text(
        facade.read_text(encoding="utf-8").replace(
            "['PublicName']", "['PublicName', 'NewName']"
        ),
        encoding="utf-8",
    )
    module = _load_generator(GENERATOR, "knowledge_generator_ungrouped")

    with pytest.raises(ValueError, match=r"missing \['NewName'\]"):
        module.render(source_tree)


def test_check_mode_fails_stale_without_creating_or_rewriting_outputs(source_tree):
    """Check mode reports missing or stale pages and never writes them."""
    script = source_tree / "scripts/generate_knowledge_reference.py"
    command = [sys.executable, "-I", "-B", str(script), "--check"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    expected = source_tree / "docs/reference/api-index.qmd"
    assert result.returncode == 1
    assert "Generated reference is stale" in result.stderr
    assert not expected.exists()

    for relative, content in _render_outputs(source_tree).items():
        destination = source_tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    expected.write_text("stale\n", encoding="utf-8")
    before = {
        path.relative_to(source_tree): path.read_bytes()
        for path in source_tree.rglob("*.qmd")
    }

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 1
    assert "Generated reference is stale" in result.stderr
    assert {
        path.relative_to(source_tree): path.read_bytes()
        for path in source_tree.rglob("*.qmd")
    } == before


def test_repository_api_pages_match_the_current_sources():
    """Fail the repository suite if generated API pages are stale."""
    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-I",
        "-B",
        str(repository / "scripts/generate_knowledge_reference.py"),
        "--check",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, (
        f"API reference freshness check failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _api_groups() -> dict[str, dict[str, object]]:
    """One task group per package, listing the fixture's one declared name."""
    group = {
        "title": "Group",
        "title_zh": "群組",
        "gloss": "Gloss.",
        "gloss_zh": "說明。",
        "page": "index.qmd",
        "names": ["PublicName"],
    }
    return {
        namespace: {"title": "Task", "title_zh": "工作", "groups": [group]}
        for namespace in NAMESPACES
    }


def _render_outputs(root: Path) -> dict[str, str]:
    module = _load_generator(
        root / "scripts/generate_knowledge_reference.py",
        "knowledge_generator_fixture",
    )
    return module.render(root)


def _load_generator(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]
    return module
