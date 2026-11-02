"""Check source generation and failure boundaries with real public source files."""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from setuptools import build_meta

import build_backend


@pytest.fixture
def source_tree(tmp_path):
    """Copy public inputs and the real backend into an isolated producer tree."""
    repository = Path(build_backend.__file__).parent
    root = tmp_path / "producer"
    root.mkdir()
    for name in ("README.md", "pyproject.toml", "build_backend.py"):
        shutil.copy2(repository / name, root / name)
    (root / "docs").mkdir()
    shutil.copy2(repository / "docs/architecture.qmd", root / "docs/architecture.qmd")
    asset = next(
        path
        for path in sorted((repository / "docs/assets").rglob("*"))
        if path.is_file()
    )
    destination = root / asset.relative_to(repository)
    destination.parent.mkdir(parents=True)
    shutil.copy2(asset, destination)
    bundle = root / "src/scgsim/_agent_knowledge"
    bundle.mkdir(parents=True)
    (bundle / "README.md").write_bytes(b"existing installed readme\n")
    return root


@pytest.fixture
def backend(source_tree):
    """Load the copied PEP 517 implementation with its own source location."""
    module_name = (
        "build_backend_" + hashlib.sha256(str(source_tree).encode()).hexdigest()
    )
    spec = importlib.util.spec_from_file_location(
        module_name, source_tree / "build_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[module_name]


def _bundle_state(root):
    bundle = root / "src/scgsim/_agent_knowledge"
    return {
        path.relative_to(bundle).as_posix(): (
            ("symlink", path.readlink())
            if path.is_symlink()
            else ("file", path.read_bytes())
        )
        for path in bundle.rglob("*")
        if path.is_symlink() or path.is_file()
    }


def test_generate_preserves_source_bytes_and_skips_unchanged_outputs(
    source_tree, backend
):
    """Install real README, page and asset bytes and retain unchanged file times."""
    backend.generate_knowledge()
    bundle = source_tree / "src/scgsim/_agent_knowledge"
    for source in (
        source_tree / "README.md",
        *sorted((source_tree / "docs").rglob("*")),
    ):
        if source.is_file():
            assert (
                bundle / source.relative_to(source_tree)
            ).read_bytes() == source.read_bytes()
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    identifiers = [resource["id"] for resource in manifest["resources"]]
    assert identifiers == sorted(identifiers)
    assert set(identifiers) == {"readme", "docs/architecture"}
    for resource in manifest["resources"]:
        digest = hashlib.sha256((bundle / resource["path"]).read_bytes()).hexdigest()
        assert resource["content_sha256"] == digest
        assert resource["provenance"]["source_sha256"] == digest
    before = {
        path: path.stat().st_mtime_ns for path in bundle.rglob("*") if path.is_file()
    }
    backend.generate_knowledge()
    assert before == {path: path.stat().st_mtime_ns for path in before}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"No front matter\n", "needs YAML front matter"),
        (b'---\ntitle: "Title"\n', "unterminated YAML front matter"),
        (b'---\ntitle: "First"\ntitle: "Second"\n---\n', "repeats front matter field"),
        (b"---\ntitle: 7\n---\n", "must be a nonempty string"),
        (b'---\ntitle: " "\n---\n', "must be a nonempty string"),
        (b'---\ntitle: "Title"\n---\n', "missing front matter fields"),
        (b"---\ntitle: broken-json\n---\n", "Expecting value"),
    ],
)
def test_invalid_page_fails_before_any_output_changes(
    source_tree, backend, content, message
):
    """Preserve installed outputs when authored page metadata is invalid."""
    (source_tree / "docs/architecture.qmd").write_bytes(content)
    before = _bundle_state(source_tree)
    with pytest.raises(ValueError, match=message):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before


def test_readme_error_precedes_missing_project_description(source_tree, backend):
    """Read and validate the README before consuming its project summary."""
    (source_tree / "README.md").write_text("README without a Markdown title\n")
    project = source_tree / "pyproject.toml"
    project.write_text(
        "\n".join(
            line
            for line in project.read_text().splitlines()
            if not line.startswith("description =")
        )
    )
    before = _bundle_state(source_tree)
    with pytest.raises(ValueError, match="missing a Markdown H1 title"):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before


def test_duplicate_ids_precede_project_name_access(source_tree, backend):
    """Reject duplicate resources before reading manifest distribution fields."""
    page = source_tree / "docs/architecture.qmd"
    page.write_text(
        page.read_text().replace(
            'scgsim-id: "docs/architecture"', 'scgsim-id: "readme"'
        )
    )
    project = source_tree / "pyproject.toml"
    project.write_text(
        "\n".join(
            line
            for line in project.read_text().splitlines()
            if not line.startswith("name =")
        )
    )
    before = _bundle_state(source_tree)
    with pytest.raises(ValueError, match="resource IDs must be unique"):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before


@pytest.mark.parametrize(
    "source_name", ["README.md", "docs/architecture.qmd", "asset", "docs"]
)
def test_symlink_sources_are_rejected_before_writing(source_tree, backend, source_name):
    """Keep source symlinks from becoming installed knowledge content."""
    if source_name == "asset":
        source = next(
            path for path in (source_tree / "docs/assets").rglob("*") if path.is_file()
        )
    else:
        source = source_tree / source_name
    outside = source_tree.parent / "external-source"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=outside.is_dir())
    before = _bundle_state(source_tree)
    with pytest.raises(ValueError, match="regular non-symlink"):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before


@pytest.mark.parametrize("source_name", ["README.md", "docs"])
@pytest.mark.parametrize("replacement", ["missing", "wrong_kind"])
def test_required_source_paths_keep_their_file_and_directory_contracts(
    source_tree, backend, source_name, replacement
):
    """Reject missing inputs and file/directory substitutions before writing."""
    source = source_tree / source_name
    if source.is_dir():
        shutil.rmtree(source)
        if replacement == "wrong_kind":
            source.write_text("not a source directory")
    else:
        source.unlink()
        if replacement == "wrong_kind":
            source.mkdir()
    before = _bundle_state(source_tree)
    with pytest.raises(ValueError, match="regular non-symlink"):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before


def test_failed_validation_does_not_create_a_bundle(source_tree, backend):
    """Keep a fresh destination absent when source validation fails."""
    bundle = source_tree / "src/scgsim/_agent_knowledge"
    shutil.rmtree(bundle)
    (source_tree / "README.md").write_text("missing title")
    with pytest.raises(ValueError, match="missing a Markdown H1 title"):
        backend.generate_knowledge()
    assert not bundle.exists()


@pytest.mark.parametrize("unexpected_kind", ["file", "symlink"])
def test_unexpected_output_is_preserved_before_writing(
    source_tree, backend, unexpected_kind
):
    """Preserve unknown installed files and links for inspection."""
    unexpected = source_tree / "src/scgsim/_agent_knowledge/unexpected"
    if unexpected_kind == "file":
        unexpected.write_bytes(b"keep this file")
    else:
        unexpected.symlink_to(source_tree / "docs", target_is_directory=True)
    before = _bundle_state(source_tree)
    with pytest.raises(
        ValueError, match="Unexpected files in generated knowledge directory"
    ):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before


def test_output_symlink_precedes_unexpected_file_detection(source_tree, backend):
    """Reject redirected targets before checking the installed file inventory."""
    bundle = source_tree / "src/scgsim/_agent_knowledge"
    outside = source_tree.parent / "external-output"
    outside.write_bytes(b"keep external output")
    (bundle / "README.md").unlink()
    (bundle / "README.md").symlink_to(outside)
    (bundle / "unexpected").write_bytes(b"keep unexpected")
    before = _bundle_state(source_tree)
    with pytest.raises(ValueError, match="must not traverse a symlink"):
        backend.generate_knowledge()
    assert _bundle_state(source_tree) == before
    assert outside.read_bytes() == b"keep external output"


def test_other_pep517_hooks_retain_setuptools_identity(backend):
    """Forward metadata, requirement and editable hooks without substitute logic."""
    for name in (
        "get_requires_for_build_sdist",
        "get_requires_for_build_wheel",
        "prepare_metadata_for_build_wheel",
        "build_editable",
        "get_requires_for_build_editable",
        "prepare_metadata_for_build_editable",
    ):
        assert getattr(backend, name) is getattr(build_meta, name)
