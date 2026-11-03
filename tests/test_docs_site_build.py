"""Exercise real publication files with explicitly doubled network and render commands."""

from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEVELOP, STABLE, HISTORY, PRESENTATION = (letter * 40 for letter in "abcd")


def _source_files(version, label, *, historical=False):
    config = f"""project:
  type: website
  source-label: {label}
execute:
  enabled: false
askr:
  languages:
    current: en
    items:
    - code: en
      href: /
    - code: zh-TW
      href: /zh-TW/
website:
  title: {label}
  repo-url: https://github.com/NCUPHY-LTLab/scgsim
  repo-branch: old
  repo-actions:
  - source
  - issue
  navbar:
    - text: Tutorial
      href: docs/current.qmd
  sidebar:
    - id: tutorial
      contents:
      - docs/current.qmd
      - docs/legacy.qmd
format:
  askr-html:
    label: {label}
"""
    return {
        "pyproject.toml": f'[project]\nversion = "{version}"\n',
        "_quarto.yml": config,
        "zh-TW/_quarto.yml": config,
        "index.qmd": f"Original {label} home body.\n",
        "zh-TW/index.qmd": f"Original {label} Chinese body.\n",
        "docs/legacy.qmd": f"Original {label} legacy body.\n",
        **(
            {}
            if historical
            else {"docs/current.qmd": f"Original {label} current body.\n"}
        ),
        "docs/styles.css": f"original {label} consumer stylesheet",
        "_extensions/arfiligol/askr/_extension.yml": "version: 0.13.3\n",
        "_extensions/arfiligol/askr/theme.txt": f"theme from {label}",
    }


def _archive(files, commit):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, value in files.items():
            data = value.encode()
            info = tarfile.TarInfo(f"scgsim-{commit}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


@dataclass
class _BuildHarness:
    """Own real temporary inputs and observations of doubled external operations."""

    root: Path
    script: Path
    sources: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    branches: dict = field(default_factory=lambda: {"develop": DEVELOP, "main": STABLE})
    render_failure: int | None = None
    missing_home: bool = False
    quarto_version: str = "1.10.18"
    renders: int = 0

    def source_export(self, files):
        """Write a real clean export for preview identity and inventory checks."""
        destination = self.root / "export"
        for name, value in files.items():
            path = destination / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(value)
        return destination

    def open_archive(self, url, *, timeout):
        """Double only the network boundary with genuine tar archive responses."""
        commit = url.rsplit("/", 1)[1]
        assert url == f"https://codeload.github.com/NCUPHY-LTLab/scgsim/tar.gz/{commit}"
        assert timeout == 60
        self.calls.append(["archive", commit])
        return io.BytesIO(_archive(self.sources[commit], commit))

    def run_command(self, args, **kwargs):
        """Double git lookup and Quarto rendering while writing observable output files."""
        assert kwargs["check"] is True
        if args[:3] == ["git", "ls-remote", "--heads"]:
            self.calls.append(["heads", *args])
            stdout = "".join(
                f"{sha}\trefs/heads/{name}\n" for name, sha in self.branches.items()
            )
        elif args == ["quarto-test", "--version"]:
            self.calls.append(["quarto-version"])
            stdout = self.quarto_version + "\n"
        else:
            assert args[:2] == ["quarto-test", "render"]
            assert args[3:] == ["--profile", "pages", "--no-execute"]
            source = Path(args[2])
            self.calls.append(["render", source.name, *args[3:]])
            index = self.renders
            self.renders += 1
            if index == self.render_failure:
                raise subprocess.CalledProcessError(19, ["quarto-test", "render"])
            rendered = source / "_site"
            rendered.mkdir()
            if not self.missing_home:
                (rendered / "index.html").write_text("Doubled Quarto home output.\n")
            observed = {
                str(path.relative_to(source)): path.read_text()
                for path in sorted(source.rglob("*"))
                if path.is_file() and "_site" not in path.parts
            }
            (rendered / "render-observation.json").write_text(
                json.dumps(observed, sort_keys=True)
            )
            stdout = ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    def current_commit(self, args, **kwargs):
        """Double the one local Git identity lookup without changing the checkout."""
        assert args == ["git", "rev-parse", "HEAD"]
        assert kwargs == {"cwd": self.script.resolve().parents[1], "text": True}
        self.calls.append(["presentation-head"])
        return PRESENTATION + "\n"

    def invoke(self, *extra):
        """Run the actual CLI and serialize observations outside its operation catch."""
        output = self.root / "publication"
        sys.argv = [
            str(self.script),
            "--output",
            str(output),
            "--catalog",
            str(self.root / "catalog.json"),
            "--quarto",
            "quarto-test",
            *map(str, extra),
        ]
        caught = None
        try:
            runpy.run_path(str(self.script), run_name="__main__")
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.CalledProcessError,
        ) as error:
            caught = error
            outcome = {
                "error": type(error).__name__,
                "message": str(error).replace(str(self.root), "<case>"),
            }
        else:
            outcome = {"status": "success"}
        files = {
            path.relative_to(output).as_posix(): path.read_text()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        if destination := os.environ.get("DOCS_SITE_EVIDENCE"):
            evidence = Path(destination)
            evidence.mkdir(exist_ok=True, parents=True)
            value = {
                "doubles": "Network responses and Quarto/git commands only; archive extraction, config binding, file copies and final rename are real.",
                "outcome": outcome,
                "calls": self.calls,
                "files": files,
            }
            (evidence / (self.root.name + ".json")).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
        if caught is not None:
            raise caught
        return output


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    """Use real source files with explicit network and render-command test doubles."""
    script = Path(
        os.environ.get("DOCS_SITE_SCRIPT", ROOT / "scripts/build_docs_site.py")
    )
    harness = _BuildHarness(tmp_path, script)
    harness.sources = {
        DEVELOP: _source_files("2.0.0rc15", "development"),
        STABLE: _source_files("1.1.0", "stable"),
        HISTORY: _source_files("1.0.0", "historical", historical=True),
        PRESENTATION: _source_files("2.0.0rc15", "presentation"),
    }
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "development_ref": "develop",
                "stable_ref": "main",
                "historical": [
                    {"version": "1.1.0", "commit": STABLE},
                    {"version": "1.0.0", "commit": HISTORY},
                ],
            }
        )
    )
    monkeypatch.setattr(urllib.request, "urlopen", harness.open_archive)
    monkeypatch.setattr(subprocess, "run", harness.run_command)
    monkeypatch.setattr(subprocess, "check_output", harness.current_commit)
    monkeypatch.setattr(sys, "argv", [])
    rename = os.rename

    def observed_rename(source, destination):
        harness.calls.append(["rename", Path(source).name, Path(destination).name])
        return rename(source, destination)

    monkeypatch.setattr(os, "rename", observed_rename)
    return harness


def test_versions_overlay_only_presentation_and_publish_after_all_renders(publisher):
    """Retain immutable bodies and catalog order while overlaying historical menus."""
    output = publisher.invoke()
    build = json.loads((output / "build.json").read_text())
    assert build["latest_stable"] == "/scgsim/v1.1.0/"
    assert [entry["branches"] for entry in build["versions"]] == [
        ["develop"],
        ["main", "historical"],
        ["historical"],
    ]
    assert build["preview"] is False
    assert [call[1] for call in publisher.calls if call[0] == "archive"] == [
        DEVELOP,
        PRESENTATION,
        STABLE,
        HISTORY,
    ]
    assert publisher.calls[-1] == ["rename", "site", "publication"]
    assert publisher.renders == 3
    historical = json.loads((output / "v1.0.0/render-observation.json").read_text())
    assert historical["docs/legacy.qmd"] == "Original historical legacy body.\n"
    assert "source-label: historical" in historical["_quarto.yml"]
    assert "label: presentation" in historical["_quarto.yml"]
    assert "href: docs/legacy.qmd" in historical["_quarto.yml"]
    assert "docs/current.qmd" not in historical["_quarto.yml"]
    assert "docs/styles.css" not in historical
    assert (
        historical["_extensions/arfiligol/askr/theme.txt"] == "theme from presentation"
    )
    profile = json.loads(historical["_quarto-pages.yml"])
    assert profile["execute"] == {"enabled": False}
    assert profile["website"]["repo-branch"] == HISTORY
    assert "/scgsim/v1.0.0/zh-TW/" in historical["zh-TW/_quarto.yml"]
    for name in ("index.html", "main/index.html"):
        assert 'content="0;url=/scgsim/v1.1.0/"' in (output / name).read_text()
    assert "/scgsim/v2.0.0-dev/" in (output / "develop/index.html").read_text()
    assert "location.search + location.hash" in (output / "404.html").read_text()
    for version in ("v2.0.0-dev", "v1.1.0", "v1.0.0"):
        menu = json.loads((output / version / "askr-versions.json").read_text())
        assert menu["versions"][1] == {
            "label": "1.1.0",
            "root": "/scgsim/v1.1.0/",
            "latest": True,
        }


@pytest.mark.parametrize("label", ["Research", "研究"])
def test_new_area_is_absent_from_versions_without_its_pages(publisher, label):
    """Adding an Area must not block publication of unchanged older content."""
    presentation = publisher.sources[PRESENTATION]
    config = presentation["_quarto.yml"]
    config = config.replace(
        "  sidebar:\n",
        f"    - text: {label}\n      href: docs/research/index.qmd\n  sidebar:\n",
    ).replace(
        "format:\n",
        "    - id: research\n      contents:\n      - docs/research/index.qmd\nformat:\n",
    )
    presentation["_quarto.yml"] = config
    presentation["docs/research/index.qmd"] = "Current research overview.\n"

    output = publisher.invoke()
    historical = json.loads((output / "v1.0.0/render-observation.json").read_text())
    assert f"text: {label}" not in historical["_quarto.yml"]
    assert "id: research" not in historical["_quarto.yml"]
    assert "href: docs/legacy.qmd" in historical["_quarto.yml"]
    assert historical["docs/legacy.qmd"] == "Original historical legacy body.\n"


def test_unknown_missing_menu_destination_still_fails(publisher):
    """Omitting an empty known Area must not conceal an unrelated broken link."""
    presentation = publisher.sources[PRESENTATION]
    presentation["_quarto.yml"] = presentation["_quarto.yml"].replace(
        "  sidebar:\n",
        "    - text: Unmapped\n      href: docs/missing.qmd\n  sidebar:\n",
    )
    with pytest.raises(RuntimeError, match="no page for the Unmapped Area"):
        publisher.invoke()


def test_preview_binds_inventory_without_published_commit_attribution(publisher):
    """A clean export has its own content and presentation inventory identity."""
    source = publisher.source_export(_source_files("2.0.0rc15", "candidate"))
    inventory = {
        path.relative_to(source).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = publisher.invoke(
        "--candidate-source", source, "--presentation-commit", PRESENTATION
    )
    manifest = json.loads((output / "build.json").read_text())
    assert manifest["preview"] is True
    assert manifest["presentation_export_sha256"] == digest
    candidate = manifest["versions"][0]
    assert candidate["candidate_inventory"] == inventory
    assert candidate["candidate_export_sha256"] == digest
    assert "content_commit" not in candidate
    assert "presentation_commit" not in manifest
    assert [call[1] for call in publisher.calls if call[0] == "archive"] == [
        STABLE,
        HISTORY,
    ]
    observed = json.loads((output / "v2.0.0-dev/render-observation.json").read_text())
    assert "repo-actions: [issue]" in observed["_quarto.yml"]
    assert "repo-branch" not in json.loads(observed["_quarto-pages.yml"])["website"]
    assert not (source / "_quarto-pages.yml").exists()


@pytest.mark.parametrize("failed_render", (0, 1, 2))
def test_render_failure_never_exposes_partial_publication(publisher, failed_render):
    """A failed required version leaves no combined output or staging directory."""
    publisher.render_failure = failed_render
    with pytest.raises(subprocess.CalledProcessError):
        publisher.invoke()
    assert not (publisher.root / "publication").exists()
    assert not list(publisher.root.glob("scgsim-pages-*"))
    assert all(call[0] != "rename" for call in publisher.calls)


@pytest.mark.parametrize("first_error", ("quarto", "catalog", "output"))
def test_tool_catalog_output_failure_priority(publisher, first_error):
    """Tool version, catalog decoding and existing output fail in the original order."""
    (publisher.root / "publication").mkdir()
    if first_error != "output":
        (publisher.root / "catalog.json").write_text("invalid json")
    if first_error == "quarto":
        publisher.quarto_version = "0"
    expected = {
        "quarto": RuntimeError,
        "catalog": json.JSONDecodeError,
        "output": FileExistsError,
    }
    with pytest.raises(expected[first_error]):
        publisher.invoke()
    assert not any(call[0] == "archive" for call in publisher.calls)


@pytest.mark.parametrize("forbidden", (".git", "_site", ".quarto", "symlink"))
def test_candidate_must_be_a_clean_export(publisher, forbidden):
    """Preview identity rejects checkout, render and symbolic-link source exports."""
    source = publisher.source_export(_source_files("2.0.0rc15", "candidate"))
    if forbidden == "symlink":
        (source / "linked").symlink_to(source / "index.qmd")
    else:
        (source / forbidden).mkdir()
    with pytest.raises(ValueError, match="candidate"):
        publisher.invoke("--candidate-source", source)
    assert not any(call[0] == "archive" for call in publisher.calls)


@pytest.mark.parametrize("failure", ("prerelease", "history-version", "root-conflict"))
def test_invalid_version_sources_fail_before_any_render(publisher, failure):
    """Source version mismatches and conflicting roots fail before rendering starts."""
    if failure == "prerelease":
        publisher.sources[STABLE] = _source_files("1.1.0rc1", "stable")
        message = "current stable source has prerelease metadata"
    elif failure == "history-version":
        publisher.sources[HISTORY] = _source_files(
            "1.0.1", "historical", historical=True
        )
        message = "historical content version differs from the catalogue"
    else:
        policy = json.loads((publisher.root / "catalog.json").read_text())
        policy["historical"][-1]["version"] = "1.1.0"
        (publisher.root / "catalog.json").write_text(json.dumps(policy))
        publisher.sources[HISTORY] = _source_files(
            "1.1.0", "historical", historical=True
        )
        message = "one version root has conflicting content commits"
    with pytest.raises(ValueError, match=message):
        publisher.invoke()
    assert publisher.renders == 0
    assert not (publisher.root / "publication").exists()


def test_missing_home_page_is_a_render_failure(publisher):
    """A zero command exit without its required home page never publishes output."""
    publisher.missing_home = True
    with pytest.raises(RuntimeError, match="render did not produce its home page"):
        publisher.invoke()
    assert publisher.renders == 1
    assert not (publisher.root / "publication").exists()


@pytest.mark.parametrize("profile", ("_quarto-pages.yml", "zh-TW/_quarto-pages.yml"))
def test_source_owned_pages_profile_remains_an_error(publisher, profile):
    """Publisher profiles must not replace an existing source-owned profile."""
    source = publisher.source_export(
        {**_source_files("2.0.0rc15", "candidate"), profile: "original profile\n"}
    )
    with pytest.raises(
        RuntimeError, match="source already owns the Pages profile name"
    ):
        publisher.invoke("--candidate-source", source)
    assert publisher.renders == 0
    assert not (publisher.root / "publication").exists()


def test_build_metadata_records_verified_askr_commit(publisher):
    """Publication metadata names the separately verified Askr 0.13.3 commit."""
    output = publisher.invoke()
    assert (
        json.loads((output / "build.json").read_text())["askr_commit"]
        == "243d3dbdd234004bbe0538b9518fb76027903915"
    )
