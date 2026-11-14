"""The documentation links SCGSim's messages carry point at the published root of the running version.

The roots must be the ones ``scripts/build_docs_site.py`` publishes, the package's own version must have one,
and every page and section a message links to must exist in both language editions, or the link a user follows
lands nowhere.
"""

from __future__ import annotations

import re
import runpy
from importlib.metadata import version
from pathlib import Path

import pytest

from scgsim.documentation import SITE_URL, documentation_root

ROOT = Path(__file__).resolve().parents[1]
SITE_BUILD = runpy.run_path(str(ROOT / "scripts" / "build_docs_site.py"))


def published_root(package_version: str) -> str:
    """The root the site build publishes ``package_version`` under, by its own rule.

    ``develop`` publishes ``v<version_line>-dev``, and ``main``, whose version must be a release, publishes
    ``v<version>``; ``version_line`` refuses a version the site cannot publish.
    """
    line = SITE_BUILD["version_line"](package_version)
    slug = f"v{line}-dev" if line != package_version else f"v{package_version}"
    return SITE_BUILD["SITE_URL"] + slug + "/"


def test_the_package_version_has_a_published_root():
    """A version bump the site cannot publish fails here, as the documentation build would fail on it."""
    assert documentation_root(version("scgsim")) == published_root(version("scgsim"))


@pytest.mark.parametrize(
    ("package_version", "root"),
    [
        ("3.0.0rc1", "v3.0.0-dev/"),
        ("2.0.0rc15", "v2.0.0-dev/"),
        ("2.1.0.dev3", "v2.1.0-dev/"),
        ("2.0.0", "v2.0.0/"),
        ("1.1.0", "v1.1.0/"),
    ],
)
def test_a_version_links_to_the_root_the_site_publishes_it_under(package_version, root):
    """A prerelease is published from ``develop`` under its release line's ``-dev`` root, a release under its own."""
    assert published_root(package_version) == SITE_URL + root
    assert documentation_root(package_version) == SITE_URL + root
    assert SITE_URL == SITE_BUILD["SITE_URL"]


@pytest.mark.parametrize(
    "package_version", ["2.0", "2.1.0a1", "2.0.0.post1", "2.0.0+local"]
)
def test_a_version_the_site_cannot_publish_has_no_root(package_version):
    """The site build refuses these versions, so they have no root and no link is made for them."""
    with pytest.raises(ValueError, match="unsupported documentation package version"):
        SITE_BUILD["version_line"](package_version)
    assert documentation_root(package_version) is None


LINK = re.compile(r'see_also\(\s*"(?P<page>[^"]+)",\s*"(?P<anchor>[^"]+)"\s*\)')


def linked_sections():
    """Every ``(page, anchor)`` that ``see_also`` is called with in the package source."""
    found = set()
    for path in (ROOT / "src").rglob("*.py"):
        found.update((m["page"], m["anchor"]) for m in LINK.finditer(path.read_text()))
    return sorted(found)


def test_messages_link_to_sections_somewhere():
    """The package carries at least the two pinch links, so the next test checks something."""
    assert {
        ("docs/known-geometry-problems", "boundary-point-pinch"),
        ("docs/known-geometry-problems", "pinch-inside-the-layout"),
    } <= set(linked_sections())


@pytest.mark.parametrize("edition", [".", "zh-TW"])
@pytest.mark.parametrize(("page", "anchor"), linked_sections())
def test_every_linked_section_exists_in_both_editions(edition, page, anchor):
    """The page renders to ``<page>.html`` and carries the section's ``{#anchor}`` in each language."""
    source = ROOT / edition / f"{page}.qmd"
    assert source.is_file(), source
    assert re.search(rf"\{{#{re.escape(anchor)}\}}", source.read_text()), anchor
