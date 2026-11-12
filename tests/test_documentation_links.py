"""The documentation links SCGSim's messages carry point at the published root of the running version.

The roots must be the ones ``scripts/build_docs_site.py`` publishes, and every page and section a message links
to must exist in both language editions, or the link a user follows lands nowhere.
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

from scgsim.documentation import SITE_URL, documentation_root

ROOT = Path(__file__).resolve().parents[1]
SITE_BUILD = runpy.run_path(str(ROOT / "scripts" / "build_docs_site.py"))


@pytest.mark.parametrize(
    ("version", "root"),
    [
        ("2.0.0rc15", "v2.0.0-dev/"),
        ("2.1.0.dev3", "v2.1.0-dev/"),
        ("2.0.0", "v2.0.0/"),
        ("1.1.0", "v1.1.0/"),
    ],
)
def test_a_version_links_to_the_root_the_site_publishes_it_under(version, root):
    """A prerelease is published from ``develop`` under its release line's ``-dev`` root, a release under its own.

    The slugs repeat the site build's rule: ``develop`` publishes ``v<version_line>-dev``, and ``main``, whose
    version must be a release, publishes ``v<version>``.
    """
    line = SITE_BUILD["version_line"](version)
    published = f"v{line}-dev/" if line != version else f"v{version}/"
    assert published == root
    assert documentation_root(version) == SITE_BUILD["SITE_URL"] + root
    assert SITE_URL == SITE_BUILD["SITE_URL"]


@pytest.mark.parametrize("version", ["2.0", "2.0.0.post1", "2.0.0+local"])
def test_a_version_the_site_cannot_publish_is_refused_as_the_site_build_refuses_it(
    version,
):
    """The site build has no root for these versions, so no link is made for them either."""
    with pytest.raises(ValueError, match="unsupported documentation package version"):
        SITE_BUILD["version_line"](version)
    with pytest.raises(ValueError, match="unsupported documentation package version"):
        documentation_root(version)


LINK = re.compile(r'page_url\(\s*"(?P<page>[^"]+)",\s*"(?P<anchor>[^"]+)"\s*\)')


def linked_sections():
    """Every ``(page, anchor)`` that ``page_url`` is called with in the package source."""
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
