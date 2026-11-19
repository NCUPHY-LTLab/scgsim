"""An old section of a page whose sections went to other pages still leads to that section, in both editions.

``scripts/build_bilingual_docs.py`` writes a redirect at each such old URL that looks the fragment up first. Each
target must be a page of both editions with that anchor, or an old section link lands on a page without it.
"""

from __future__ import annotations

import json
import posixpath
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "scripts" / "build_bilingual_docs.py"))
TARGETS = sorted(
    (page, fragment, target)
    for page, sections in BUILD["SPLIT_PAGES"].items()
    for fragment, target in sections.items()
)


@pytest.mark.parametrize("edition", [".", "zh-TW"])
@pytest.mark.parametrize(("page", "fragment", "target"), TARGETS)
def test_every_old_section_leads_to_a_page_with_its_anchor(
    edition, page, fragment, target
):
    """The target is a page of the edition, and the anchor it names is on that page."""
    path, _, anchor = target.partition("#")
    source = (
        ROOT
        / edition
        / posixpath.normpath(
            posixpath.join(posixpath.dirname(page), path)
        ).removesuffix(".html")
    ).with_suffix(".qmd")
    assert source.is_file(), source
    assert not (ROOT / edition / page).with_suffix(".qmd").exists(), (
        "the old page is gone"
    )
    if anchor:
        text = source.read_text(encoding="utf-8")
        assert re.search(
            rf'\{{#{re.escape(anchor)}\}}|id="{re.escape(anchor)}"', text
        ), anchor
    else:
        assert not fragment, "only the page itself has no anchor"


def test_the_redirect_looks_the_fragment_up_and_keeps_an_unknown_one(tmp_path):
    """Each edition gets the page, its whole map, and the replacing page for the page itself."""
    for edition in (tmp_path, tmp_path / "zh-TW"):
        (edition / "docs").mkdir(parents=True)
    BUILD["write_split_page_redirects"](tmp_path)
    for page, sections in BUILD["SPLIT_PAGES"].items():
        for edition in (tmp_path, tmp_path / "zh-TW"):
            html = (edition / page).read_text(encoding="utf-8")
            routes = json.loads(re.search(r"var redirects = (\{.*?\});", html, re.S)[1])
            assert routes == sections
            assert f'content="0;url={sections[""]}"' in html
            assert 'redirects[""] + location.search + location.hash' in html
