"""Links from SCGSim's messages to the published documentation of the version that raised them.

The documentation site publishes one root per version (``scripts/build_docs_site.py``): a release at
``v<version>/``, built from ``main``, and the development line at ``v<release line>-dev/``, built from
``develop``, whose package version carries a ``.devN`` or ``rcN`` suffix.  A message built here therefore
points at the pages of the code the user runs, not at whatever the latest release says.

The root is looked up when a link is made, never on import: a link is an aid inside a message, so a version
the site cannot publish only leaves the link out, and the message itself is raised as it is.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

SITE_URL = "https://ncuphy-ltlab.github.io/scgsim/"
# The version forms the site publishes; ``build_docs_site.version_line`` refuses any other.
_PUBLISHED_VERSION = re.compile(r"(\d+\.\d+\.\d+)(\.dev\d+|rc\d+)?")


def documentation_root(package_version: str) -> str | None:
    """The published root of ``package_version``'s documentation, ending in ``/``; ``None`` if the site has none."""
    match = _PUBLISHED_VERSION.fullmatch(package_version)
    if match is None:
        return None
    release, prerelease = match.groups()
    slug = f"v{release}-dev" if prerelease else f"v{release}"
    return f"{SITE_URL}{slug}/"


def page_url(page: str, anchor: str) -> str | None:
    """The URL of section ``anchor`` of the English page ``page`` (a ``.qmd`` path without its suffix).

    ``None`` when the running SCGSim has no published documentation root: its version is one the site does not
    publish, or it runs from a source tree with no installed metadata.
    """
    try:
        root = documentation_root(version("scgsim"))
    except PackageNotFoundError:
        return None
    return None if root is None else f"{root}{page}.html#{anchor}"


def see_also(page: str, anchor: str) -> str:
    """The closing sentence `` See <URL>`` of a message, or nothing when there is no published page to link.

    No full stop follows the URL, so that a terminal or a copy does not take it into the link.
    """
    url = page_url(page, anchor)
    return "" if url is None else f" See {url}"
