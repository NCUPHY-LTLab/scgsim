"""Links from SCGSim's messages to the published documentation of the version that raised them.

The documentation site publishes one root per version (``scripts/build_docs_site.py``): a release at
``v<version>/``, built from ``main``, and the development line at ``v<release line>-dev/``, built from
``develop``, whose package version carries a ``.devN`` or ``rcN`` suffix.  A message built here therefore
points at the pages of the code the user runs, not at whatever the latest release says.
"""

from __future__ import annotations

import re
from importlib.metadata import version

SITE_URL = "https://ncuphy-ltlab.github.io/scgsim/"
# The version forms the site publishes; ``build_docs_site.version_line`` refuses any other.
_PUBLISHED_VERSION = re.compile(r"(\d+\.\d+\.\d+)(\.dev\d+|rc\d+)?")


def documentation_root(package_version: str) -> str:
    """The published root of ``package_version``'s documentation, ending in ``/``.

    Raises ``ValueError`` for a version the site cannot publish, as the site build does.
    """
    match = _PUBLISHED_VERSION.fullmatch(package_version)
    if match is None:
        raise ValueError(
            f"unsupported documentation package version: {package_version!r}"
        )
    release, prerelease = match.groups()
    slug = f"v{release}-dev" if prerelease else f"v{release}"
    return f"{SITE_URL}{slug}/"


# Resolved once, so a version the site cannot publish fails on import, not while a refusal is being written.
_ROOT = documentation_root(version("scgsim"))


def page_url(page: str, anchor: str) -> str:
    """The URL of section ``anchor`` of the English page ``page`` (a ``.qmd`` path without its suffix)."""
    return f"{_ROOT}{page}.html#{anchor}"
