"""Render immutable documentation sources into one versioned Pages publication.

Published content comes from frozen Git archives. An explicit clean export is
supported for unpublished development previews, with its own inventory identity.
Historical bodies remain original; only publisher-owned presentation is overlaid.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict, cast

REPOSITORY = "NCUPHY-LTLab/scgsim"
REPO_URL = f"https://github.com/{REPOSITORY}"
SITE_URL = "https://ncuphy-ltlab.github.io/scgsim/"
PREFIX = "/scgsim/"
QUARTO_VERSION = "1.10.18"
ASKR_COMMIT = "243d3dbdd234004bbe0538b9518fb76027903915"


@dataclass(frozen=True)
class SourceSelection:
    """Choose an unpublished development export or immutable presentation source."""

    candidate_source: Path | None = None
    presentation_commit: str | None = None


class _HistoricalVersion(TypedDict):
    version: str
    commit: str


class _VersionPolicy(TypedDict):
    development_ref: str
    stable_ref: str
    historical: list[_HistoricalVersion]


class _SourceIdentity(TypedDict, total=False):
    content_commit: str
    candidate_export_sha256: str
    candidate_inventory: dict[str, str]
    presentation_commit: str
    presentation_export_sha256: str


class _CandidateExport(TypedDict):
    candidate_export_sha256: str


class _PresentationExport(TypedDict):
    presentation_export_sha256: str


class _VersionEntry(_SourceIdentity):
    slug: str
    root: str
    package_version: str
    label: str
    branches: list[str]
    askr_version: NotRequired[str]


class _MenuEntry(TypedDict):
    label: str
    root: str
    latest: NotRequired[bool]


class _VersionManifest(TypedDict):
    mode: str
    versions: list[_MenuEntry]


_WebsiteSettings = TypedDict(
    "_WebsiteSettings",
    {"site-url": str, "page-footer": dict[str, str], "repo-branch": NotRequired[str]},
)


def build(
    branches: dict[str, str],
    output: Path,
    quarto: str = "quarto",
    *,
    catalog: Path | None = None,
    source_selection: SourceSelection | None = None,
) -> None:
    """Stage every immutable version before exposing the combined publication."""
    _check_quarto_version(quarto)
    source_selection = source_selection or SourceSelection()
    catalog = catalog or Path(__file__).with_name("pages_versions.json")
    policy = cast(_VersionPolicy, json.loads(catalog.read_text()))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Presentation has an explicit immutable authority; preview uses its exact export.
    if (
        source_selection.candidate_source is None
        and source_selection.presentation_commit is None
    ):
        source_selection = SourceSelection(
            presentation_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            ).strip()
        )
    with tempfile.TemporaryDirectory(
        prefix="scgsim-pages-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        sources = _SourceCatalog(branches, policy, work, source_selection)
        sources.collect()
        site = _stage_publication(sources, work, quarto)
        output.parent.mkdir(parents=True, exist_ok=True)
        # No combined output is exposed until every required render completes.
        os.rename(site, output)


def _check_quarto_version(quarto: str) -> None:
    """Require the pinned renderer before reading policy or staging any sources."""
    actual = subprocess.run(
        [quarto, "--version"], check=True, capture_output=True, text=True
    )
    if actual.stdout.strip() != QUARTO_VERSION:
        raise RuntimeError(f"Pages requires Quarto {QUARTO_VERSION}")


def resolve_branches() -> dict[str, str]:
    """Capture both branch heads in one remote lookup."""
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--heads",
            f"{REPO_URL}.git",
            "refs/heads/develop",
            "refs/heads/main",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    branches = {}
    for line in result.stdout.splitlines():
        sha, ref = line.split()
        if ref not in {"refs/heads/develop", "refs/heads/main"} or not re.fullmatch(
            r"[0-9a-f]{40}", sha
        ):
            raise ValueError("remote returned an invalid commit identity")
        branches[ref.removeprefix("refs/heads/")] = sha
    if "develop" not in branches:
        raise RuntimeError("remote develop branch is missing")
    return branches


def snapshot(sha: str, directory: Path) -> Path:
    """Download and unpack the exact archived source identity."""
    directory.mkdir()
    archive = directory / "source.tar.gz"
    url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{sha}"
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        archive.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream)
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    archive.unlink()
    (root,) = directory.iterdir()
    return root


def export_inventory(source: Path) -> dict[str, str]:
    """Bind an explicit export without attributing it to a published commit."""
    if (
        (source / ".git").exists()
        or (source / "_site").exists()
        or (source / ".quarto").exists()
    ):
        raise ValueError(
            "candidate must be a clean source export, not a checkout or render"
        )
    rows = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"candidate export contains a symbolic link: {path}")
        if path.is_file():
            rows[path.relative_to(source).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return rows


def package_version(source: Path) -> str:
    """Read the package version from its original source metadata."""
    return tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]


def version_line(version: str) -> str:
    """Resolve the release line used by development version roots."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:\.dev\d+|rc\d+)?", version)
    if match is None:
        raise ValueError(f"unsupported documentation package version: {version!r}")
    return match[1]


def presentation_blocks(config: str) -> dict[str, str]:
    """Select the two existing top-level Quarto presentation mappings."""
    starts = list(re.finditer(r"(?m)^([A-Za-z][\w-]*):", config))
    blocks = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(config)
        blocks[match[1]] = config[match.start() : end]
    return {name: blocks[name] for name in ("website", "format")}


def historical_website(website: str, source: Path) -> str:
    """Apply the publisher menu only to pages owned by this historical source."""
    # Area landings follow the first available entry in that same current sidebar.
    # The publisher configuration remains the sole maintained navigation authority.
    sidebars = list(re.finditer(r"(?m)^[ ]+- id: ([^\n]+)\n", website))
    landings = {}
    empty_areas = []
    for index, match in enumerate(sidebars):
        end = sidebars[index + 1].start() if index + 1 < len(sidebars) else len(website)
        pages = re.findall(r"(?m)^\s+- (\S+\.qmd)\s*$", website[match.end() : end])
        destination = next((page for page in pages if (source / page).is_file()), None)
        landings[match[1]] = destination
        if destination is None:
            empty_areas.append((match.start(), end))
        for page in pages:
            landings[page] = destination

    # New Areas have no content in older releases; keep them out of those menus.
    for start, end in reversed(empty_areas):
        website = website[:start] + website[end:]

    def landing(match: re.Match[str]) -> str:
        area, page = match["label"].strip("\"' "), match["page"]
        if (source / page).is_file():
            return match[0]
        replacement = landings.get(page, landings.get(area.lower()))
        if replacement is None:
            if page in landings or area.lower() in landings:
                return ""
            raise RuntimeError(f"historical source has no page for the {area} Area")
        return match[0].replace(page, replacement)

    website = re.sub(
        r"(?m)^(?P<indent>[ ]+)- text: (?P<label>[^\n]+)\n(?P=indent)  href: (?P<page>\S+\.qmd)[ ]*$",
        landing,
        website,
    )
    return "".join(
        line
        for line in website.splitlines(keepends=True)
        if not (match := re.fullmatch(r"\s+- (\S+\.qmd)\s*", line))
        or (source / match[1]).is_file()
    )


def overlay_presentation(source: Path, presentation: Path) -> None:
    """Replace publisher-owned navigation and theme while retaining historical bodies."""
    current = presentation_blocks((presentation / "_quarto.yml").read_text())
    current["website"] = historical_website(current["website"], source)
    original = (source / "_quarto.yml").read_text()
    blocks = presentation_blocks(original)
    for name in ("website", "format"):
        original = original.replace(blocks[name], current[name], 1)
    (source / "_quarto.yml").write_text(original)
    # Native Askr presentation supersedes the historical consumer stylesheet.
    (source / "docs/styles.css").unlink(missing_ok=True)
    vendor = source / "_extensions/arfiligol/askr"
    if vendor.exists():
        shutil.rmtree(vendor)
    shutil.copytree(presentation / "_extensions/arfiligol/askr", vendor)


def redirect_page(destination: str) -> str:
    """Build a redirect that retains the browser query and fragment."""
    safe = html.escape(destination, quote=True)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>SCGSim documentation</title><meta http-equiv="refresh" content="0;url={safe}">'
        f'</head><body><p>Open <a href="{safe}">SCGSim documentation</a>.</p>'
        "<script>location.replace("
        + json.dumps(destination)
        + "+location.search+location.hash);</script></body></html>"
    )


def missing_page(roots: list[str], aliases: dict[str, str], latest: str) -> str:
    """Route known aliases and version roots without inventing missing pages."""
    routes = json.dumps({"roots": roots, "aliases": aliases, "latest": latest}).replace(
        "<", "\\u003c"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>SCGSim documentation page not found</title></head><body><main>"
        "<h1>Documentation page not found</h1>"
        '<p id="reason">This path is not present in the published documentation.</p>'
        f'<p><a href="{html.escape(latest, quote=True)}">Open latest stable documentation</a></p>'
        "</main><script>const routes="
        + routes
        + ";"
        + r"""
const path = location.pathname;
function within(root) { return path === root.slice(0, -1) || path.startsWith(root); }
const alias = Object.keys(routes.aliases).find(within);
if (alias) {
  const rest = path === alias.slice(0, -1) ? '' : path.slice(alias.length);
  location.replace(routes.aliases[alias] + rest + location.search + location.hash);
} else {
  const root = routes.roots.find(within);
  if (root) location.replace(root + location.search + location.hash);
}
</script></body></html>"""
    )


def display_identity(value: str) -> str:
    """Keep complete visible identities while allowing native HTML line wrapping."""
    return "<wbr>".join(
        html.escape(value[index : index + 16]) for index in range(0, len(value), 16)
    )


@dataclass
class _SourceCatalog:
    """Carry acquired sources, identities and ordered version roots together."""

    branches: dict[str, str]
    policy: _VersionPolicy
    work: Path
    selection: SourceSelection
    entries: list[_VersionEntry] = field(default_factory=list, init=False)
    sources: dict[str, Path] = field(default_factory=dict, init=False)
    presentation: Path = field(init=False)
    presentation_identity: _SourceIdentity = field(init=False)

    def collect(self) -> None:
        """Acquire development, stable and historical versions in catalog order."""
        development, identity = self.prepare_development()
        self.add(
            development,
            package_version(development),
            identity,
            self.policy["development_ref"],
        )
        self.include_stable()
        self.include_historical()

    def prepare_development(self) -> tuple[Path, _SourceIdentity]:
        """Bind either the clean preview inventory or immutable source archives."""
        candidate_source = self.selection.candidate_source
        if candidate_source is not None:
            rows = export_inventory(candidate_source)
            digest = hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            development = self.work / "candidate"
            shutil.copytree(candidate_source, development)
            self.presentation = development
            self.presentation_identity = {"presentation_export_sha256": digest}
            development_identity: _SourceIdentity = {
                "candidate_export_sha256": digest,
                "candidate_inventory": rows,
            }
        else:
            development = snapshot(
                self.branches[self.policy["development_ref"]], self.work / "development"
            )
            presentation_commit = cast(str, self.selection.presentation_commit)
            self.presentation = snapshot(
                presentation_commit, self.work / "presentation"
            )
            self.presentation_identity = {"presentation_commit": presentation_commit}
            development_identity = {
                "content_commit": self.branches[self.policy["development_ref"]]
            }
        return development, development_identity

    def add(
        self, source: Path, version: str, identity: _SourceIdentity, branch: str
    ) -> None:
        """Deduplicate matching roots while retaining every source branch label."""
        prerelease = branch == self.policy["development_ref"]
        slug = "v" + version_line(version) + "-dev" if prerelease else "v" + version
        existing = next((item for item in self.entries if item["slug"] == slug), None)
        if existing:
            if existing.get("content_commit") != identity.get("content_commit"):
                raise ValueError("one version root has conflicting content commits")
            existing["branches"].append(branch)
            return
        entry: _VersionEntry = {
            "slug": slug,
            "root": PREFIX + slug + "/",
            "package_version": version,
            "label": version_line(version) + " dev" if prerelease else version,
            "branches": [branch],
            **identity,
        }
        self.entries.append(entry)
        self.sources[slug] = source

    def include_stable(self) -> None:
        """Include the current stable branch only when its source is a release."""
        stable_ref = self.policy["stable_ref"]
        if stable_ref in self.branches:
            stable = snapshot(self.branches[stable_ref], self.work / "stable")
            version = package_version(stable)
            if version != version_line(version):
                raise ValueError("current stable source has prerelease metadata")
            self.add(
                stable,
                version,
                {"content_commit": self.branches[stable_ref]},
                stable_ref,
            )

    def include_historical(self) -> None:
        """Check pinned versions against original bodies without downloading duplicates."""
        for pinned in self.policy["historical"]:
            existing = next(
                (
                    item
                    for item in self.entries
                    if item.get("content_commit") == pinned["commit"]
                    and item["package_version"] == pinned["version"]
                ),
                None,
            )
            if existing:
                existing["branches"].append("historical")
                continue
            historical = snapshot(
                pinned["commit"], self.work / ("history-" + pinned["version"])
            )
            if package_version(historical) != pinned["version"]:
                raise ValueError(
                    "historical content version differs from the catalogue"
                )
            self.add(
                historical,
                pinned["version"],
                {"content_commit": pinned["commit"]},
                "historical",
            )

    def latest_stable(self) -> _VersionEntry:
        """Select the latest numeric release after all required sources are present."""
        stable_entries = [
            item
            for item in self.entries
            if self.policy["development_ref"] not in item["branches"]
        ]
        return max(
            stable_entries,
            key=lambda item: tuple(map(int, item["package_version"].split("."))),
        )


def _stage_publication(sources: _SourceCatalog, work: Path, quarto: str) -> Path:
    """Render the ordered catalog and write publication routes into one staging tree."""
    latest = sources.latest_stable()
    aliases = {
        PREFIX + "develop/": sources.entries[0]["root"],
        PREFIX + "main/": latest["root"],
    }
    manifest: _VersionManifest = {
        "mode": "inline",
        "versions": [
            cast(
                _MenuEntry,
                {
                    "label": item["label"],
                    "root": item["root"],
                    **({"latest": True} if item is latest else {}),
                },
            )
            for item in sources.entries
        ],
    }
    site = work / "site"
    site.mkdir()
    _render_versions(sources, site, manifest, quarto)
    _write_routes(site, sources.entries, latest, aliases)
    _write_build_manifest(site, sources, latest)
    return site


def _render_versions(
    sources: _SourceCatalog, site: Path, manifest: _VersionManifest, quarto: str
) -> None:
    """Bind each original source to its presentation and copy its completed render."""
    for entry in sources.entries:
        source = sources.sources[entry["slug"]]
        actual_presentation: _SourceIdentity
        if "historical" in entry["branches"]:
            overlay_presentation(source, sources.presentation)
            actual_presentation = sources.presentation_identity
        elif "content_commit" in entry:
            actual_presentation = {"presentation_commit": entry["content_commit"]}
        else:
            actual_presentation = {
                "presentation_export_sha256": cast(_CandidateExport, entry)[
                    "candidate_export_sha256"
                ]
            }
        cast(dict[str, object], entry).update(actual_presentation)
        render_source(source, entry, actual_presentation, quarto)
        destination = site / entry["slug"]
        shutil.copytree(source / "_site", destination)
        (destination / "askr-versions.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


def _write_routes(
    site: Path,
    entries: list[_VersionEntry],
    latest: _VersionEntry,
    aliases: dict[str, str],
) -> None:
    """Write the stable landing page, branch aliases and version-aware missing page."""
    (site / "index.html").write_text(redirect_page(latest["root"]))
    for alias, destination in aliases.items():
        target = site / alias.removeprefix(PREFIX)
        target.mkdir()
        (target / "index.html").write_text(redirect_page(destination))
    (site / "404.html").write_text(
        missing_page([item["root"] for item in entries], aliases, latest["root"])
    )


def _write_build_manifest(
    site: Path, sources: _SourceCatalog, latest: _VersionEntry
) -> None:
    """Record source identities and preview status beside the complete publication."""
    (site / "build.json").write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "quarto_version": QUARTO_VERSION,
                "askr_commit": ASKR_COMMIT,
                **sources.presentation_identity,
                "branches": sources.branches,
                "versions": sources.entries,
                "latest_stable": latest["root"],
                "preview": sources.selection.candidate_source is not None,
            },
            indent=2,
        )
        + "\n"
    )


def render_source(
    source: Path,
    entry: _VersionEntry,
    presentation_identity: _SourceIdentity,
    quarto: str,
) -> None:
    """Bind source identity, write Pages profiles and require a rendered home page."""
    website = _render_website(source, entry, presentation_identity)
    _write_render_profiles(source, website, entry.get("content_commit"), entry["root"])
    subprocess.run(
        [quarto, "render", str(source), "--profile", "pages", "--no-execute"],
        check=True,
    )
    if not (source / "_site/index.html").is_file():
        raise RuntimeError("render did not produce its home page")


def _render_website(
    source: Path, entry: _VersionEntry, presentation_identity: _SourceIdentity
) -> _WebsiteSettings:
    """Bind original content and presentation identities to the source theme metadata."""
    root = entry["root"]

    content = entry.get("content_commit")

    if content:
        attribution = f'content <a href="{REPO_URL}/tree/{content}">{display_identity(content)}</a>'
    else:
        attribution = "unpublished candidate · export " + display_identity(
            cast(_CandidateExport, entry)["candidate_export_sha256"]
        )

    presentation = presentation_identity.get("presentation_commit")

    if presentation:
        display = f'presentation <a href="{REPO_URL}/tree/{presentation}">{display_identity(presentation)}</a>'
    else:
        display = "unpublished presentation · export " + display_identity(
            cast(_PresentationExport, presentation_identity)[
                "presentation_export_sha256"
            ]
        )

    extension = (source / "_extensions/arfiligol/askr/_extension.yml").read_text()

    match = re.search(r"(?m)^version:\s*([^\s]+)\s*$", extension)

    if match is None:
        raise RuntimeError("source Askr version is missing")

    theme_version = match[1]

    entry["askr_version"] = theme_version

    website: _WebsiteSettings = {
        "site-url": "https://ncuphy-ltlab.github.io" + root,
        "page-footer": {
            "center": f"SCGSim {entry['package_version']} · {attribution} · {display} · Askr {theme_version}"
        },
    }

    if content:
        website["repo-branch"] = content

    return website


def _write_render_profiles(
    source: Path, website: _WebsiteSettings, content: str | None, root: str
) -> None:
    """Configure both language projects without replacing source-owned profiles."""
    _bind_project(source / "_quarto.yml", root, content)

    profile = source / "_quarto-pages.yml"

    if profile.exists():
        raise RuntimeError("source already owns the Pages profile name")

    profile_data = {
        "project": {"output-dir": "_site"},
        "execute": {"enabled": False},
        "website": website,
    }

    chinese = source / "zh-TW"

    if (chinese / "_quarto.yml").is_file():
        _bind_project(chinese / "_quarto.yml", root, content)
        child_profile = chinese / "_quarto-pages.yml"
        if child_profile.exists():
            raise RuntimeError("Chinese source already owns the Pages profile name")
        child_profile.write_text(
            json.dumps(
                {
                    "project": {"output-dir": "../_site/zh-TW"},
                    "execute": {"enabled": False},
                    "website": {**website, "site-url": website["site-url"] + "zh-TW/"},
                }
            )
        )

    profile.write_text(json.dumps(profile_data))


def _bind_project(config_path: Path, root: str, content: str | None) -> None:
    # Bind arrays in the base mapping: profile arrays append in Quarto.
    """Bind array settings in the base project because Quarto profiles append arrays."""
    config = config_path.read_text()
    config = re.sub(
        r"(?m)^  repo-actions:[^\n]*(?:\n[ ]{2,4}-[^\n]*)*\n?",
        "  repo-actions: [source, issue]\n" if content else "  repo-actions: [issue]\n",
        config,
    )
    if content:
        config = re.sub(r"(?m)^  repo-branch:.*\n", "", config)
        config = re.sub(
            r"(?m)^(  repo-url:.*)$", r"\1\n  repo-branch: " + content, config
        )

    def language_roots(block: re.Match[str]) -> str:
        return re.sub(
            r"(?m)^([ ]+href:[ ]*)/(zh-TW/)?[ ]*$",
            lambda link: link[1] + root + (link[2] or ""),
            block[0],
        )

    config = re.sub(r"(?ms)^askr:\n.*?(?=^[A-Za-z][\w-]*:|\Z)", language_roots, config)
    config_path.write_text(config)


def main() -> None:
    """Resolve remote heads and run the requested immutable publication build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--quarto", default="quarto")
    parser.add_argument(
        "--candidate-source",
        type=Path,
        help="explicit clean exported development preview, never published-commit attribution",
    )
    parser.add_argument("--presentation-commit")
    arguments = parser.parse_args()
    build(
        resolve_branches(),
        arguments.output,
        arguments.quarto,
        catalog=arguments.catalog,
        source_selection=SourceSelection(
            arguments.candidate_source, arguments.presentation_commit
        ),
    )


if __name__ == "__main__":
    main()
