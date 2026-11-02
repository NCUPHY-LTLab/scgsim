"""Generate installed offline knowledge from authored sources before building.

README.md and Quarto pages own content and producer metadata. This build-only
adapter writes package data without importing SCGSim or exposing a runtime API.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from setuptools import build_meta as _setuptools
from setuptools.build_meta import *  # noqa: F403 (forward remaining PEP 517 hooks)

_PAGE_FIELDS = {
    "scgsim-id": "id",
    "scgsim-kind": "kind",
    "title": "title",
    "description": "summary",
    "scgsim-status": "status",
}


class _ProjectMetadata(TypedDict):
    name: str
    version: str
    description: str


@dataclass(frozen=True)
class _ResourceMetadata:
    identifier: str
    kind: str
    title: str
    summary: str
    status: str


class _ManifestResource(TypedDict):
    id: str
    kind: str
    title: str
    summary: str
    status: str
    path: str
    content_sha256: str
    provenance: dict[str, str]


@dataclass(frozen=True)
class _Resource:
    source_ref: str
    output_path: str
    content: bytes
    metadata: _ResourceMetadata

    def manifest_entry(self) -> _ManifestResource:
        """Describe one source and its unchanged installed content."""
        digest = hashlib.sha256(self.content).hexdigest()
        return {
            "id": self.metadata.identifier,
            "kind": self.metadata.kind,
            "title": self.metadata.title,
            "summary": self.metadata.summary,
            "status": self.metadata.status,
            "path": self.output_path,
            "content_sha256": digest,
            "provenance": {"source_ref": self.source_ref, "source_sha256": digest},
        }


def generate_knowledge() -> None:
    """Collect sources, construct the manifest, validate targets, then write data."""
    root = Path(__file__).resolve().parent
    project: _ProjectMetadata = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    readme = _read_readme(root, project)
    outputs = {readme.output_path: readme.content}
    resources = [readme.manifest_entry()]
    _collect_documentation(root, outputs, resources)
    outputs["manifest.json"] = _manifest_bytes(project, resources)
    bundle = root / "src/scgsim/_agent_knowledge"
    _validate_destination(bundle, outputs)
    _write_bundle(bundle, outputs)


def _read_readme(root: Path, project: _ProjectMetadata) -> _Resource:
    source_ref = "README.md"
    content = _source_bytes(root / source_ref, source_ref)
    title = ""
    for line in content.decode("utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        raise ValueError(
            f"Knowledge README is missing a Markdown H1 title: {source_ref}"
        )
    return _Resource(
        source_ref=source_ref,
        output_path=source_ref,
        content=content,
        metadata=_ResourceMetadata(
            identifier="readme",
            kind="project",
            title=title,
            summary=project["description"],
            status="CONVERGING",
        ),
    )


def _collect_documentation(
    root: Path, outputs: dict[str, bytes], resources: list[_ManifestResource]
) -> None:
    docs = root / "docs"
    if docs.resolve() != docs or not docs.is_dir():
        raise ValueError(
            "Knowledge source directory must be a regular non-symlink directory: docs"
        )
    pages = sorted(
        docs.rglob("*.qmd"), key=lambda path: path.relative_to(root).as_posix()
    )
    for source in pages:
        source_ref = source.relative_to(root).as_posix()
        content = _source_bytes(source, source_ref)
        frontmatter = _frontmatter(content, source_ref, tuple(_PAGE_FIELDS))
        metadata = {
            manifest_field: frontmatter[field]
            for field, manifest_field in _PAGE_FIELDS.items()
        }
        resource = _Resource(
            source_ref=source_ref,
            output_path=source_ref,
            content=content,
            metadata=_ResourceMetadata(
                identifier=metadata["id"],
                kind=metadata["kind"],
                title=metadata["title"],
                summary=metadata["summary"],
                status=metadata["status"],
            ),
        )
        outputs[resource.output_path] = resource.content
        resources.append(resource.manifest_entry())

    # Public figures are auxiliary files, not Catalog text resources.
    assets = sorted(
        (source for source in (docs / "assets").rglob("*") if source.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for source in assets:
        source_ref = source.relative_to(root).as_posix()
        outputs[source_ref] = _source_bytes(source, source_ref)


def _manifest_bytes(
    project: _ProjectMetadata, resources: list[_ManifestResource]
) -> bytes:
    identifiers = [resource["id"] for resource in resources]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Knowledge resource IDs must be unique")
    start_resources = [
        resource["id"] for resource in resources if resource["kind"] == "start"
    ]
    if start_resources != ["docs/examples"]:
        raise ValueError("Knowledge needs docs/examples as its sole start resource")
    resources.sort(key=lambda resource: resource["id"])
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "distribution": project["name"],
        "distribution_version": project["version"],
        "resources": resources,
    }
    manifest["bundle_id"] = "sha256:" + hashlib.sha256(_canonical(manifest)).hexdigest()
    return _canonical(manifest) + b"\n"


def _validate_destination(bundle: Path, outputs: Mapping[str, bytes]) -> None:
    for relative in outputs:
        target = bundle / relative
        if target.resolve() != target:
            raise ValueError("Knowledge output must not traverse a symlink")
    if bundle.exists():
        unexpected = {
            item.relative_to(bundle).as_posix()
            for item in bundle.rglob("*")
            if item.is_symlink()
            or (
                not item.is_dir() and item.relative_to(bundle).as_posix() not in outputs
            )
        }
        if unexpected:
            raise ValueError(
                "Unexpected files in generated knowledge directory; "
                "preserve and inspect them before rebuilding"
            )


def _write_bundle(bundle: Path, outputs: Mapping[str, bytes]) -> None:
    for relative, content in outputs.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _frontmatter(
    content: bytes, source_ref: str, required: tuple[str, ...]
) -> dict[str, str]:
    text = content.decode("utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"Knowledge source needs YAML front matter: {source_ref}")

    values: dict[str, str] = {}
    closed = False
    for line in lines[1:]:
        if line == "---":
            closed = True
            break
        key, separator, raw_value = line.partition(":")
        if not separator or key not in required:
            continue
        if key in values:
            raise ValueError(
                f"Knowledge source repeats front matter field {key!r}: {source_ref}"
            )
        value = json.loads(raw_value.strip())
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Knowledge source field {key!r} must be a nonempty string: {source_ref}"
            )
        values[key] = value

    if not closed:
        raise ValueError(
            f"Knowledge source has unterminated YAML front matter: {source_ref}"
        )
    missing = set(required).difference(values)
    if missing:
        raise ValueError(
            f"Knowledge source is missing front matter fields {sorted(missing)!r}: {source_ref}"
        )
    return values


def _source_bytes(source: Path, source_ref: str) -> bytes:
    if source.resolve() != source or not source.is_file():
        raise ValueError(
            f"Knowledge source must be a regular non-symlink file: {source_ref}"
        )
    return source.read_bytes()


def build_sdist(sdist_directory, config_settings=None):
    """Generate knowledge, then delegate the sdist build to setuptools."""
    generate_knowledge()
    return _setuptools.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Generate knowledge, then delegate the wheel build to setuptools."""
    generate_knowledge()
    return _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory)
