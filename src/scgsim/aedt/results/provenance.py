"""Record the runtime source in a run receipt, and encode and check the receipt's current schema."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Literal, cast

from scgsim.aedt.io import file_sha256
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_ROOT,
    current_runtime_source_paths,
)

_RECEIPT_SCHEMA_PREFIX = "scgsim.aedt.receipt.v"
_RECEIPT_SCHEMA_NUMBER = 3

RECEIPT_V3 = f"{_RECEIPT_SCHEMA_PREFIX}{_RECEIPT_SCHEMA_NUMBER}"
"""The receipt schema every handoff and completed run is written with."""

_SOURCE_SCHEMA_PREFIX = "scgsim.aedt.runtime-source.v"
_SOURCE_SCHEMA_NUMBER = 30

SOURCE_SCHEMA = f"{_SOURCE_SCHEMA_PREFIX}{_SOURCE_SCHEMA_NUMBER}"
"""The runtime-source schema: a generated import closure checked by shape."""

ENTRY_MODULES = ("src/scgsim/__init__.py", "src/scgsim/aedt/__init__.py")
"""Paths every AEDT runtime source contains.

Every AEDT import passes through both package initializers, so a list without
either is not an AEDT closure; as initializers they stay put when modules move
inside ``scgsim/aedt``.
"""

_RECORD_KEYS = {
    "schema_version",
    "stage",
    "modules",
    "content_sha256",
    "revision_observation",
}


def record_runtime_source(stage: Literal["prepared", "actual"]) -> dict[str, Any]:
    """Hash the current runtime source into one record for the given stage.

    ``prepared`` is recorded when a handoff is written and ``actual`` when
    execution starts; the two are kept apart because they may differ.
    """
    modules = _module_manifest()
    revision = _observed_revision()
    return {
        "schema_version": SOURCE_SCHEMA,
        "stage": stage,
        "modules": modules,
        "content_sha256": _content_digest(modules),
        "revision_observation": (
            {"status": "available", "revision": revision}
            if revision is not None
            else {"status": "unavailable"}
        ),
    }


def _module_manifest() -> list[dict[str, str]]:
    return [
        {
            "module": _module_name(path),
            "path": path,
            "sha256": file_sha256(SOURCE_ROOT / path.removeprefix("src/")),
        }
        for path in current_runtime_source_paths()
    ]


def _module_name(path: str) -> str:
    return path.removeprefix("src/").removesuffix(".py").replace("/", ".")


def _content_digest(modules: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        modules, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observed_revision() -> str | None:
    try:
        direct_url = distribution("scgsim").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    revision = ""
    if direct_url is not None:
        try:
            revision = str(
                json.loads(direct_url).get("vcs_info", {}).get("commit_id", "")
            )
        except (TypeError, ValueError):
            revision = ""
    if not revision:
        source_root = next(
            (
                parent
                for parent in Path(__file__).resolve().parents
                if (parent / ".git").exists() and (parent / "pyproject.toml").is_file()
            ),
            None,
        )
        if source_root is not None:
            try:
                completed = subprocess.run(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                revision = ""
            else:
                revision = completed.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None


def validate_runtime_source(
    value: Any, *, stage: Literal["prepared", "actual"]
) -> None:
    """Check a recorded runtime source by schema, shape and digest.

    The recorded hashes are not compared with installed code: a receipt states
    which bytes ran, and later code may differ.
    """
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    _validate_schema(value.get("schema_version"), stage)
    if set(value) != _RECORD_KEYS or value.get("stage") != stage:
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    modules = value["modules"]
    _validate_module_records(modules, stage)
    if value["content_sha256"] != _content_digest(modules):
        raise RuntimeError(f"{stage} runtime source content digest is invalid")
    _validate_revision_observation(value["revision_observation"], stage=stage)


def _validate_schema(schema: Any, stage: str) -> None:
    """Refuse every schema but the current one, naming how to read an older one."""
    if schema == SOURCE_SCHEMA:
        return
    if _is_earlier_source_schema(schema):
        raise RuntimeError(
            f"{stage} runtime source {schema} was written by an earlier SCGSim, "
            f"which this version does not read (it reads {SOURCE_SCHEMA}); read "
            "this result with the SCGSim commit that wrote it, named by the "
            "receipt's revision_observation when Git was available"
        )
    raise RuntimeError(f"{stage} runtime source provenance is invalid")


def _validate_module_records(modules: Any, stage: str) -> None:
    """Check that the records name a sorted, unique AEDT closure under src/scgsim."""
    if not isinstance(modules, list):
        raise RuntimeError(f"{stage} runtime source module manifest is invalid")
    paths: list[str] = []
    for item in modules:
        path = item.get("path") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != {"module", "path", "sha256"}
            or not isinstance(path, str)
            or not path.startswith("src/scgsim/")
            or ".." in Path(path).parts
            or item["module"] != _module_name(path)
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise RuntimeError(f"{stage} runtime source module manifest is invalid")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise RuntimeError(f"{stage} runtime source paths are not sorted and unique")
    missing = [path for path in ENTRY_MODULES if path not in paths]
    if missing:
        raise RuntimeError(
            f"{stage} runtime source lacks the AEDT entry modules {missing}"
        )


def _validate_revision_observation(value: Any, *, stage: str) -> None:
    """Unavailable VCS provenance is distinct from missing measured code bytes."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage} runtime source revision observation is invalid")
    if value == {"status": "unavailable"}:
        return
    if (
        set(value) != {"status", "revision"}
        or value.get("status") != "available"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("revision", "")))
    ):
        raise RuntimeError(f"{stage} runtime source revision observation is invalid")


def require_current_receipt(receipt: Any) -> None:
    """Refuse a receipt an earlier SCGSim wrote before reading anything it binds.

    The receipt schema and the schema of each runtime-source record it carries
    are checked first, so an old result is refused with how to read it rather
    than by whichever older request or result format it reaches next.
    """
    schema = receipt.get("schema_version") if isinstance(receipt, dict) else None
    if schema != RECEIPT_V3:
        number = (
            schema.removeprefix(_RECEIPT_SCHEMA_PREFIX)
            if isinstance(schema, str) and schema.startswith(_RECEIPT_SCHEMA_PREFIX)
            else ""
        )
        if number.isdigit() and int(number) < _RECEIPT_SCHEMA_NUMBER:
            raise RuntimeError(
                f"receipt {schema} was written by an earlier SCGSim, which this "
                f"version does not read (it reads {RECEIPT_V3}); read this result "
                "with the SCGSim commit that wrote it"
            )
        raise RuntimeError("handoff receipt schema is invalid")
    for key, stage in (
        ("prepared_runtime_source", "prepared"),
        ("runtime_source", "actual"),
    ):
        record = receipt.get(key)
        if isinstance(record, dict) and _is_earlier_source_schema(
            record.get("schema_version")
        ):
            _validate_schema(record.get("schema_version"), stage)
    cohort = receipt.get("preparation_cohort")
    if cohort in _EARLIER_PREPARATION_COHORTS:
        raise RuntimeError(
            f"receipt of preparation cohort {cohort} was executed from a handoff an "
            "earlier SCGSim wrote, which this version does not read (it reads "
            "prepared_v3); read this result with the SCGSim commit that wrote it"
        )
    # A malformed or unknown record is refused as invalid by the full checks.


_EARLIER_PREPARATION_COHORTS = {"verified_legacy_v1", "prepared_v2"}
"""Cohorts of receipt v1 and v2 handoffs; a V3 receipt names one when it ran one."""


def _is_earlier_source_schema(schema: Any) -> bool:
    number = (
        schema.removeprefix(_SOURCE_SCHEMA_PREFIX)
        if isinstance(schema, str) and schema.startswith(_SOURCE_SCHEMA_PREFIX)
        else ""
    )
    return number.isdigit() and int(number) < _SOURCE_SCHEMA_NUMBER


def initial_receipt_payload(  # noqa: PLR0913 -- retain the codec's explicit keyword interface.
    *,
    schema_version: str,
    mode: Any,
    requested: Any,
    pdk_materials: Any,
    vacuum_material_id: Any,
    source: Any,
    outputs: Any,
    prepared_at_utc: Any,
    prepared_runtime_source_value: Any,
) -> dict[str, Any]:
    """Build the canonical field order for an initial receipt."""
    if schema_version != RECEIPT_V3:
        raise ValueError("initial receipt schema is unsupported")
    if outputs != {}:
        raise ValueError("initial receipt outputs must be empty")
    if prepared_runtime_source_value is None:
        raise ValueError("v3 initial receipt requires prepared runtime source")
    return {
        "schema_version": schema_version,
        "status": "not_run",
        "mode": mode,
        "requested": requested,
        "pdk_materials": pdk_materials,
        "vacuum_material_id": vacuum_material_id,
        "source": source,
        "prepared_runtime_source": prepared_runtime_source_value,
        "outputs": outputs,
        "prepared_at_utc": prepared_at_utc,
    }


def encode_initial_receipt(value: Mapping[str, Any]) -> bytes:
    """Encode exact initial-receipt bytes with fixed indentation."""
    canonical = initial_receipt_payload(
        schema_version=cast(str, value.get("schema_version")),
        mode=value.get("mode"),
        requested=value.get("requested"),
        pdk_materials=value.get("pdk_materials"),
        vacuum_material_id=value.get("vacuum_material_id"),
        source=value.get("source"),
        prepared_runtime_source_value=value.get("prepared_runtime_source"),
        outputs=value.get("outputs"),
        prepared_at_utc=value.get("prepared_at_utc"),
    )
    if dict(value) != canonical:
        raise ValueError("initial receipt members are not canonical")
    return (json.dumps(canonical, indent=2) + "\n").encode("utf-8")


def initial_receipt_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact versioned initial-receipt codec bytes."""
    return hashlib.sha256(encode_initial_receipt(value)).hexdigest()
