"""Versioned AEDT runtime source provenance and receipt codecs."""

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
    SOURCE_SCHEMA as SOURCE_SCHEMA,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V1 as SOURCE_SCHEMA_V1,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V2 as SOURCE_SCHEMA_V2,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V3 as SOURCE_SCHEMA_V3,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V4 as SOURCE_SCHEMA_V4,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V5 as SOURCE_SCHEMA_V5,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V6 as SOURCE_SCHEMA_V6,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V7 as SOURCE_SCHEMA_V7,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V8 as SOURCE_SCHEMA_V8,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V9 as SOURCE_SCHEMA_V9,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V10 as SOURCE_SCHEMA_V10,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V11 as SOURCE_SCHEMA_V11,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V12 as SOURCE_SCHEMA_V12,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V13 as SOURCE_SCHEMA_V13,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V15 as SOURCE_SCHEMA_V15,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V16 as SOURCE_SCHEMA_V16,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V17 as SOURCE_SCHEMA_V17,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V18 as SOURCE_SCHEMA_V18,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V19 as SOURCE_SCHEMA_V19,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V20 as SOURCE_SCHEMA_V20,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V21 as SOURCE_SCHEMA_V21,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V22 as SOURCE_SCHEMA_V22,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V23 as SOURCE_SCHEMA_V23,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V24 as SOURCE_SCHEMA_V24,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V25 as SOURCE_SCHEMA_V25,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V26 as SOURCE_SCHEMA_V26,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V27 as SOURCE_SCHEMA_V27,
)
from scgsim.aedt.results.runtime_inventory import (
    SOURCE_SCHEMA_V28 as SOURCE_SCHEMA_V28,
)
from scgsim.aedt.results.runtime_inventory import (
    current_runtime_source_paths,
    runtime_source_paths,
)

RECEIPT_V1 = "scgsim.aedt.receipt.v1"

RECEIPT_V2 = "scgsim.aedt.receipt.v2"

RECEIPT_V3 = "scgsim.aedt.receipt.v3"


# V17 remains a frozen historical inventory. V18 follows the current AEDT
# ownership split and declares the relocated helpers and neutral geometry
# representation used by the current candidate.


# V1–V18 inventories remain historical; V19 adds the public model-stage reader.


# V1–V19 remain exact historical inventories. V20 binds the sole neutral
# physical interpreter and its source serialization dependency.


# V1–V20 retain exact historical scopes; V21 adds shared memory attribution.


# V1–V21 remain exact historical inventories; V22 binds explicit version
# selection and the native compatibility routing used by that selection.


# V1–V25 retain their historical module paths and receipt interpretation.
# V26 records current implementation owners after responsibility splits.
# V27 retains its committed ordered closure. V28 records the current AST-resolved
# internal import closure as a frozen roster.


def _module_manifest() -> list[dict[str, str]]:
    source_root = Path(__file__).resolve().parents[3]
    return [
        {
            "module": path.removeprefix("src/").removesuffix(".py").replace("/", "."),
            "path": path,
            "sha256": file_sha256(source_root / path.removeprefix("src/")),
        }
        for path in current_runtime_source_paths()
    ]


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


def prepared_runtime_source() -> dict[str, Any]:
    """Return complete module hashes plus an optional truthful Git observation."""
    modules = _module_manifest()
    revision = _observed_revision()
    return {
        "schema_version": SOURCE_SCHEMA,
        "stage": "prepared",
        "modules": modules,
        "content_sha256": _content_digest(modules),
        "revision_observation": (
            {"status": "available", "revision": revision}
            if revision is not None
            else {"status": "unavailable"}
        ),
    }


def runtime_source_identity() -> dict[str, Any]:
    """Bind execution to measured module bytes and a truthful Git observation."""
    modules = _module_manifest()
    revision = _observed_revision()
    return {
        "schema_version": SOURCE_SCHEMA,
        "stage": "actual",
        "modules": modules,
        "content_sha256": _content_digest(modules),
        "revision_observation": (
            {"status": "available", "revision": revision}
            if revision is not None
            else {"status": "unavailable"}
        ),
    }


def validate_runtime_source(
    value: Any, *, stage: Literal["prepared", "actual"]
) -> None:
    """Validate detached provenance shape without comparing it to installed bytes."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    modules = value.get("modules")
    schema = value.get("schema_version")
    expected_paths = runtime_source_paths(schema)
    if (
        expected_paths is None
        or value.get("stage") != stage
        or not isinstance(modules, list)
        or len(modules) != len(expected_paths)
    ):
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    _validate_module_records(modules, expected_paths, schema, stage)
    if value.get("content_sha256") != _content_digest(modules):
        raise RuntimeError(f"{stage} runtime source content digest is invalid")
    if stage == "prepared":
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            "revision_observation",
        }:
            raise RuntimeError("prepared runtime source provenance is invalid")
        _validate_revision_observation(value.get("revision_observation"), stage=stage)
    elif schema in {
        SOURCE_SCHEMA_V13,
        SOURCE_SCHEMA_V15,
        SOURCE_SCHEMA_V16,
        SOURCE_SCHEMA_V17,
        SOURCE_SCHEMA_V18,
        SOURCE_SCHEMA_V19,
        SOURCE_SCHEMA_V20,
        SOURCE_SCHEMA_V21,
        SOURCE_SCHEMA_V22,
        SOURCE_SCHEMA_V23,
        SOURCE_SCHEMA_V24,
        SOURCE_SCHEMA_V25,
        SOURCE_SCHEMA_V26,
        SOURCE_SCHEMA_V27,
        SOURCE_SCHEMA_V28,
    }:
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            "revision_observation",
        }:
            raise RuntimeError("actual runtime source provenance is invalid")
        _validate_revision_observation(value.get("revision_observation"), stage=stage)
    else:
        required_hashes = {
            "run_py_sha256",
            "spec_py_sha256",
            "hfss_convergence_py_sha256",
            "q2d_convergence_py_sha256",
            "q3d_convergence_py_sha256",
            "matrix_export_py_sha256",
        }
        current = schema == SOURCE_SCHEMA_V12
        revision_field = "revision_observation" if current else "revision"
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            revision_field,
            *required_hashes,
        } or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, "")))
            for key in required_hashes
        ):
            raise RuntimeError("actual runtime source legacy identity is invalid")
        if current:
            _validate_revision_observation(
                value.get("revision_observation"), stage=stage
            )
        elif not re.fullmatch(r"[0-9a-f]{40}", str(value.get("revision", ""))):
            raise RuntimeError("actual runtime source legacy identity is invalid")
        by_path = {item["path"]: item["sha256"] for item in modules}
        expected_legacy = {
            "run_py_sha256": by_path["scgsim/aedt/run.py"],
            "spec_py_sha256": by_path["scgsim/aedt/spec.py"],
            "hfss_convergence_py_sha256": by_path["scgsim/aedt/_hfss_convergence.py"],
            "q2d_convergence_py_sha256": by_path["scgsim/aedt/_q2d_convergence.py"],
            # Preserve the existing legacy key's historical source-file binding.
            "q3d_convergence_py_sha256": by_path["scgsim/aedt/_q2d_convergence.py"],
            "matrix_export_py_sha256": by_path["scgsim/aedt/_matrix_export.py"],
        }
        if any(value[key] != digest for key, digest in expected_legacy.items()):
            raise RuntimeError("actual runtime source legacy identity is invalid")


def _validate_module_records(
    modules: list[Any],
    expected_paths: tuple[str, ...],
    schema: Any,
    stage: str,
) -> None:
    """Check ordered module identities before validating the enclosing digest."""
    paths: list[str] = []
    for item in modules:
        expected_module = (
            item["path"].removeprefix("src/").removesuffix(".py").replace("/", ".")
            if isinstance(item, dict) and isinstance(item.get("path"), str)
            else None
        )
        if (
            not isinstance(item, dict)
            or set(item) != {"module", "path", "sha256"}
            or item["module"] != expected_module
            or not isinstance(item["path"], str)
            or not item["path"].startswith(
                "src/scgsim/"
                if schema
                in {
                    SOURCE_SCHEMA_V13,
                    SOURCE_SCHEMA_V15,
                    SOURCE_SCHEMA_V16,
                    SOURCE_SCHEMA_V17,
                    SOURCE_SCHEMA_V18,
                    SOURCE_SCHEMA_V19,
                    SOURCE_SCHEMA_V20,
                    SOURCE_SCHEMA_V21,
                    SOURCE_SCHEMA_V22,
                    SOURCE_SCHEMA_V23,
                    SOURCE_SCHEMA_V24,
                    SOURCE_SCHEMA_V25,
                    SOURCE_SCHEMA_V26,
                    SOURCE_SCHEMA_V27,
                    SOURCE_SCHEMA_V28,
                }
                else "scgsim/"
            )
            or Path(item["path"]).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise RuntimeError(f"{stage} runtime source module manifest is invalid")
        paths.append(item["path"])
    if tuple(paths) != expected_paths:
        raise RuntimeError(f"{stage} runtime source module manifest is invalid")


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


def initial_receipt_payload(  # noqa: PLR0913 -- retain the versioned codec's explicit keyword interface.
    *,
    schema_version: str,
    mode: Any,
    requested: Any,
    pdk_materials: Any,
    vacuum_material_id: Any,
    source: Any,
    outputs: Any,
    prepared_at_utc: Any,
    prepared_runtime_source_value: Any = None,
) -> dict[str, Any]:
    """Build the canonical field order for an initial v1, v2, or v3 receipt."""
    if schema_version not in {RECEIPT_V1, RECEIPT_V2, RECEIPT_V3}:
        raise ValueError("initial receipt schema is unsupported")
    if outputs != {}:
        raise ValueError("initial receipt outputs must be empty")
    result = {
        "schema_version": schema_version,
        "status": "not_run",
        "mode": mode,
        "requested": requested,
        "pdk_materials": pdk_materials,
        "vacuum_material_id": vacuum_material_id,
        "source": source,
    }
    if schema_version in {RECEIPT_V2, RECEIPT_V3}:
        if prepared_runtime_source_value is None:
            version = schema_version.rsplit(".", 1)[-1]
            raise ValueError(
                f"{version} initial receipt requires prepared runtime source"
            )
        result["prepared_runtime_source"] = prepared_runtime_source_value
    elif prepared_runtime_source_value is not None:
        raise ValueError("v1 initial receipt excludes prepared runtime source")
    result["outputs"] = outputs
    result["prepared_at_utc"] = prepared_at_utc
    return result


def encode_initial_receipt(value: Mapping[str, Any]) -> bytes:
    """Encode exact historical initial-receipt bytes with fixed indentation."""
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
