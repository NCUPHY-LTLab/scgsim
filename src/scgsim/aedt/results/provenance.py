"""Versioned AEDT runtime source provenance and receipt codecs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Literal

from scgsim.aedt._io import file_sha256

RECEIPT_V3 = "scgsim.aedt.receipt.v3"

# V19 is no longer written; the name stays only because the terminal-evidence
# gates in `resolve.py` and `prepared.py` still compare against it.
SOURCE_SCHEMA_V19 = "scgsim.aedt.runtime-source.v19"

SOURCE_SCHEMA_V21 = "scgsim.aedt.runtime-source.v21"

SOURCE_SCHEMA = SOURCE_SCHEMA_V21

_RUNTIME_SOURCE_V21_PATHS = (
    "src/scgsim/aedt/__init__.py",
    "src/scgsim/aedt/_benchmark.py",
    "src/scgsim/aedt/_data.py",
    "src/scgsim/aedt/_io.py",
    "src/scgsim/aedt/_rlc.py",
    "src/scgsim/aedt/_terminal_chain.py",
    "src/scgsim/aedt/eigenmode.py",
    "src/scgsim/aedt/epr/analysis.py",
    "src/scgsim/aedt/epr/cache.py",
    "src/scgsim/aedt/epr/fields.py",
    "src/scgsim/aedt/epr/geometry.py",
    "src/scgsim/aedt/epr/models.py",
    "src/scgsim/aedt/epr/native.py",
    "src/scgsim/aedt/epr/selection.py",
    "src/scgsim/aedt/epr/workflow.py",
    "src/scgsim/aedt/preparation/cohort.py",
    "src/scgsim/aedt/preparation/geometry.py",
    "src/scgsim/aedt/preparation/handoff.py",
    "src/scgsim/aedt/preparation/junction_partition.py",
    "src/scgsim/aedt/preparation/modeling.py",
    "src/scgsim/aedt/preparation/planar.py",
    "src/scgsim/aedt/preparation/q3d_geometry.py",
    "src/scgsim/aedt/presentation/benchmark.py",
    "src/scgsim/aedt/presentation/epr.py",
    "src/scgsim/aedt/presentation/epr_history.py",
    "src/scgsim/aedt/presentation/epr_surfaces.py",
    "src/scgsim/aedt/presentation/runs.py",
    "src/scgsim/aedt/results/benchmark.py",
    "src/scgsim/aedt/results/convergence/_evidence_files.py",
    "src/scgsim/aedt/results/convergence/common.py",
    "src/scgsim/aedt/results/convergence/hfss.py",
    "src/scgsim/aedt/results/convergence/q2d.py",
    "src/scgsim/aedt/results/convergence/q3d.py",
    "src/scgsim/aedt/results/epr.py",
    "src/scgsim/aedt/results/matrices.py",
    "src/scgsim/aedt/results/prepared.py",
    "src/scgsim/aedt/results/provenance.py",
    "src/scgsim/aedt/results/resolve.py",
    "src/scgsim/aedt/run.py",
    "src/scgsim/aedt/runtime/_text.py",
    "src/scgsim/aedt/runtime/_timing.py",
    "src/scgsim/aedt/runtime/benchmark.py",
    "src/scgsim/aedt/runtime/families/hfss.py",
    "src/scgsim/aedt/runtime/families/q2d.py",
    "src/scgsim/aedt/runtime/families/q3d.py",
    "src/scgsim/aedt/runtime/handoff_checks.py",
    "src/scgsim/aedt/runtime/native/common.py",
    "src/scgsim/aedt/runtime/native/hfss_eigenmode.py",
    "src/scgsim/aedt/runtime/native/hfss_geometry.py",
    "src/scgsim/aedt/runtime/native/hfss_lumped.py",
    "src/scgsim/aedt/runtime/native/q3d_bodies.py",
    "src/scgsim/aedt/runtime/receipts.py",
    "src/scgsim/aedt/runtime/transaction.py",
    "src/scgsim/aedt/specs/__init__.py",
    "src/scgsim/aedt/specs/_support.py",
    "src/scgsim/aedt/specs/common.py",
    "src/scgsim/aedt/specs/epr.py",
    "src/scgsim/aedt/specs/geometry.py",
    "src/scgsim/aedt/specs/hfss.py",
    "src/scgsim/aedt/specs/hfss_prepared.py",
    "src/scgsim/aedt/specs/modeling.py",
    "src/scgsim/aedt/specs/parse.py",
    "src/scgsim/aedt/specs/q2d.py",
    "src/scgsim/aedt/specs/q3d.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/_primitives/entities.py",
    "src/scgsim/geometry/_primitives/geometry_refs.py",
    "src/scgsim/geometry/_primitives/loops.py",
    "src/scgsim/geometry/_primitives/spatial.py",
    "src/scgsim/geometry/_primitives/surface_records.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/models/lumped.py",
    "src/scgsim/geometry/models/physical.py",
    "src/scgsim/geometry/models/representation.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/evidence.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/physical.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/representation.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/source/_normalization.py",
    "src/scgsim/geometry/source/_polygon_ops.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/ground.py",
    "src/scgsim/geometry/source/intents.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/route_a.py",
    "src/scgsim/semantics/snapshot.py",
)


def _module_manifest() -> list[dict[str, str]]:
    source_root = Path(__file__).resolve().parents[3]
    return [
        {
            "module": path.removeprefix("src/").removesuffix(".py").replace("/", "."),
            "path": path,
            "sha256": file_sha256(source_root / path.removeprefix("src/")),
        }
        for path in _RUNTIME_SOURCE_V21_PATHS
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
    if schema != SOURCE_SCHEMA:
        raise RuntimeError(
            f"{stage} runtime source schema {schema!r} is unsupported; "
            f"only {SOURCE_SCHEMA!r} is supported"
        )
    expected_paths = _RUNTIME_SOURCE_V21_PATHS
    if (
        value.get("stage") != stage
        or not isinstance(modules, list)
        or len(modules) != len(expected_paths)
    ):
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
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
            or not item["path"].startswith("src/scgsim/")
            or Path(item["path"]).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise RuntimeError(f"{stage} runtime source module manifest is invalid")
        paths.append(item["path"])
    if tuple(paths) != expected_paths:
        raise RuntimeError(f"{stage} runtime source module manifest is invalid")
    if value.get("content_sha256") != _content_digest(modules):
        raise RuntimeError(f"{stage} runtime source content digest is invalid")
    if set(value) != {
        "schema_version",
        "stage",
        "modules",
        "content_sha256",
        "revision_observation",
    }:
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    _validate_revision_observation(value.get("revision_observation"), stage=stage)


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


def initial_receipt_payload(
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
    """Build the canonical field order for an initial v3 receipt."""
    if schema_version != RECEIPT_V3:
        raise ValueError(
            f"initial receipt schema {schema_version!r} is unsupported; "
            f"only {RECEIPT_V3!r} is supported"
        )
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
    """Encode the initial-receipt bytes with fixed indentation."""
    canonical = initial_receipt_payload(
        schema_version=value.get("schema_version"),
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
