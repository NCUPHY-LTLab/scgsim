"""Capture SCGSim source and installed-package identities for a study."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def capture_host_resources() -> dict[str, Any]:
    """Record standard-library CPU and process-affinity observations."""
    cpu_count = os.cpu_count()
    try:
        affinity_cpu_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        affinity = {
            "value": None,
            "reason": f"process CPU affinity is unavailable: {type(exc).__name__}",
        }
    else:
        affinity = {"value": affinity_cpu_ids, "reason": None}
    machine = platform.machine() or None
    processor = platform.processor() or None
    return {
        "logical_cpu_count": {
            "value": cpu_count,
            "reason": None if cpu_count is not None else "os.cpu_count returned None",
        },
        "process_affinity_cpu_ids": affinity,
        "machine_architecture": machine,
        "processor_description": processor,
    }


def capture_scgsim_provenance(repository_root: Path) -> dict[str, Any]:
    """Record available Git and installed distribution identities without gating."""
    repository = {
        "head": _git_text(repository_root, ("rev-parse", "HEAD")),
        "tree": _git_text(repository_root, ("rev-parse", "HEAD^{tree}")),
        "working_tree_dirty": _git_dirty(repository_root),
    }
    try:
        distribution = importlib.metadata.distribution("scgsim")
    except importlib.metadata.PackageNotFoundError:
        installed = {
            "version": None,
            "version_reason": "installed scgsim distribution metadata is unavailable",
            "vcs_commit_id": None,
            "vcs_commit_reason": "installed distribution metadata is unavailable",
            "editable": None,
        }
    else:
        try:
            direct_url = distribution.read_text("direct_url.json")
        except (OSError, UnicodeError) as exc:
            direct_url = None
            direct_reason = f"direct_url.json could not be read: {type(exc).__name__}"
        else:
            direct_reason = None
        direct_record = _direct_url_provenance(direct_url)
        if direct_reason is not None:
            direct_record["vcs_commit_reason"] = direct_reason
        installed = {"version": distribution.version, **direct_record}
    return {"repository": repository, "installed_distribution": installed}


def _direct_url_provenance(direct_url: str | None) -> dict[str, Any]:
    """Extract a VCS commit identity without persisting a local source URL."""
    if direct_url is None:
        return {
            "vcs_commit_id": None,
            "vcs_commit_reason": "direct_url.json is unavailable",
            "editable": None,
        }
    try:
        record = json.loads(direct_url)
    except json.JSONDecodeError:
        return {
            "vcs_commit_id": None,
            "vcs_commit_reason": "direct_url.json is not valid JSON",
            "editable": None,
        }
    vcs_info = record.get("vcs_info")
    commit_id = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    dir_info = record.get("dir_info")
    editable = dir_info.get("editable") if isinstance(dir_info, dict) else None
    return {
        "vcs_commit_id": commit_id if isinstance(commit_id, str) else None,
        "vcs_commit_reason": (
            None
            if isinstance(commit_id, str)
            else "direct_url.json has no vcs_info.commit_id"
        ),
        "editable": editable if isinstance(editable, bool) else None,
    }


def _git_text(repository_root: Path, arguments: tuple[str, ...]) -> dict[str, Any]:
    """Observe one Git text value, preserving unavailable reasons as data."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"value": None, "reason": f"Git could not run: {type(exc).__name__}"}
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        return {
            "value": None,
            "reason": f"git {' '.join(arguments)} exited {completed.returncode}",
        }
    return {"value": value, "reason": None}


def _git_dirty(repository_root: Path) -> dict[str, Any]:
    """Observe tracked and untracked worktree changes without listing file paths."""
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"value": None, "reason": f"Git could not run: {type(exc).__name__}"}
    if completed.returncode != 0:
        return {
            "value": None,
            "reason": f"git status exited {completed.returncode}",
        }
    return {"value": bool(completed.stdout.strip()), "reason": None}
