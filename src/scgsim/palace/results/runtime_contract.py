"""Check what a handoff declares about Palace and what its config dry-run observed.

``palace_identity`` in the handoff states how the solver is identified and that the config must pass a native
dry-run before the solve; the runtime-compatibility record holds what that dry-run observed."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import (
    CONFIG,
    RUNTIME_COMPATIBILITY,
    RUNTIME_COMPATIBILITY_SCHEMA,
    sha256,
)
from scgsim.palace.results.artifacts import (
    expect_mapping,
    expect_scalar,
)
from scgsim.refusal import refusal


def validate_palace_identity(payload: Mapping[str, Any]) -> None:
    """Validate solver provenance without introducing version-based eligibility policy."""
    if not payload:
        raise ValueError(
            refusal("palace_identity must be a non-empty mapping", found=payload)
        )
    expect_scalar(payload, "config_schema", str)
    probe = payload.get("config_compatibility_probe")
    if probe is None:
        expect_scalar(payload, "runtime", str)
        return
    if probe != "native_dry_run":
        raise ValueError(
            refusal(
                "palace_identity config compatibility probe is invalid",
                found=probe,
                expected="native_dry_run",
            )
        )
    semantics = payload.get("config_compatibility_semantics")
    if semantics != "required_before_solve":
        raise ValueError(
            refusal(
                "palace_identity must require config compatibility before solving",
                found=semantics,
                expected="required_before_solve",
            )
        )
    semantics = payload.get("runtime_identity_semantics")
    if semantics != "observed_provenance_only":
        raise ValueError(
            refusal(
                "palace_identity must treat runtime identity as observed provenance only",
                found=semantics,
                expected="observed_provenance_only",
            )
        )
    if (
        payload.get("runtime") is not None
        or payload.get("runtime_git_commit") is not None
    ):
        raise ValueError(
            refusal(
                "palace_identity must not declare a runtime version or Git eligibility Gate",
                found={
                    "runtime": payload.get("runtime"),
                    "runtime_git_commit": payload.get("runtime_git_commit"),
                },
            )
        )


def validate_runtime_compatibility(
    payload: Mapping[str, Any], palace_identity: Mapping[str, Any], *, root: Path
) -> None:
    """Bind successful native dry-run observations to the exact sealed config bytes."""
    if payload.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA:
        raise ValueError(
            refusal(
                f"Palace runtime compatibility schema in {RUNTIME_COMPATIBILITY} is invalid",
                found=payload.get("schema"),
                expected=RUNTIME_COMPATIBILITY_SCHEMA,
            )
        )
    config = expect_mapping(payload.get("config"), "runtime compatibility config")
    expected_config = {
        "path": CONFIG,
        "bytes": (root / CONFIG).stat().st_size,
        "sha256": sha256(root / CONFIG),
        "declared_schema": expect_scalar(palace_identity, "config_schema", str),
    }
    if config != expected_config:
        raise ValueError(
            refusal(
                "Palace runtime compatibility probe does not bind the sealed config",
                found=config,
                expected=expected_config,
            )
        )
    compatibility = expect_mapping(
        payload.get("compatibility"), "runtime compatibility result"
    )
    if compatibility != {
        "method": "native_dry_run",
        "probe_exit_code": 0,
        "verified": True,
    }:
        raise ValueError(
            refusal(
                "Palace did not accept the sealed config during dry-run; read the dry-run log in "
                f"{RUNTIME_COMPATIBILITY}",
                found=compatibility,
            )
        )
    observed = expect_mapping(
        payload.get("observed_identity"), "runtime observed identity"
    )
    _validate_observed_version(observed)
    runtime_binary_identity = observed.get("runtime_binary_identity")
    runtime_binary_sha256 = observed.get("runtime_binary_sha256")
    if (runtime_binary_identity is None) != (runtime_binary_sha256 is None):
        raise ValueError(
            refusal(
                "Palace runtime binary provenance must be complete or absent",
                found={
                    "runtime_binary_identity": runtime_binary_identity,
                    "runtime_binary_sha256": runtime_binary_sha256,
                },
            )
        )
    if runtime_binary_identity is not None:
        if (
            not isinstance(runtime_binary_identity, str)
            or not runtime_binary_identity
            or Path(runtime_binary_identity).name != runtime_binary_identity
        ):
            raise ValueError(
                refusal(
                    "Palace runtime binary identity must be a basename",
                    found=runtime_binary_identity,
                )
            )
        if (
            not isinstance(runtime_binary_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", runtime_binary_sha256) is None
        ):
            raise ValueError(
                refusal(
                    "Palace runtime binary SHA-256 is not 64 hex digits",
                    found=runtime_binary_sha256,
                )
            )
    command = expect_mapping(payload.get("command"), "runtime compatibility command")
    if command.get("command_style") not in {"binary", "wrapper"}:
        raise ValueError(
            refusal(
                "Palace runtime compatibility command style is invalid",
                found=command.get("command_style"),
                expected="binary or wrapper",
            )
        )
    executable_identity = expect_scalar(command, "executable_identity", str)
    if Path(executable_identity).name != executable_identity:
        raise ValueError(
            refusal(
                "Palace runtime executable must be a basename",
                found=executable_identity,
            )
        )
    if runtime_binary_identity is not None and command.get("command_style") != "binary":
        raise ValueError(
            refusal(
                "Palace runtime binary SHA may describe only the configured binary",
                found=command.get("command_style"),
                expected="binary",
            )
        )


def _validate_observed_version(observed: Any) -> None:
    """Validate optional native version observations without making them an execution Gate."""
    native_git_tag = observed.get("native_git_tag")
    if native_git_tag is not None and (
        not isinstance(native_git_tag, str) or not native_git_tag
    ):
        raise ValueError(
            refusal(
                "Palace observed native GitTag must be a non-empty string",
                found=native_git_tag,
            )
        )
    native_schema_version = observed.get("native_schema_version")
    if native_schema_version is not None and (
        not isinstance(native_schema_version, str) or not native_schema_version
    ):
        raise ValueError(
            refusal(
                "Palace observed native schema version must be non-empty text or null",
                found=native_schema_version,
            )
        )
    version_probe_exit_code = observed.get("version_probe_exit_code")
    if version_probe_exit_code is not None and (
        isinstance(version_probe_exit_code, bool)
        or not isinstance(version_probe_exit_code, int)
    ):
        raise ValueError(
            refusal(
                "Palace version probe exit code must be an integer or null",
                found=version_probe_exit_code,
            )
        )
