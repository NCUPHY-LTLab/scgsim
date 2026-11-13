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
from scgsim.geometry.models.refusals import Declared
from scgsim.palace.results.artifacts import (
    RETURNED_RUN_FOLDER,
    expect_mapping,
    expect_scalar,
)
from scgsim.refusal import found_text


def validate_palace_identity(payload: Mapping[str, Any]) -> None:
    """Validate solver provenance without introducing version-based eligibility policy."""
    if not payload:
        raise ValueError(
            Declared(
                "palace_identity must be a non-empty mapping", RETURNED_RUN_FOLDER
            ).refusal(found_text(payload))
        )
    expect_scalar(payload, "config_schema", str)
    probe = payload.get("config_compatibility_probe")
    if probe is None:
        expect_scalar(payload, "runtime", str)
        return
    if probe != "native_dry_run":
        raise ValueError(
            Declared(
                "palace_identity config compatibility probe is invalid",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(probe, "native_dry_run"))
        )
    semantics = payload.get("config_compatibility_semantics")
    if semantics != "required_before_solve":
        raise ValueError(
            Declared(
                "palace_identity must require config compatibility before solving",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(semantics, "required_before_solve"))
        )
    semantics = payload.get("runtime_identity_semantics")
    if semantics != "observed_provenance_only":
        raise ValueError(
            Declared(
                "palace_identity must treat runtime identity as observed provenance only",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(semantics, "observed_provenance_only"))
        )
    if (
        payload.get("runtime") is not None
        or payload.get("runtime_git_commit") is not None
    ):
        raise ValueError(
            Declared(
                "palace_identity must not declare a runtime version or Git eligibility Gate",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    {
                        "runtime": payload.get("runtime"),
                        "runtime_git_commit": payload.get("runtime_git_commit"),
                    }
                )
            )
        )


def validate_runtime_compatibility(
    payload: Mapping[str, Any], palace_identity: Mapping[str, Any], *, root: Path
) -> None:
    """Bind successful native dry-run observations to the exact sealed config bytes."""
    _validate_dry_run(payload, palace_identity, root)
    runtime_binary_identity = _observed_binary_identity(payload)
    _validate_probe_command_record(payload, runtime_binary_identity)


def _validate_dry_run(
    payload: Mapping[str, Any], palace_identity: Mapping[str, Any], root: Path
) -> None:
    """The record has the current schema, names the sealed config bytes, and says Palace accepted them."""
    if payload.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA:
        raise ValueError(
            Declared(
                f"Palace runtime compatibility schema in {RUNTIME_COMPATIBILITY} is invalid",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(payload.get("schema"), RUNTIME_COMPATIBILITY_SCHEMA))
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
            Declared(
                "Palace runtime compatibility probe does not bind the sealed config",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(config, expected_config))
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
            Declared(
                "Palace did not accept the sealed config during dry-run; read the dry-run log in "
                f"{RUNTIME_COMPATIBILITY}",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(compatibility))
        )


def _observed_binary_identity(payload: Mapping[str, Any]) -> str | None:
    """The observed Palace version, and the binary identity: a basename and its SHA-256, both or neither."""
    observed = expect_mapping(
        payload.get("observed_identity"), "runtime observed identity"
    )
    _validate_observed_version(observed)
    runtime_binary_identity = observed.get("runtime_binary_identity")
    runtime_binary_sha256 = observed.get("runtime_binary_sha256")
    if (runtime_binary_identity is None) != (runtime_binary_sha256 is None):
        raise ValueError(
            Declared(
                "Palace runtime binary provenance must be complete or absent",
                RETURNED_RUN_FOLDER,
            ).refusal(
                found_text(
                    {
                        "runtime_binary_identity": runtime_binary_identity,
                        "runtime_binary_sha256": runtime_binary_sha256,
                    }
                )
            )
        )
    if runtime_binary_identity is not None:
        if (
            not isinstance(runtime_binary_identity, str)
            or not runtime_binary_identity
            or Path(runtime_binary_identity).name != runtime_binary_identity
        ):
            raise ValueError(
                Declared(
                    "Palace runtime binary identity must be a basename",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(runtime_binary_identity))
            )
        if (
            not isinstance(runtime_binary_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", runtime_binary_sha256) is None
        ):
            raise ValueError(
                Declared(
                    "Palace runtime binary SHA-256 is not 64 hex digits",
                    RETURNED_RUN_FOLDER,
                ).refusal(found_text(runtime_binary_sha256))
            )
    return runtime_binary_identity


def _validate_probe_command_record(
    payload: Mapping[str, Any], runtime_binary_identity: str | None
) -> None:
    """The probe's command: a known style, a basename executable, and a binary hash only for a binary."""
    command = expect_mapping(payload.get("command"), "runtime compatibility command")
    if command.get("command_style") not in {"binary", "wrapper"}:
        raise ValueError(
            Declared(
                "Palace runtime compatibility command style is invalid",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(command.get("command_style"), "binary or wrapper"))
        )
    executable_identity = expect_scalar(command, "executable_identity", str)
    if Path(executable_identity).name != executable_identity:
        raise ValueError(
            Declared(
                "Palace runtime executable must be a basename", RETURNED_RUN_FOLDER
            ).refusal(found_text(executable_identity))
        )
    if runtime_binary_identity is not None and command.get("command_style") != "binary":
        raise ValueError(
            Declared(
                "Palace runtime binary SHA may describe only the configured binary",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(command.get("command_style"), "binary"))
        )


def _validate_observed_version(observed: Any) -> None:
    """Validate optional native version observations without making them an execution Gate."""
    native_git_tag = observed.get("native_git_tag")
    if native_git_tag is not None and (
        not isinstance(native_git_tag, str) or not native_git_tag
    ):
        raise ValueError(
            Declared(
                "Palace observed native GitTag must be a non-empty string",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(native_git_tag))
        )
    native_schema_version = observed.get("native_schema_version")
    if native_schema_version is not None and (
        not isinstance(native_schema_version, str) or not native_schema_version
    ):
        raise ValueError(
            Declared(
                "Palace observed native schema version must be non-empty text or null",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(native_schema_version))
        )
    version_probe_exit_code = observed.get("version_probe_exit_code")
    if version_probe_exit_code is not None and (
        isinstance(version_probe_exit_code, bool)
        or not isinstance(version_probe_exit_code, int)
    ):
        raise ValueError(
            Declared(
                "Palace version probe exit code must be an integer or null",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(version_probe_exit_code))
        )
