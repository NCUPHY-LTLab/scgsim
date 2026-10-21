"""Scope pinned PyAEDT's documented legacy local gRPC transport to one run.

Native operation semantics remain shared. The 2024.2 path leaves transport
configuration untouched; 2023.2 temporarily uses the pre-secure gRPC arguments.
Both keep exception cleanup with the transaction instead of SDK-wide release.
No installation discovery, SDK instance or connection is performed here.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator

from scgsim.aedt.versions import AedtVersionProfile


@contextmanager
def native_compatibility(
    profile: AedtVersionProfile, *, evidence: dict[str, Any] | None = None
) -> Iterator[dict[str, Any]]:
    """Keep the transaction sole release owner until its cleanup has completed."""
    # This context is entered only at the explicit native execution boundary.
    from ansys.aedt.core.generic.settings import settings

    if evidence is None:
        evidence = {}
    previous_release = settings.release_on_exception
    release_evidence = {
        "requested": False, "previous": previous_release, "state": "not_selected"
    }
    evidence["release_on_exception"] = release_evidence
    primary: BaseException | None = None
    try:
        settings.release_on_exception = False
        release_evidence["state"] = "selected"
        with _native_transport_compatibility(profile, evidence=evidence):
            yield evidence
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            settings.release_on_exception = previous_release
            release_evidence["state"] = "restored"
        except Exception as exc:
            release_evidence["state"] = "restore_failed"
            release_evidence["error"] = f"{type(exc).__name__}: {exc}"
            if primary is not None:
                primary.add_note(
                    "SDK release-on-exception restoration: " + release_evidence["error"]
                )
            else:
                raise RuntimeError("failed to restore SDK release-on-exception policy") from exc


@contextmanager
def _native_transport_compatibility(
    profile: AedtVersionProfile, *, evidence: dict[str, Any] | None = None
) -> Iterator[dict[str, Any]]:
    """Hold process-local transport overrides through the owned Desktop release."""
    if evidence is None:
        evidence = {}
    evidence.update({
        "requested_aedt_version": profile.requested_version,
        "legacy_local_grpc": profile.legacy_local_grpc,
        "state": "unchanged",
    })
    if not profile.legacy_local_grpc:
        yield evidence
        return

    # Import only at explicit native execution, never during offline preparation.
    from ansys.aedt.core.generic.settings import settings

    overrides = {
        "use_grpc_api": True,
        "grpc_secure_mode": False,
        "grpc_local": True,
        "grpc_listen_all": False,
    }
    previous = {name: getattr(settings, name) for name in overrides}
    environment_key = "PYAEDT_USE_PRE_GRPC_ARGS"
    had_environment = environment_key in os.environ
    previous_environment = os.environ.get(environment_key)
    previous_launch_environment = settings.aedt_environment_variables
    evidence["settings"] = dict(overrides)
    evidence["legacy_arguments"] = True
    primary: BaseException | None = None
    try:
        for name, value in overrides.items():
            setattr(settings, name, value)
        # PyAEDT replays this mapping immediately before launch; scope the same
        # flag there so it cannot undo the selected process environment value.
        settings.aedt_environment_variables = {
            **previous_launch_environment, environment_key: "True"
        }
        os.environ[environment_key] = "True"
        evidence["state"] = "selected"
        yield evidence
    except BaseException as exc:
        primary = exc
        raise
    finally:
        errors: list[Exception] = []
        for name, value in previous.items():
            try:
                setattr(settings, name, value)
            except Exception as exc:
                errors.append(exc)
        try:
            settings.aedt_environment_variables = previous_launch_environment
        except Exception as exc:
            errors.append(exc)
        try:
            if had_environment:
                os.environ[environment_key] = previous_environment
            else:
                os.environ.pop(environment_key, None)
        except Exception as exc:
            errors.append(exc)
        evidence["state"] = "restore_failed" if errors else "restored"
        if errors:
            evidence["restore_errors"] = [
                f"{type(error).__name__}: {error}" for error in errors
            ]
        if errors:
            failure = RuntimeError("failed to restore process-local AEDT transport state")
            for error in errors:
                failure.add_note(f"{type(error).__name__}: {error}")
            if primary is not None:
                primary.add_note(str(failure))
                for error in errors:
                    primary.add_note(f"transport restoration: {type(error).__name__}: {error}")
            else:
                raise failure from errors[0]
