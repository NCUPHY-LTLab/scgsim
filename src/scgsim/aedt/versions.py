"""Explicit AEDT request versions and their pinned transport profiles.

Profiles describe source routing; native availability is observed at execution.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AedtVersionProfile:
    """One supported requested release and its backend transport selection."""

    requested_version: str
    pyaedt_version: str
    legacy_local_grpc: bool


_PROFILES = {
    "2023.2": AedtVersionProfile("2023.2", "1.3.0", True),
    "2024.2": AedtVersionProfile("2024.2", "1.3.0", False),
}


def resolve_aedt_version(value: str) -> AedtVersionProfile:
    """Resolve an exact public version; omission and release aliases are unsupported."""
    if not isinstance(value, str):
        raise TypeError("aedt_version must be '2023.2' or '2024.2'")
    try:
        return _PROFILES[value]
    except KeyError as exc:
        raise ValueError("aedt_version must be '2023.2' or '2024.2'") from exc


def normalize_aedt_version(value: str) -> str:
    """Return the exact supported requested version without guessing a release."""
    return resolve_aedt_version(value).requested_version
