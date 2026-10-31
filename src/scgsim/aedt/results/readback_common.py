"""Small checks shared by every receipt readback validator.

Owns: confining receipt paths to the handoff directory, checking a file against a
recorded SHA-256, and the native-ID shape check. It does not know any mode's
evidence layout; those live in the ``*_readback.py`` modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256
from scgsim.aedt.results.provenance import SOURCE_SCHEMA_V19

_HEX_DIGITS = "0123456789abcdef"


def contained_in_handoff_root(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; refuse absolute paths and ``..`` escapes.

    The receipt is data from disk, so a path it names must never leave the handoff.
    """
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    return resolved


def verified_output(
    root: Path, relative: Any, hashes: dict[str, Any], hash_key: str | None = None
) -> Path:
    """Return the file at ``relative`` only if its SHA-256 equals ``hashes[hash_key]``.

    ``hash_key`` defaults to ``relative``; the spec is keyed ``spec_sha256`` instead.
    """
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("receipt has an invalid relative output path")
    path = contained_in_handoff_root(root, relative)
    expected = hashes.get(hash_key or relative)
    if (
        not isinstance(expected, str)
        or not path.is_file()
        or file_sha256(path) != expected
    ):
        raise RuntimeError(f"receipt output hash mismatch: {relative}")
    return path


def is_sha256_hex(value: str) -> bool:
    """Whether ``value`` is a 64-character lowercase hexadecimal digest."""
    return len(value) == 64 and all(character in _HEX_DIGITS for character in value)


def positive_unique_ids(value: Any, count: int) -> bool:
    """Whether ``value`` is a list of ``count`` distinct positive integer native IDs.

    ``bool`` is an ``int`` subclass, so it is excluded explicitly.
    """
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
        and len(set(value)) == count
    )


def uses_current_terminal_rules(receipt: dict[str, Any]) -> bool:
    """Whether the receipt names runtime source schema V19 exactly.

    Exactly V19 selects the "current" Terminal rules: the de-embed flag and
    renormalizing impedance of wave ports are accepted as observed. Any other
    schema must show de-embed on and 50 ohm. A later schema therefore falls back to the old rules;
    this is a known issue awaiting the Human, so the comparison is left as is.
    """
    return receipt.get("runtime_source", {}).get("schema_version") == SOURCE_SCHEMA_V19
