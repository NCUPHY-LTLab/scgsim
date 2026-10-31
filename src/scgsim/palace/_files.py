"""File hashing shared by Palace handoff writing and returned-run checking.

Owns streaming SHA-256 of a file. It does not own which files are hashed or how the
digests are compared; ``execution/handoff.py`` and ``results/_artifact_files.py``
decide that. ``execution/returned_receipt.py`` keeps its own copy on purpose: it is a
pure-stdlib script that runs on machines without scgsim installed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of ``path``, read in 1 MiB chunks so large archives stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
