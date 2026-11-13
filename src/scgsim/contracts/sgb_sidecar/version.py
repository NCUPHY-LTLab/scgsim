"""The version of the SGB sidecar format, and the digest of the schema that version stands for.

Owns the one place the version is written: the writer stamps it into every sidecar and every reader accepts
exactly this version.  ``SCHEMA_DIGEST`` is the SHA-256 of the JSON Schema derived from :mod:`.records`
(:func:`.schema.schema_digest`).  A test derives the schema again and compares, so a change to any record,
field, enum or bound fails until the version is raised here and the new digest pinned beside it (#170).
"""

from typing import Final, Literal

SidecarVersion = Literal["scgsim.sgb.sidecar.v4"]
"""The type of a sidecar's ``schema`` field: the one value a sidecar of this definition carries."""

SIDECAR_SCHEMA: Final[SidecarVersion] = "scgsim.sgb.sidecar.v4"
SCHEMA_DIGEST: Final = (
    "sha256:db7b6f059c4be7cb7ab1d6c86804cd3a9fe9d2afd29046e3ebaef70016d67e83"
)
