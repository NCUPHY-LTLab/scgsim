"""The version of the SGB sidecar format, and the digest of the schema that version stands for.

Owns the one place the version is written: the writer stamps it into every sidecar and every reader accepts
exactly this version.  ``SCHEMA_DIGEST`` is the SHA-256 of the JSON Schema derived from :mod:`.records`
(:func:`.schema.schema_digest`).  A test derives the schema again and compares, so a change to any record,
field, enum or bound fails until the version is raised here and the new digest pinned beside it (#170).
"""

from typing import Final, Literal

SidecarVersion = Literal["scgsim.sgb.sidecar.v3"]
"""The type of a sidecar's ``schema`` field: the one value a sidecar of this definition carries."""

SIDECAR_SCHEMA: Final[SidecarVersion] = "scgsim.sgb.sidecar.v3"
SCHEMA_DIGEST: Final = (
    "sha256:6f5c1040ef5b90422a04a733d9076f0521288dec87f9f93b451985e0e9d044cb"
)
