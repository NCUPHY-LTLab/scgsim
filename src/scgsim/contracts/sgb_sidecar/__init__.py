"""The SGB sidecar contract: one definition shared by the writer, every reader and the JSON Schema (#170).

``version``
    ``SIDECAR_SCHEMA``, the one version constant, and ``SCHEMA_DIGEST``, the digest of the schema it stands
    for.  Any change to the definition raises the version.
``records``
    every table and record of the sidecar as a ``TypedDict``, with its enums, bounds and the table each id
    names (``markers``).  ``values`` holds the value types and ids, ``features`` the mesh-feature records.
``schema``
    the JSON Schema derived from the records; ``scripts/generate_sidecar_schema.py`` writes it to
    ``scgsim/geometry/sgb/sidecar/sidecar.schema.json``.
``read``
    ``read_sidecar`` and ``check_sidecar``: version, structure (``structure``) and references
    (``references``), raising ``SidecarError``.
``views``
    what readers need and the file stores once: native tags, group members, the load-side expectations by
    tag, the embeddings, ports and mesh features in reading order.

The package imports nothing else from ``scgsim`` and only the standard library.
"""

from .errors import SidecarError
from .read import check_sidecar, read_sidecar
from .records import Sidecar
from .schema import schema_digest, sidecar_schema
from .version import SCHEMA_DIGEST, SIDECAR_SCHEMA

__all__ = [
    "SCHEMA_DIGEST",
    "SIDECAR_SCHEMA",
    "Sidecar",
    "SidecarError",
    "check_sidecar",
    "read_sidecar",
    "schema_digest",
    "sidecar_schema",
]
