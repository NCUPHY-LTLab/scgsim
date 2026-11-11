"""The shared sidecar reader: a sidecar is checked once, the same way, before any reader indexes it.

Owns the order of the checks: the file is JSON (``NaN`` and ``Infinity`` are not), the version is the one
this definition describes, the document has the structure of :class:`.records.Sidecar` (checked against the
schema derived from it, :mod:`.structure`), and the references v2 can resolve point at what exists
(:mod:`.references`).  The first check that fails raises :class:`.errors.SidecarError` saying what failed and
where.  The document is returned as decoded, unchanged, typed as a :class:`.records.Sidecar`.
"""

import functools
import json
from pathlib import Path
from typing import Any, cast

from .errors import SidecarError
from .records import Sidecar
from .references import require_references
from .schema import sidecar_schema
from .structure import require_known_keywords, require_structure
from .version import SIDECAR_SCHEMA


def read_sidecar(path: Path | str) -> Sidecar:
    """The checked sidecar stored in the file ``path``."""
    return check_sidecar(_decoded(Path(path)))


def check_sidecar(document: object) -> Sidecar:
    """``document``, a decoded sidecar, once its version, structure and references hold."""
    _require_version(document)
    require_structure(document, _schema())
    sidecar = cast(Sidecar, document)
    require_references(sidecar)
    return sidecar


def _decoded(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_not_json)
    # A JSONDecodeError, a UnicodeDecodeError, or the NaN or Infinity that _not_json refuses.
    except ValueError as error:
        raise SidecarError(f"sidecar {path.name} is not JSON: {error}") from None


def _not_json(constant: str) -> Any:
    raise ValueError(f"{constant} is not a JSON number")


def _require_version(document: object) -> None:
    """The version first, so a sidecar of another version is named as such, not by its first difference."""
    if not isinstance(document, dict):
        raise SidecarError(
            f"a sidecar is a JSON object, not {type(document).__name__}."
        )
    version = document.get("schema")
    if version != SIDECAR_SCHEMA:
        raise SidecarError(
            f"sidecar schema {version!r} is not supported; this reader reads {SIDECAR_SCHEMA!r}."
        )


@functools.cache
def _schema() -> dict[str, Any]:
    """The derived schema, checked once to use only keywords :mod:`.structure` enforces."""
    schema = sidecar_schema()
    require_known_keywords(schema)
    return schema
