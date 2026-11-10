"""Own deterministic junction record identities, format markers and plain values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class JunctionPartitionFormat:
    """Carry the metadata keys and identity markers for junction partitions."""

    metadata_key: str = "junction_partitions"
    source_hash_key: str = "junction_authored_input_sha256"
    method: str = "scgsim.geometry.junction-partition.v1"
    id_prefix: str = "junction"
    junction_metadata_key: str = "junction_id"
    support_schema: str = "scgsim.geometry.junction-support-plane.v1"


def canonical_sha256(value):
    """Hash canonical JSON while preserving the existing junction identities."""
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def plain_record(value: Any) -> Any:
    """Detach dataclasses, mappings and sequences into plain record values."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: plain_record(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_record(item) for item in value]
    return value


DEFAULT_FORMAT = JunctionPartitionFormat()
PRECISION_UM = 1e-09
SUPPORT_PLANE_SCHEMA = DEFAULT_FORMAT.support_schema
