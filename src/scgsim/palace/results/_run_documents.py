"""The shapes of the JSON documents in a Palace run directory.

These ``TypedDict`` classes only describe the files ``execution/handoff.py`` and
``execution/returned_receipt.py`` write. They check nothing. A document read from disk
is untrusted until the validators in this package have run, so every field is optional
(``total=False``) and the validators read it with ``.get``.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ._artifact_files import HashRecord


class PalaceIdentity(TypedDict, total=False):
    """What the handoff requires of Palace: its --dry-run must accept the config."""

    config_schema: str
    config_compatibility_probe: str
    config_compatibility_semantics: str
    runtime_identity_semantics: str


class JobIdentity(TypedDict, total=False):
    """The scheduler job that ran the solve (``manual`` outside a scheduler)."""

    job_id: str
    job_name: str


class ArchiveRecord(TypedDict, total=False):
    """The handoff tar archive as the resource record names it."""

    path: str
    sha256: str


class HandoffMetadata(TypedDict, total=False):
    """``metadata/palace_handoff_metadata.json``: what SCGSim prepared."""

    schema_version: int
    status: str
    kind: str
    manual_intent: bool
    profile: str
    route: str
    route_a_thin_film: dict[str, Any]
    problem: str
    handoff_id: str
    returned_receipt_path: str
    returned_receipt_recorder: str
    palace_identity: PalaceIdentity
    source_revisions: dict[str, Any]
    execution_identity: dict[str, Any]
    command: dict[str, Any]
    setup_commands: list[str]
    petsc_options: list[str]
    requested_resources: dict[str, Any]
    resolved_resources: dict[str, Any]
    script: str
    config: str
    mesh: str
    hashes: list[HashRecord]


class RunMetadata(TypedDict, total=False):
    """``metadata/palace_run_metadata.json``; the receipt recorder updates its status."""

    schema_version: int
    status: str
    kind: str
    route: str
    route_a_thin_film: dict[str, Any]
    problem: str
    handoff_id: str
    returned_receipt_path: str
    palace_identity: PalaceIdentity
    source_revisions: dict[str, Any]
    execution_identity: dict[str, Any]
    hashes: list[HashRecord]
    exit_code: int
    job_identity: JobIdentity


class ResourceRecord(TypedDict, total=False):
    """``metadata/palace_resource_record.json``: the scheduler's or script's account."""

    schema_version: int
    status: str
    kind: str
    profile: str
    route: str
    route_a_thin_film: dict[str, Any]
    problem: str
    handoff_id: str
    source_revisions: dict[str, Any]
    palace_identity: PalaceIdentity
    execution_identity: dict[str, Any]
    requested_resources: dict[str, Any]
    resolved_resources: dict[str, Any]
    archive: ArchiveRecord
    exit_code: int
    job_identity: JobIdentity
    returned_receipt_path: str


class ArchiveManifest(TypedDict, total=False):
    """``metadata/palace_handoff_archive_manifest.json``: every member of the archive."""

    schema_version: int
    archive: str
    input_hashes: list[HashRecord]
    members: list[dict[str, Any]]
    archive_members_excluded_from_self_listing: list[str]
    archive_members_excluded: list[str]


class ReturnedReceiptPayload(TypedDict, total=False):
    """``metadata/palace_returned_run_receipt.json``: the run script's own account."""

    schema: str
    schema_version: int
    handoff_id: str
    route: str
    route_a_thin_film: dict[str, Any]
    problem: str
    status: str
    exit_code: int
    execution_stage: str
    solver_invoked: bool
    preflight_exit_code: int
    solver_exit_code: int | None
    tee_exit_code: int | None
    identity_verified: bool
    timestamp_utc: str
    job_identity: JobIdentity
    input_hashes: list[HashRecord]
    output_files: list[dict[str, Any]]
    log: dict[str, Any]
    solver_identity: dict[str, Any]
    runtime_compatibility: dict[str, Any]


class IndexMap(TypedDict, total=False):
    """``metadata/palace_index_map.json``: which config entry is which named group."""

    schema_version: int
    entries: list[dict[str, Any]]
    ground_boundary: dict[str, Any]  # Electrostatic only
    epr_request: dict[str, Any]  # Eigenmode, only when a selection was requested
