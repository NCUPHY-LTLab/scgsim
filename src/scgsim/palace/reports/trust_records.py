"""The records a trust report holds besides its snapshots: its identity cards and the provenance it keeps."""

from __future__ import annotations

from typing import Any, Required, TypedDict

from scgsim.palace.mesh.summary import MeshSummary


class ReportProvenance(TypedDict, total=False):
    """The handoff's source and resource identities a trust report keeps, its mesh summary and the receipt.

    A handoff field it did not record, and the receipt of a run that returned none, are left out.
    """

    source_revisions: dict[str, Any]
    palace_identity: dict[str, Any]
    hashes: list[dict[str, Any]]
    requested_resources: dict[str, Any]
    resolved_resources: dict[str, Any]
    route_a_thin_film: dict[str, Any]
    mesh_summary: Required[MeshSummary]
    returned_receipt: dict[str, Any]


class ReportIdentity(TypedDict):
    """The identity cards of a trust report: the handoff, the parallel resources, the receipt and Palace's tag."""

    handoff_id: str
    handoff_id_short: str | None
    resources: str | None
    receipt: str | None
    git_tag: str | None
