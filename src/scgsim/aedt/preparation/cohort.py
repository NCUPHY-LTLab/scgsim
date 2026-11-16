"""The canonical file set of a prepared handoff and the hash cohort that binds it."""

from __future__ import annotations

from scgsim.aedt.preparation.sgb_handoff import sgb_model_paths
from scgsim.aedt.specs.hfss import (
    HfssEprAnalysisSpec,
)
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec

MANIFEST_PATH = "metadata/aedt_handoff_manifest.json"


def canonical_handoff_paths(spec: AedtSpec) -> list[str]:
    """Return the complete canonical file set for one prepared workflow."""
    paths = ["run_aedt.sh", "aedt_spec.json"]
    if not isinstance(spec, Q2dSpec):
        paths.extend(sgb_model_paths(spec.model))
    if isinstance(spec, HfssEprAnalysisSpec):
        paths.extend(f"saved/{item['path']}" for item in spec.saved_solution.members)
    paths.extend(
        (
            "metadata/aedt_handoff_metadata.json",
            "metadata/aedt_run_receipt.json",
            MANIFEST_PATH,
        )
    )
    return paths


def canonical_member_paths(spec: AedtSpec) -> list[str]:
    """Return canonical manifest members excluding the manifest itself."""
    return [path for path in canonical_handoff_paths(spec) if path != MANIFEST_PATH]
