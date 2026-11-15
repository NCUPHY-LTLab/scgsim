"""The canonical file set of a prepared handoff and the hash cohort that binds it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from scgsim.aedt.io import file_sha256
from scgsim.aedt.preparation.sgb_handoff import sgb_model_paths
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec

MANIFEST_PATH = "metadata/aedt_handoff_manifest.json"


def canonical_handoff_paths(spec: AedtSpec) -> list[str]:
    """Return the complete canonical file set for one prepared workflow."""
    paths = ["run_aedt.sh", "aedt_spec.json"]
    if isinstance(spec, (HfssEprSpec, HfssEprAnalysisSpec, Q3dSpec)):
        paths.extend(sgb_model_paths(spec.model))
    if isinstance(spec, HfssEprAnalysisSpec):
        paths.extend(f"saved/{item['path']}" for item in spec.saved_solution.members)
    elif not isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec, Q2dSpec, Q3dSpec)):
        paths.append("geometry/design.gds")
        if spec.modeling is not None:
            paths.append("geometry/source.gds")
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


def validate_hfss_import_source(
    root: Path, raw_payload: Mapping, spec: AedtSpec
) -> None:
    """Bind explicit native import lineage to immutable original GDS bytes."""
    if isinstance(
        spec,
        (HfssEprSpec, HfssEprAnalysisSpec, HfssDrivenGeometrySpec, Q2dSpec, Q3dSpec),
    ):
        return
    gds = raw_payload["gds"]
    if (
        gds.get("source_path") != "geometry/source.gds"
        or gds.get("path") != "geometry/design.gds"
    ):
        raise RuntimeError("HFSS source/import GDS paths are not canonical")
    if file_sha256(root / "geometry/source.gds") != gds.get("source_sha256"):
        raise RuntimeError("HFSS original GDS source identity differs")
    keys = (
        "layer",
        "datatype",
        "layer_name",
        "physical_layer_id",
        "native_import_layer",
    )
    expected = [
        {key: item[key] for key in keys} for item in spec.effective_layer_imports
    ]
    if gds.get("import_layer_map") != expected:
        raise RuntimeError("HFSS import layer mapping differs from declared source")
