"""Results an earlier SCGSim wrote are refused with how to read them.

An old result is read with the SCGSim commit that wrote it; this version reads
only the formats it writes.
"""

from __future__ import annotations

import json

import pytest
from test_aedt_public_preparation import file_only_geometry, produce_request

from scgsim.aedt import (
    EprAnalysisRequest,
    PreparedPlanarGeometry,
    SurfaceEprSpec,
    resolve_epr_result,
    resolve_saved_solution,
)
from scgsim.aedt.io import write_json
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    record_runtime_source,
    require_current_receipt,
)
from scgsim.aedt.results.resolve import resolve_results
from scgsim.aedt.specs.parse import parse_aedt_spec


@pytest.mark.parametrize("schema", ["scgsim.aedt.receipt.v1", "scgsim.aedt.receipt.v2"])
def test_an_earlier_receipt_is_refused_with_how_to_read_it(tmp_path, schema):
    """A completed run of an earlier receipt schema is not resolved."""
    write_json(
        tmp_path / "metadata/aedt_run_receipt.json",
        {"schema_version": schema, "status": "completed"},
    )
    with pytest.raises(
        RuntimeError,
        match=(
            rf"receipt {schema} was written by an earlier SCGSim.*"
            "read this result with the SCGSim commit that wrote it"
        ),
    ):
        resolve_results(tmp_path)


@pytest.mark.parametrize("schema", ["scgsim.aedt.receipt.v4", "other", None])
def test_an_unknown_receipt_schema_is_invalid(schema):
    """A schema that is neither current nor earlier is invalid, not 'earlier'."""
    with pytest.raises(RuntimeError, match="handoff receipt schema is invalid"):
        require_current_receipt({"schema_version": schema})


@pytest.mark.parametrize(
    ("key", "stage"),
    [("prepared_runtime_source", "prepared"), ("runtime_source", "actual")],
)
def test_a_current_receipt_from_an_earlier_runtime_source_is_refused_first(key, stage):
    """The runtime-source refusal comes before any request or result is read."""
    record = record_runtime_source(stage)
    record["schema_version"] = "scgsim.aedt.runtime-source.v29"
    with pytest.raises(
        RuntimeError,
        match=rf"{stage} runtime source scgsim\.aedt\.runtime-source\.v29 was written",
    ):
        require_current_receipt({"schema_version": RECEIPT_V3, key: record})


def _epr_request_payload(tmp_path):
    plan = produce_request(tmp_path / "request")
    return json.loads(plan.spec_path.read_text())


@pytest.mark.parametrize(
    "schema",
    [
        "scgsim.aedt.hfss-eigenmode-epr.v1",
        "scgsim.aedt.hfss-eigenmode-epr-analysis.v1",
    ],
)
def test_an_earlier_epr_request_schema_is_unsupported(tmp_path, schema):
    """The v1 EPR request and saved-field analysis schemas are not decoded."""
    payload = {**_epr_request_payload(tmp_path), "schema_version": schema}
    with pytest.raises(ValueError, match="unsupported AEDT schema"):
        parse_aedt_spec(payload, base_dir=tmp_path)


def test_a_request_without_a_physical_representation_is_refused(tmp_path):
    """Requests written before the neutral physical plan are not decoded."""
    payload = _epr_request_payload(tmp_path)
    del payload["physical_representation"]
    with pytest.raises(
        ValueError, match="require a tagged neutral physical_representation"
    ):
        parse_aedt_spec(payload, base_dir=tmp_path)


def test_an_earlier_planar_geometry_schema_is_refused():
    """The v1 prepared planar geometry payload is not decoded."""
    payload = file_only_geometry().to_payload()
    payload["schema_version"] = "scgsim.aedt.epr-planar.v1"
    with pytest.raises(ValueError, match="prepared planar payload is not canonical"):
        PreparedPlanarGeometry.from_payload(payload)


def test_records_without_their_current_members_are_refused():
    """The v1 member sets of a surface contribution and an EPR request are refused."""
    surface = SurfaceEprSpec("MM1", "poly", "MM", "top").to_payload()
    for name in ("loss_tangent", "source", "preset"):
        del surface[name]
    with pytest.raises(
        ValueError, match="surface EPR payload members are not canonical"
    ):
        SurfaceEprSpec.from_payload(surface)
    request = EprAnalysisRequest().to_payload()
    del request["bulk_domain_ids"]
    with pytest.raises(
        ValueError, match="EPR analysis request members are not canonical"
    ):
        EprAnalysisRequest.from_payload(request)


def test_an_earlier_epr_result_is_refused_with_how_to_read_it(tmp_path):
    """A v1 EPR result file is not loaded."""
    path = tmp_path / "epr-result.json"
    write_json(
        path,
        {
            "schema_version": "scgsim.aedt.epr-result.v1",
            "result_kind": "saved_field",
            "setup_name": "Setup",
            "rows": [],
            "provenance": {},
        },
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"EPR result scgsim\.aedt\.epr-result\.v1 was written by an earlier "
            "SCGSim.*read this result with the SCGSim commit that wrote it"
        ),
    ):
        resolve_epr_result(path)


def test_an_earlier_saved_solution_manifest_is_refused_with_how_to_read_it(tmp_path):
    """A v1 saved-solution manifest is not resolved."""
    path = tmp_path / "saved-solution-manifest.json"
    write_json(path, {"schema_version": "scgsim.aedt.saved-solution-manifest.v1"})
    with pytest.raises(
        RuntimeError,
        match=(
            r"saved-solution manifest scgsim\.aedt\.saved-solution-manifest\.v1 "
            "was written by an earlier SCGSim.*with the SCGSim commit that wrote it"
        ),
    ):
        resolve_saved_solution(path)
