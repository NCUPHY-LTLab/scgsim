"""Results an earlier SCGSim wrote are refused with how to read them.

An old result is read with the SCGSim commit that wrote it; this version reads
only the formats it writes.
"""

from __future__ import annotations

import pytest

from scgsim.aedt.io import write_json
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    record_runtime_source,
    require_current_receipt,
)
from scgsim.aedt.results.resolve import resolve_results


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
