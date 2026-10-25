"""Shared prepared-cache codec tests use files and producer CLC, never AEDT."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import FrozenInstanceError

import pytest

from scgsim.aedt.epr.evidence import PreparedEprEvidence, SavedCacheEvidence
from scgsim.aedt.epr.fields import compile_named_expression, _parse_native_expression_readback


def _prepared(tmp_path):
    expression, definitions, _ = compile_named_expression(
        purpose="file_evidence", operations=["Scalar_Constant(1)"],
        solution="Setup1 : LastAdaptive", phase_degrees=0,
        dependencies={}, selection={},
    )
    payload = "".join(block for _, block in definitions).encode()
    (tmp_path / "saved.clc").write_bytes(payload)
    expression["definition_status"] = "native_readback_verified"
    expression["native_readback"] = {
        "status": "native_clc_saved_and_field_selection_read_back",
        "api": "FieldsReporter.SaveNamedExpressions",
        "path": "metadata/epr_expressions/saved.clc",
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "expression_definition_names": [name for name, _ in definitions],
        "definitions": _parse_native_expression_readback(payload),
    }
    item = {"title": "Required", "expression": expression["name"],
            "intrinsics": "Phase='0deg'", "is_convergence": False}
    observation = {
        "saved_configuration_source": {"project_sha256": "readback-time-hash"},
        "use_cache_for": ["Pass"],
        "items": [{key: item[key] for key in ("title", "expression", "intrinsics")}],
    }
    receipt = {
        "setup": {"name": "Setup1", "requested_delta": 0.02},
        "cache": {"items": [item], "serialized_readback": observation},
        "expressions": [expression],
        "save": {"project_sha256": "different-final-save-hash"},
    }
    return receipt


def test_prepared_roundtrip_preserves_existing_three_wire_fields(tmp_path):
    receipt = _prepared(tmp_path)
    evidence = PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path)
    assert evidence.to_payload() == {key: receipt[key] for key in ("setup", "cache", "expressions")}
    assert set(evidence.required_definitions) == {receipt["expressions"][0]["name"]}
    assert evidence.required_definitions_payload() == receipt["expressions"][0]["native_readback"]["definitions"]


def test_missing_optional_observations_stay_missing(tmp_path):
    receipt = _prepared(tmp_path)
    saved = PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path).saved_cache.to_payload()
    assert "frequency_control" not in saved
    assert "convergence_targets" not in saved
    assert "convergence_fields" not in saved
    assert saved["saved_configuration_source"]["project_sha256"] != receipt["save"]["project_sha256"]


def test_requested_and_observed_controls_remain_distinct_raw_data(tmp_path):
    receipt = _prepared(tmp_path)
    saved = receipt["cache"]["serialized_readback"]
    saved["frequency_control"] = {
        "requested": {"max_delta_frequency": 0.02},
        "observed_fields": {"MaxDeltaFreq": "2"},
        "unavailable": ["MaximumPasses"],
    }
    saved["convergence_targets"] = [{
        "requested_native_fields": {"IsConvergence": True},
        "observed_fields": {"values": {}, "unavailable": ["IsConvergence"]},
        "raw_representation_differences": [],
    }]
    saved["other_observation"] = {"missing": None, "raw": [True, 0, "unknown"]}
    evidence = PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path)
    assert evidence.saved_cache.to_payload() == saved
    assert evidence.to_payload()["cache"] == receipt["cache"]


def test_frozen_codec_has_no_mutable_input_or_output_alias(tmp_path):
    receipt = _prepared(tmp_path)
    expected = copy.deepcopy(receipt)
    evidence = PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path)
    receipt["cache"]["items"][0]["title"] = "changed"
    detached = evidence.to_payload()
    detached["cache"]["items"][0]["title"] = "also changed"
    assert evidence.to_payload()["cache"] == expected["cache"]
    with pytest.raises(TypeError):
        evidence.cache["other"] = None
    with pytest.raises(TypeError):
        evidence.saved_cache.observation["other"] = None
    with pytest.raises(FrozenInstanceError):
        evidence.setup = {}


@pytest.mark.parametrize("missing", ["setup", "cache", "expressions"])
def test_required_preparation_fields_remain_required(tmp_path, missing):
    receipt = _prepared(tmp_path)
    del receipt[missing]
    with pytest.raises(RuntimeError):
        PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path)


@pytest.mark.parametrize("damage", ["empty_items", "source", "serialized", "use_cache_for", "persisted", "required_clc"])
def test_shared_consumer_rejects_existing_evidence_failures(tmp_path, damage):
    receipt = _prepared(tmp_path)
    cache = receipt["cache"]
    if damage == "empty_items":
        cache["items"] = []
    elif damage == "source":
        cache["serialized_readback"]["saved_configuration_source"] = {}
    elif damage == "serialized":
        del cache["serialized_readback"]
    elif damage == "use_cache_for":
        cache["serialized_readback"]["use_cache_for"] = ["Iteration"]
    elif damage == "persisted":
        cache["serialized_readback"]["items"][0]["expression"] = "Changed"
    else:
        (tmp_path / "saved.clc").write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path)


def test_saved_cache_codec_is_the_same_producer_consumer_interpretation(tmp_path):
    receipt = _prepared(tmp_path)
    cache = receipt["cache"]
    output = SavedCacheEvidence.from_payload(cache["items"], cache["serialized_readback"]).to_payload()
    cache["serialized_readback"] = output
    consumer = PreparedEprEvidence.from_payload(receipt, evidence_root=tmp_path)
    assert consumer.saved_cache.to_payload() == output
