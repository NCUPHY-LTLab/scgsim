"""Q2D model-only handoff contract; no native project is represented."""

from __future__ import annotations

import json

import pytest

from scgsim.aedt import (
    MatrixRunControl,
    PdkMaterial,
    Q2dConductorSpec,
    Q2dRectangleSpec,
    Q2dSpec,
    prepare_handoff,
)
from scgsim.aedt.runtime.families.q2d import run_q2d
from scgsim.aedt.specs.common import Q2D_SCHEMA_VERSION, Q2D_SCHEMA_VERSION_V2
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.aedt.results.provenance import SOURCE_SCHEMA_V25, validate_runtime_source


def _spec(version: str, run_control: MatrixRunControl | None) -> Q2dSpec:
    return Q2dSpec(
        aedt_version=version,
        project_name="PreparedQ2d",
        design_name="Q2DDesign1",
        materials={
            "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
            "silicon": PdkMaterial("silicon", "dielectric", False, "Silicon"),
            "metal": PdkMaterial("metal", "superconductor", True, None),
        },
        vacuum_material_id="vacuum",
        rectangles=(
            Q2dRectangleSpec("Substrate", (0.0, -10.0), (20.0, 10.0), "silicon"),
            Q2dRectangleSpec("SignalObject", (8.0, 0.0), (4.0, 0.2), "metal"),
            Q2dRectangleSpec("GroundObject", (0.0, 0.0), (4.0, 0.2), "metal"),
        ),
        conductors=(
            Q2dConductorSpec("Signal", "SignalLine", ("SignalObject",), 0.2),
            Q2dConductorSpec("Ground", "ReferenceGround", ("GroundObject",), 0.2),
        ),
        run_control=run_control,
        region_padding_um=(10.0, 10.0, 10.0, 10.0),
    )


@pytest.mark.parametrize("version", ("2023.2", "2024.2"))
def test_q2d_model_only_handoff_roundtrips_explicit_null_without_native_output(
    tmp_path, version
):
    plan = prepare_handoff(
        spec=_spec(version, None), output_dir=tmp_path / f"q2d-{version}"
    )

    payload = json.loads(plan.spec_path.read_text(encoding="utf-8"))
    parsed = parse_aedt_spec(payload, base_dir=plan.run_dir)
    receipt = json.loads(plan.receipt_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == Q2D_SCHEMA_VERSION_V2
    assert payload["run_control"] is None
    assert parsed.to_payload() == payload
    assert parsed.aedt_version == version
    assert receipt["status"] == "not_run"
    assert receipt["prepared_runtime_source"]["schema_version"] == SOURCE_SCHEMA_V25
    validate_runtime_source(receipt["prepared_runtime_source"], stage="prepared")
    assert not (plan.run_dir / "PreparedQ2d.aedt").exists()
    assert not (plan.run_dir / "results").exists()


def test_q2d_model_only_payload_requires_explicit_null_run_control():
    payload = _spec("2024.2", None).to_payload()
    assert payload["schema_version"] == Q2D_SCHEMA_VERSION_V2

    missing = dict(payload)
    del missing["run_control"]
    with pytest.raises(ValueError, match="explicitly declare run_control"):
        Q2dSpec.from_payload(missing)

    with_control = {
        **payload,
        "run_control": MatrixRunControl("Setup1", 6.0, 4, 0.1).to_payload(),
    }
    with pytest.raises(ValueError, match="model-only null run_control"):
        Q2dSpec.from_payload(with_control)


def test_model_only_q2d_cannot_enter_the_solver(tmp_path):
    with pytest.raises(ValueError, match="requires an explicit complete run_control"):
        run_q2d(None, tmp_path, _spec("2024.2", None))


def test_controlled_q2d_payload_remains_v1():
    spec = _spec("2024.2", MatrixRunControl("Setup1", 6.0, 4, 0.1))
    payload = spec.to_payload()
    assert payload["schema_version"] == Q2D_SCHEMA_VERSION
    assert isinstance(payload["run_control"], dict)
    assert Q2dSpec.from_payload(payload).to_payload() == payload
