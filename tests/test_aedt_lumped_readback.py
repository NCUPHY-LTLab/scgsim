"""Offline regressions for saved native lumped-RLC readback checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scgsim.aedt.results.hfss_readback import validate_lumped_readback
from scgsim.aedt.specs.common import LumpedRlc


def _lumped_receipt():
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    support = {
        "support_id": "support",
        "source": {"support_id": "support", "kind": "neutral"},
        "effective": {"contact_points_um": points},
        "object_name": "support_sheet",
        "native_object_id": 10,
        "native_face_ids": [20],
    }
    treatment = LumpedRlc("support", "series", resistance_ohm=2.0)
    requested = {
        "topology": treatment.topology,
        "native_topology": "Serial",
        "contact_points_um": points,
        "resistance_ohm": treatment.resistance_ohm,
        "inductance_h": treatment.inductance_h,
        "capacitance_f": treatment.capacitance_f,
    }
    record = {
        "support_id": "support",
        "source": treatment.to_payload(),
        "requested": requested,
        "object_name": support["object_name"],
        "native_object_id": support["native_object_id"],
        "native_face_ids": support["native_face_ids"],
        "assignment": {
            "covered_objects": [support["object_name"]],
            "covered_faces": [20],
        },
        "saved_boundary": {
            "BoundType": "Lumped RLC",
            "CurrentLine": {
                "GeometryPosition": [
                    {
                        "IsAttachedToEntity": False,
                        "PositionType": "AbsolutePosition",
                        "XPosition": point[0],
                        "YPosition": point[1],
                        "ZPosition": point[2],
                    }
                    for point in points
                ]
            },
        },
        "saved_model_units": "um",
        "saved_project_sha256": "a" * 64,
        "saved_contact_points_um": points,
        "native_properties": {
            "RLC Type": "Serial",
            "Use Resist": True,
            "Resistance": "2ohm",
            "Use Induct": False,
            "Use Cap": False,
        },
        "normalized_values_si": {"resistance_ohm": "2"},
    }
    prepared = SimpleNamespace(
        source={"lumped_supports": [support]},
        source_sha256="source-hash",
        lumped_rlcs=(treatment,),
    )
    receipt = {
        "geometry": {
            "source_sha256": "source-hash",
            "lumped_supports": [
                {
                    **support,
                    "native_object_type": "Sheet",
                }
            ],
            "lumped_rlcs": [record],
        }
    }
    return receipt, prepared


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("topology", "generic RLC native topology differs"),
        ("enabled", "generic RLC enable differs"),
        ("scalar", "generic RLC native component value differs"),
        ("normalized", "generic RLC native component value differs"),
    ),
)
def test_saved_lumped_readback_rejects_native_component_drift(damage, message):
    """Reject saved topology, enable flags, raw values, and normalized values."""
    receipt, prepared = _lumped_receipt()
    geometry = receipt["geometry"]
    record = geometry["lumped_rlcs"][0]
    if damage == "topology":
        record["native_properties"]["RLC Type"] = "Parallel"
    elif damage == "enabled":
        record["native_properties"]["Use Resist"] = False
    elif damage == "scalar":
        record["native_properties"]["Resistance"] = "3ohm"
    else:
        record["normalized_values_si"]["resistance_ohm"] = "3"

    with pytest.raises(RuntimeError, match=message):
        validate_lumped_readback(receipt, prepared)
