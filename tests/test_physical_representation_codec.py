"""Exercise canonical physical-plan reconstruction with real source declarations."""

from dataclasses import replace

import pytest

from scgsim.geometry import interpret_physical_layers as facade_interpreter
from scgsim.geometry.interpretation.physical import (
    interpret_physical_layers as planning_interpreter,
)
from scgsim.geometry.models.physical import (
    ApproximationProfile,
    PhysicalLayerDeclaration,
    PhysicalRepresentationPlan,
    interpret_physical_layers,
    source_plain,
)
from scgsim.semantics.snapshot import canonical_sha256, freeze_plain

PROFILE_NAMES = (
    "solid",
    "thin_film",
    "substrate_face",
    "metal_gap_equivalent",
    "route_b",
)


def _declaration(physical_id, interval, feature_kind, side=None):
    return PhysicalLayerDeclaration(
        physical_layer_id=physical_id,
        source_level_id=physical_id,
        source_level_and_gds_pair={"level": physical_id, "gds_pair": [1, 0]},
        source_interval_um=interval,
        material_id="metal",
        feature_kind=feature_kind,
        physical_dimensions={"normal_thickness_um": interval[1] - interval[0]}
        if side
        else {"axial_span_um": interval[1] - interval[0]},
        source_contact_facts={"layer_attachment_face": side} if side else {},
    )


def _source_records():
    # Authored order differs from collapse order; both must survive reconstruction.
    declarations = (
        _declaration("upper", (3.8, 4.0), "stack_film", "upper"),
        _declaration("local", (6.0, 6.1), "local_film", "upper"),
        _declaration("lower", (0.0, 0.2), "stack_film", "lower"),
        _declaration("volume", (0.5, 3.0), "finite_volume"),
    )
    supports = {
        name: {
            "geometry": {"z_min_um": low, "z_max_um": high},
            "material_id": "Si",
            "material_kind": "dielectric",
            "material_properties": {"permittivity": 11.45},
        }
        for name, low, high in (("bottom_host", -2.0, 0.0), ("top_host", 4.0, 6.0))
    }
    return declarations, supports


def _profile(name):
    parameters = {}
    if name in {"substrate_face", "metal_gap_equivalent"}:
        parameters = {
            "film_bindings": {
                "lower": {
                    "supporting_entity_id": "bottom_host",
                    "supporting_side": "upper",
                },
                "upper": {
                    "supporting_entity_id": "top_host",
                    "supporting_side": "lower",
                },
            }
        }
    return ApproximationProfile(name, "v1", parameters)


def _plan(name, *, frozen=False):
    declarations, supports = _source_records()
    profile = _profile(name)
    if frozen:
        supports = freeze_plain(supports)
    plan = interpret_physical_layers(
        declarations, profile, supporting_entities=supports
    )
    entities = {}
    for item in declarations:
        interval = (
            (1.0, 2.0)
            if item.feature_kind == "finite_volume"
            else item.source_interval_um
        )
        entities[item.physical_layer_id] = plan.map_body(
            item.physical_layer_id,
            interval,
            {
                "z_min_um": interval[0],
                "z_max_um": interval[1],
                "thickness_um": interval[1] - interval[0],
                "material_id": "metal",
                "outer_loop": [[0, 0], [3, 0], [3, 3], [0, 3]],
                "hole_loops": [[[1, 1], [2, 1], [2, 2], [1, 2]]],
            },
        )
    bounds = {
        "x_min_um": 0.0,
        "x_max_um": 3.0,
        "y_min_um": 0.0,
        "y_max_um": 3.0,
        "z_min_um": 0.0,
        "z_max_um": 6.1,
    }
    effective_bounds = {**bounds, "z_max_um": plan.map_z(bounds["z_max_um"])}
    padding = {f"{axis}_{side}_um": 1.0 for axis in "xyz" for side in ("minus", "plus")}
    enclosure = plan.project_enclosure(
        effective_body_bounds_um=effective_bounds,
        source_padding_um=padding,
        source_body_bounds_um=bounds,
    )
    return replace(plan, entities=entities, enclosures={"vacuum": enclosure})


@pytest.mark.parametrize("name", PROFILE_NAMES)
@pytest.mark.parametrize("frozen", [False, True], ids=["plain", "frozen"])
def test_payload_and_plan_id_round_trip(name, frozen):
    """Preserve all selected-profile records through plain and frozen payloads."""
    plan = _plan(name, frozen=frozen)
    payload = plan.to_payload()
    encoded = freeze_plain(payload) if frozen else payload

    restored = PhysicalRepresentationPlan.from_payload(encoded)

    assert restored.to_payload() == payload
    assert restored.plan_id == plan.plan_id
    assert [item.physical_layer_id for item in restored.declarations] == [
        "upper",
        "local",
        "lower",
        "volume",
    ]
    assert restored.entities["volume"]["source_interval_um"] == (1.0, 2.0)
    assert [item[0] for item in restored.collapse_intervals] == (
        ["lower", "upper"] if name in {"thin_film", "metal_gap_equivalent"} else []
    )


def test_interpreter_compatibility_imports_are_the_same_function():
    """Keep the existing facades bound to the model-owned interpreter."""
    assert planning_interpreter is interpret_physical_layers
    assert facade_interpreter is interpret_physical_layers
    assert interpret_physical_layers.__module__ == "scgsim.geometry.models.physical"


@pytest.mark.parametrize("source_hash", [None, "", "authored-source-hash"])
def test_source_hash_preserves_existing_truth_value_selection(source_hash):
    """Retain canonical hashing for omitted or empty source hashes."""
    declarations, _ = _source_records()
    plan = interpret_physical_layers(
        declarations, _profile("solid"), source_sha256=source_hash
    )
    assert plan.source_sha256 == (
        source_hash or canonical_sha256([item.to_payload() for item in declarations])
    )


def test_source_plain_preserves_dataclass_and_frozen_container_conversion():
    """Normalize authored dataclasses and immutable source containers identically."""
    declaration = _source_records()[0][0]
    assert source_plain(declaration) == declaration.to_payload()
    assert source_plain(freeze_plain({"declarations": [declaration.to_payload()]})) == {
        "declarations": [declaration.to_payload()]
    }


def test_undeclared_body_retains_its_source_geometry_in_the_identity_plan():
    """Keep finite solid geometry usable without a minted physical declaration."""
    plan = interpret_physical_layers((), _profile("solid"))
    geometry = {"z_min_um": 1.0, "z_max_um": 2.0, "thickness_um": 1.0}

    body = plan.map_body(None, (1.0, 2.0), geometry)

    assert body["physical_layer_id"] is None
    assert body["representation"] == "solid"
    assert body["source_geometry"] == body["effective_geometry"] == geometry
    assert body["physical_thickness_um"] == 1.0
    assert body["physical_thickness_basis"] == "source_body_axial"


@pytest.mark.parametrize("provide_both", [False, True], ids=["neither", "both"])
def test_enclosure_requires_exactly_one_source_bounds_record(provide_both):
    """Reject ambiguous or absent source bounds before projecting coordinates."""
    plan = interpret_physical_layers((), _profile("solid"))
    bounds = {} if provide_both else None

    with pytest.raises(
        ValueError,
        match="provide exactly one original body or full-envelope bounds record",
    ):
        plan.project_enclosure(
            effective_body_bounds_um={},
            source_padding_um={},
            source_body_bounds_um=bounds,
            source_envelope_bounds_um=bounds,
        )


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "extra_key",
        "missing_key",
        "collapse",
        "anchor",
        "support",
        "body",
        "enclosure",
    ],
)
def test_tampered_payload_is_rejected(change):
    """Reject changes to the schema, canonical members or derived projections."""
    payload = _plan("metal_gap_equivalent").to_payload()
    if change == "schema":
        payload["schema_version"] = "unknown"
    elif change == "extra_key":
        payload["unexpected"] = None
    elif change == "missing_key":
        del payload["entities"]
    elif change == "collapse":
        payload["collapse_intervals"][0][2] += 1.0
    elif change == "anchor":
        payload["sheet_anchors_um"]["lower"] += 1.0
    elif change == "support":
        payload["supporting_entities"]["unused"] = payload["supporting_entities"][
            "bottom_host"
        ]
    elif change == "body":
        payload["entities"]["volume"]["effective_interval_um"][1] += 1.0
    elif change == "enclosure":
        payload["enclosures"]["vacuum"]["effective_bounds_um"]["z_max_um"] += 1.0

    with pytest.raises(ValueError):
        PhysicalRepresentationPlan.from_payload(payload)
