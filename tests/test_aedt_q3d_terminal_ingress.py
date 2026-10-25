"""Real offline Q3D source lowering; no Desktop or native-operation substitute."""

import hashlib
import json

import gdstk
import pytest

from scgsim.aedt import PdkMaterial, Q3dTerminalPair
from scgsim.aedt.preparation.q3d_geometry import lower_q3d_geometry
from scgsim.geometry import (
    GeometryBuildInput,
    GeometryPlanSnapshot,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)


def _snapshot(tmp_path, *, multibody=False):
    polygons = []
    entities = []
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("TerminalIngress")
    for ordinal, (identity, net) in enumerate(
        (("source_a", "NetA"), ("sink_a", "NetA"), ("ground_b", "NetB"))
    ):
        x = 4 * ordinal
        outer = ((x, 0), (x + 2, 0), (x + 2, 2), (x, 2))
        polygon = LayoutPolygonSpec(identity + "_polygon", identity, outer)
        polygons.append(polygon)
        cell.add(gdstk.Polygon(outer, layer=ordinal + 1))
        entity_polygon_ids = (polygon.polygon_id,)
        if multibody and identity == "source_a":
            second_outer = ((0, 4), (2, 4), (2, 6), (0, 6))
            second = LayoutPolygonSpec("source_a_second", identity, second_outer)
            polygons.append(second)
            cell.add(gdstk.Polygon(second_outer, layer=ordinal + 1))
            entity_polygon_ids += (second.polygon_id,)
        entities.append(
            SemanticEntitySpec(
                identity, "conductor", "metal", "conductor", 1, "polygon",
                net_id=net, polygon_ids=entity_polygon_ids,
                geometry={"outer_loop": outer, "z_min_um": 0,
                          "z_max_um": 1, "thickness_um": 1},
            )
        )
    gds_path = tmp_path / "authored.gds"
    library.write_gds(gds_path)
    payload = gds_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {"gds_sha256": digest, "source_dbu_um": 0.001,
                "source_occurrences": []}
    source = GeometryBuildInput(tuple(polygons), tuple(entities), metadata=metadata)
    stack = {
        "layers": [{"semantic_id": e.semantic_id, "geometry": dict(e.geometry),
                    "metadata": {}} for e in entities],
        "solution_regions": {},
        "materials": {"metal": {"kind": "conductor"},
                      "vacuum": {"kind": "vacuum", "permittivity": 1.0}},
        "metadata": metadata,
    }
    return GeometryPlanSnapshot(source, json.dumps(stack), payload, digest, ())


@pytest.fixture
def snapshot(tmp_path):
    return _snapshot(tmp_path)


def _lower(snapshot, directory, **kwargs):
    return lower_q3d_geometry(
        snapshot, aedt_version="2024.2", modeling="solid", directory=directory,
        project_name="Terminals", design_name="Design",
        materials={"metal": PdkMaterial("metal", "superconductor", True, None),
                   "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum")},
        net_types={"NetA": "Signal", "NetB": "Ground"},
        physical_ground_nets=("NetB",), region_padding_um=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


def test_exact_source_entities_bind_global_z_sides_and_keep_body_provenance(snapshot, tmp_path):
    pair = Q3dTerminalPair("source_a", "-Z", "sink_a", "+Z")
    spec = _lower(snapshot, tmp_path / "paired", terminal_pairs={"NetA": pair})
    by_source = {body.source_entity_id: body for body in spec.bodies}
    signal = next(net for net in spec.nets if net.name == "NetA")
    assert (signal.source_object, signal.source_side) == (by_source["source_a"].body_id, "-Z")
    assert (signal.sink_object, signal.sink_side) == (by_source["sink_a"].body_id, "+Z")
    assert by_source["source_a"].source_polygon_id == "source_a_polygon"
    assert by_source["sink_a"].source_polygon_id == "sink_a_polygon"
    assert by_source["source_a"].net_id == by_source["sink_a"].net_id == "NetA"
    assert spec.solve_ac_rl is False
    assert spec.geometry_source["bodies"] == [body.to_payload() for body in spec.bodies]
    trace = json.loads((tmp_path / "paired" / "geometry_trace.json").read_text())
    assert trace["source_geometry"]["entities"][0]["semantic_id"] == "source_a"


def test_omitted_none_and_empty_pairs_preserve_cg_payload(snapshot, tmp_path):
    directory = tmp_path / "cg"
    baseline = _lower(snapshot, directory).to_payload()
    assert _lower(snapshot, directory, terminal_pairs=None).to_payload() == baseline
    assert _lower(snapshot, directory, terminal_pairs={}).to_payload() == baseline
    for net in baseline["nets"]:
        assert net["source_object"] is None and net["source_side"] is None
        assert net["sink_object"] is None and net["sink_side"] is None


@pytest.mark.parametrize("entity", ["absent", "source_a_polygon"])
def test_terminal_requires_exact_entity_identity_not_polygon_or_guessed_body(snapshot, tmp_path, entity):
    pair = Q3dTerminalPair(entity, "-Z", "sink_a", "+Z")
    with pytest.raises(ValueError, match="has no emitted body"):
        _lower(snapshot, tmp_path / "missing", terminal_pairs={"NetA": pair})


def test_multibody_source_entity_is_explicitly_unsupported(snapshot, tmp_path):
    actual = _snapshot(tmp_path, multibody=True)
    with pytest.raises(ValueError, match="maps to multiple bodies"):
        _lower(actual, tmp_path / "multiple", terminal_pairs={
            "NetA": Q3dTerminalPair("source_a", "-Z", "sink_a", "+Z")})


def test_cross_net_entity_cannot_be_used_as_signal_terminal(snapshot, tmp_path):
    with pytest.raises(ValueError, match="must belong to their net"):
        _lower(snapshot, tmp_path / "crossnet", terminal_pairs={
            "NetA": Q3dTerminalPair("source_a", "-Z", "ground_b", "+Z")})


def test_ground_net_cannot_accept_terminal_pair(snapshot, tmp_path):
    with pytest.raises(ValueError, match="Ground nets must not define"):
        _lower(snapshot, tmp_path / "ground", terminal_pairs={
            "NetB": Q3dTerminalPair("ground_b", "-Z", "ground_b", "+Z")})


def test_pair_key_must_select_existing_final_net(snapshot, tmp_path):
    with pytest.raises(ValueError, match="select final conductor Nets"):
        _lower(snapshot, tmp_path / "unknown_net", terminal_pairs={
            "Unknown": Q3dTerminalPair("source_a", "-Z", "sink_a", "+Z")})


@pytest.mark.parametrize("source_side,sink_side", [("Z", "+Z"), ("-Z", "up"), (None, "+Z")])
def test_invalid_global_sides_fail_at_typed_request(source_side, sink_side):
    with pytest.raises(ValueError, match="sides are invalid"):
        Q3dTerminalPair("source_a", source_side, "sink_a", sink_side)
