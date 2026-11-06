"""Component to ``palace.msh`` and config through SGB, the meshing module and Palace's lowering (needs Gmsh).

Every case runs ``build_route_mesh`` for real and then builds the config from the groups it returned.  No
Palace binary runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
from palace_sgb_support import (
    MATERIALS,
    effective_input,
    mesh_run,
    surface_epr_specs,
)
from sgb_inputs import EffectiveInput, gds_pad_in_ground, rectangular_junction_pair
from test_sgb_curved_port_landing import circular_pad_input

from scgsim.meshing import Budgets, MesherOptions, MeshRepairError
from scgsim.palace.mesh.native import lower_mesh_in_current_process
from scgsim.palace.mesh.quality.reader import read_msh22
from scgsim.palace.mesh.summary import mesh_summary
from scgsim.palace.preparation.config import (
    LayoutPortBinding,
    TerminalBinding,
    build_eigenmode_config,
    build_electrostatic_config,
    configure_numerical_controls,
)
from scgsim.palace.presentation.plotting import mesh_summary_html
from scgsim.palace.results.index_contract import (
    INDEX_MAP_SCHEMA_VERSION,
    RouteConsistency,
    validate_config_index_correspondence,
)
from scgsim.palace.results.tables import count_index_entries

pytestmark = pytest.mark.slow

SPECS = surface_epr_specs(MA=[0.5, 1.0], MS=[0.5], SA=[0.5])


def check_run(result, route):
    """Everything every run must satisfy: layout, MSH 2.2, attributes in the mesh, manifest, PEC and Terminal."""
    run = result.output_dir
    assert (run / "palace.msh").is_file()
    assert sorted(
        p.name for p in (run / "geometry").glob("semantic_geometry_route_*")
    ) == [
        f"semantic_geometry_route_{route.lower()}.{suffix}"
        for suffix in ("brep", "sidecar.json", "slivers.json", "validation.json", "xao")
    ]
    assert sorted(p.name for p in (run / "mesh").iterdir()) == [
        f"semantic_geometry_route_{route.lower()}.quality.json",
        f"semantic_geometry_route_{route.lower()}.quality.txt",
    ]
    assert not (run / "metadata" / "semantic_geometry").exists()
    assert not (run / "geometry" / ".unbanded").exists()
    mesh = read_msh22(run / "palace.msh")
    assert mesh.physical_names
    manifest = json.loads(result.mesh_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["source"]["scgsim_sgb"] == "scgsim.geometry.sgb"
    assert manifest["source"]["sidecar_schema"] == "scgsim.sgb.sidecar.v2"
    assert manifest["source"]["model_identity_digest"].startswith("model:")
    meshing = manifest["meshing"]
    assert meshing["recipe"]["name"] == "features"
    assert meshing["threshold"] == 0.01
    report = run / meshing["quality_report"]["path"]
    assert (
        hashlib.sha256(report.read_bytes()).hexdigest()
        == meshing["quality_report"]["sha256"]
    )
    # The manifest maps each primary group to its attributes and these are the physical groups of the mesh.
    listed = {
        (e["dimension"], e["attributes"][0]): e["name"] for e in manifest["groups"]
    }
    assert listed == {key: name for key, name in mesh.physical_names.items()}
    assert {e["group"] for e in manifest["groups"]} >= set(result.groups["volumes"])
    return manifest, mesh


def config_of(result, route, specs=None):
    """The Electrostatic config of a run: first Net as Terminal, second as Ground."""
    nets = sorted({r["net"] for r in result.groups["pec_surfaces"].values()})
    return build_electrostatic_config(
        groups=result.groups,
        terminals=[TerminalBinding("t", nets[0])],
        ground_net_ids=nets[1:],
        materials=MATERIALS,
        save_fields=0,
        unassigned_conductor_policy="error",
        exterior_boundary_policy="none",
        mesh_path=result.mesh_path,
        surface_epr_specs=specs or {},
        numerical=configure_numerical_controls(),
    )


def config_attributes(config):
    """Every (dimension, attribute) a config refers to."""
    boundary = set()
    section = config["Boundaries"]
    for row in section.get("Terminal", ()):
        boundary.update(row["Attributes"])
    boundary.update(section.get("Ground", {}).get("Attributes", ()))
    boundary.update(section.get("PEC", {}).get("Attributes", ()))
    for row in section.get("LumpedPort", ()):
        boundary.update(row["Attributes"])
    for row in section.get("Postprocessing", {}).get("Dielectric", ()):
        boundary.update(row["Attributes"])
    domains = {a for m in config["Domains"]["Materials"] for a in m["Attributes"]}
    return {(2, a) for a in boundary}, {(3, a) for a in domains}


def assert_config_binds_the_mesh(config, mesh):
    """Every attribute of the config is a physical group of the mesh, in its dimension."""
    surfaces, volumes = config_attributes(config)
    assert (surfaces | volumes) <= set(mesh.physical_names)


@pytest.fixture(scope="module")
def route_b(tmp_path_factory):
    """Route B pad in ground with bands on MA, MS and SA."""
    directory = tmp_path_factory.mktemp("route_b")
    effective = effective_input("B", directory)
    return mesh_run(effective, "B", directory, specs=SPECS)


@pytest.fixture(scope="module")
def route_a(tmp_path_factory):
    """Route A thin-film pair with bands on MA, MS and SA."""
    directory = tmp_path_factory.mktemp("route_a")
    return mesh_run(effective_input("A", directory), "A", directory, specs=SPECS)


@pytest.mark.parametrize("route", ["A", "B"])
def test_a_component_meshes_to_a_palace_mesh_and_a_config(route, route_a, route_b):
    """Routes A and B: mesh, manifest and config agree, with bands as separate groups."""
    result = {"A": route_a, "B": route_b}[route]
    manifest, mesh = check_run(result, route)
    names = set(mesh.physical_names.values())
    assert any("::band=" in name for name in names)
    config = config_of(result, route, SPECS)
    assert_config_binds_the_mesh(config.config, mesh)
    assert manifest["route"] == route
    assert manifest["route_policy"]["conductor_sheets"] is (route == "A")
    # The unbanded build used to find the supports is gone, and only the final model is on disk.
    assert manifest["insets"]
    assert_readers_accept_the_run(result, route, manifest, config)


def assert_readers_accept_the_run(result, route, manifest, config):
    """The manifest the build wrote and the index map of its config satisfy the readers, and render."""
    entries = [
        *config.terminal_index_map,
        *config.domain_energy_index_map,
        *config.surface_epr_index_map,
    ]
    index = {
        "schema_version": INDEX_MAP_SCHEMA_VERSION,
        "entries": entries,
        "ground_boundary": config.ground_boundary_resolution,
    }
    metadata = {
        "route": route,
        "problem": "Electrostatic",
        "route_a_thin_film": manifest.get("route_a_thin_film"),
    }
    RouteConsistency(
        route,
        metadata,
        metadata,
        manifest,
        index,
        count_index_entries(index),
        "Electrostatic",
    ).validate()
    validate_config_index_correspondence(
        config.config, index, count_index_entries(index), "Electrostatic"
    )
    summary = mesh_summary(manifest, problem="Eigenmode", fem_order=2)
    html = mesh_summary_html(
        summary, card_html=lambda label, value: f"{label}: {value}"
    )
    assert "features" in html and "Quality threshold" in html
    assert summary["dof_estimate"]["status"] == "available"


def test_route_b_gets_an_ma_ms_and_sa_band_family_per_support(route_b):
    """The bands requested from the specs: MA tops, MS bottoms and the SA interface."""
    families = {
        i["request_id"] for i in json.loads(route_b.sidecar_path.read_text())["insets"]
    }
    assert len(families) == 5
    assert any("VACUUM_REGION" in f for f in families)


def test_route_a_attributes_split_where_the_row_signature_changes(route_a):
    """Route A sheet bands carry the MA row (above) and the MS row (below) on one attribute each."""
    left_bands = [
        record
        for name, record in route_a.groups["boundary_surfaces"].items()
        if name.startswith("chip/left::band=")
    ]
    assert len(left_bands) == 3
    for record in left_bands:
        (part,) = record["parts"]
        assert {r["side"] for r in part["rows"]} == {"above", "below"}
    config = config_of(route_a, "A", SPECS).config
    assert config["Model"]["CrackInternalBoundaryElements"] is True


def test_attribute_elements_match_the_group_faces_in_the_mesh(route_b):
    """Each surface attribute holds triangles only, and every volume attribute tetrahedra only."""
    text = route_b.mesh_path.read_text().split("$Elements")[1].splitlines()
    kinds = Counter()
    for line in text[2:]:
        if line.startswith("$"):
            break
        fields = line.split()
        kinds[(int(fields[1]), int(fields[3]))] += 1
    surface_attrs = {a for (t, a) in kinds if t == 2}
    volume_attrs = {a for (t, a) in kinds if t == 4}
    expected_volumes = {
        a for v in route_b.groups["volumes"].values() for a in v["attributes"]
    }
    assert volume_attrs == expected_volumes
    expected_surfaces = {
        part["attribute"]
        for section in ("boundary_surfaces", "port_surfaces")
        for record in route_b.groups[section].values()
        for part in record["parts"]
    }
    assert surface_attrs == expected_surfaces


def test_port_footprint_is_checked_against_the_mesh(route_b, tmp_path):
    """Each port triangle is shared by two tetrahedra and each host shares a mesh edge with its PEC attributes."""
    # The worker already ran the checks; run them again on the delivered mesh, in this process.
    lower_mesh_in_current_process_check(route_b, tmp_path)


def lower_mesh_in_current_process_check(result, tmp_path):
    """Re-open ``palace.msh`` and run the structural validators on it, then break a host and expect a refusal."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only where a native mesh is checked.

    from scgsim.palace.mesh.validation import (  # noqa: PLC0415 -- the validators are the unit under test, loaded beside Gmsh.
        validate_port_pec_node_intersections,
        validate_port_tetrahedron_face_conformality,
    )

    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(result.mesh_path))
        validate_port_tetrahedron_face_conformality(result.groups)
        validate_port_pec_node_intersections(result.groups)
        broken = json.loads(json.dumps(result.groups))
        exterior = next(
            r["tags"]
            for r in broken["boundary_surfaces"].values()
            if r["interface_type"] == "EXT"
        )
        for record in broken["pec_surfaces"].values():
            if record["entity"] == "pad":
                record["tags"] = exterior
        with pytest.raises(ValueError, match="does not share a mesh edge"):
            validate_port_pec_node_intersections(broken)
    finally:
        gmsh.finalize()


def test_a_lumped_port_on_a_round_head_arm_lands_on_the_arc(tmp_path):
    """The authored port ends inside a circular pad: the arc landing meshes and the port stays conformal."""
    effective, curved = circular_pad_input(tmp_path)
    arc = EffectiveInput(curved, effective.representation)
    result = mesh_run(arc, "B", tmp_path, specs=surface_epr_specs(MA=[0.5], SA=None))
    manifest, mesh = check_run(result, "B")
    assert manifest["arrangement"]["arc_edges"] > 0
    assert result.groups["port_surfaces"]["P1"]["host_entity_ids"] == [
        "ground_0000",
        "pad",
    ]
    assert_config_binds_the_mesh(config_of(result, "B").config, mesh)


def test_a_junction_support_plane_port_meshes_on_route_a(tmp_path):
    """The rectangular junction sits on the sheet plane between two sheets and is a lumped port."""
    effective = rectangular_junction_pair()
    result = mesh_run(
        effective, "A", tmp_path, specs=surface_epr_specs(MA=None, MS=None)
    )
    _, mesh = check_run(result, "A")
    port = result.groups["port_surfaces"]["P1"]
    assert port["host_entity_ids"] == ["chip/left", "chip/right"]
    ports = [
        LayoutPortBinding(
            1,
            port["source_name"],
            port["target_layer"],
            port["source_layer"],
            tuple(port["direction"]),
            1e-9,
        )
    ]
    config = build_eigenmode_config(
        groups=result.groups,
        ports=ports,
        materials={
            **MATERIALS,
            "Si": {"kind": "dielectric", "permittivity": 4, "loss_tangent": 0},
        },
        surface_epr_specs=surface_epr_specs(MA=None, MS=None),
        numerical=configure_numerical_controls(amr_max_passes=0),
        num_modes=2,
        target_hz=None,
        eigenmode_tolerance=1e-6,
        save_fields=0,
        mesh_path=result.mesh_path,
    )
    assert_config_binds_the_mesh(config.config, mesh)


def test_a_port_on_the_metal_top_plane_meshes(tmp_path):
    """Hosts attached at their upper face put the port on the metal top (a different support plane)."""
    effective = gds_pad_in_ground(tmp_path, attachment=("upper", "upper"))
    result = mesh_run(effective, "B", tmp_path, specs=surface_epr_specs(MA=None))
    check_run(result, "B")


def test_second_order_geometry_is_delivered_in_the_palace_mesh(tmp_path):
    """``MesherOptions(geometry_order=2)`` gives a second-order palace.msh; Palace only passes it on."""
    result = mesh_run(
        effective_input("B", tmp_path),
        "B",
        tmp_path,
        mesher=MesherOptions(geometry_order=2),
    )
    manifest, _ = check_run(result, "B")
    orders = {
        t["order"]
        for t in manifest["meshing"]["statistics"]["element_types"]
        if t["dimension"] == 3
    }
    assert orders == {2}
    assert manifest["meshing"]["mesher"]["geometry_order"] == 2


def test_a_mesh_below_the_threshold_stops_the_run_with_the_report(tmp_path):
    """No repair rounds and a threshold no mesh reaches: MeshRepairError carries the report; no palace.msh."""
    with pytest.raises(MeshRepairError) as caught:
        mesh_run(
            effective_input("B", tmp_path),
            "B",
            tmp_path,
            quality_threshold=0.99,
            budgets=Budgets(max_rounds=0),
        )
    assert caught.value.report["stop"] != "clean"
    assert not (tmp_path / "run" / "palace.msh").exists()
    assert (tmp_path / "run" / "mesh").is_dir()


def test_the_unbanded_build_is_skipped_without_margins(tmp_path):
    """Specs without margins request no insets: the model has no band family."""
    result = mesh_run(
        effective_input("B", tmp_path), "B", tmp_path, specs=surface_epr_specs(MA=None)
    )
    assert json.loads(result.sidecar_path.read_text())["insets"] == []
    assert not any(
        "::band=" in name
        for name in read_msh22(result.mesh_path).physical_names.values()
    )


def test_a_mesh_with_the_wrong_sidecar_is_refused(route_b, tmp_path):
    """Palace's own check that the mesh it lowers is the one its groups came from."""
    groups = json.loads(json.dumps(route_b.groups))
    victim = next(iter(groups["volumes"].values()))
    victim["tags"] = [*victim["tags"], 99]
    # palace.msh stands in for the module's mesh: its volume groups carry the primary names.
    with pytest.raises(Exception, match="covers other entities"):
        lower_mesh_in_current_process(
            source_mesh=route_b.mesh_path, mesh_path=tmp_path / "out.msh", groups=groups
        )
    groups = json.loads(json.dumps(route_b.groups))
    next(iter(groups["volumes"].values()))["group"] = "not_a_group"
    with pytest.raises(Exception, match="no physical group"):
        lower_mesh_in_current_process(
            source_mesh=route_b.mesh_path, mesh_path=tmp_path / "out.msh", groups=groups
        )
