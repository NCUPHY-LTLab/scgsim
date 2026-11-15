"""Component to ``palace.msh`` and config through SGB, the meshing module and Palace's lowering (needs Gmsh).

Every case runs ``build_route_geometry`` and ``mesh_route_geometry`` for real and then builds the config from the groups it returned.  No
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
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.config.config import (
    build_eigenmode_config,
    build_electrostatic_config,
)
from scgsim.palace.config.config_records import (
    ConfigInput,
    EigenmodeSettings,
    ElectrostaticSettings,
    TerminalBinding,
)
from scgsim.palace.config.index import SurfaceEprEntry
from scgsim.palace.config.numerical import NumericalControls
from scgsim.palace.mesh.native import lower_mesh_in_current_process
from scgsim.palace.mesh.summary import mesh_summary
from scgsim.palace.reports.plotting import mesh_summary_html
from scgsim.palace.results.index_contract import (
    RouteConsistency,
    validate_config_index_correspondence,
)
from scgsim.palace.results.tables import count_index_entries
from scgsim.palace.simulation.core import index_map
from scgsim.palace.source.ports import LayoutPortBinding
from scgsim.palace.source.surface_epr import SurfaceEprSpecs

pytestmark = pytest.mark.slow

# A Route A sheet is one face for MA and MS, so both ask the same margins.
SPECS = surface_epr_specs(MA=[0.5, 1.0], MS=[0.5, 1.0], SA=[0.5])


def physical_names(mesh_path):
    """(dimension, tag) -> name of every physical group of a written mesh, read by Gmsh."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only where a native mesh is checked.

    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(mesh_path))
        return {
            (dim, tag): gmsh.model.getPhysicalName(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()


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
    names = physical_names(run / "palace.msh")
    assert names
    manifest = json.loads(result.mesh_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["source"]["scgsim_sgb"] == "scgsim.geometry.sgb"
    assert manifest["source"]["sidecar_schema"] == "scgsim.sgb.sidecar.v5"
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
    assert listed == names
    assert {e["group"] for e in manifest["groups"]} >= set(result.groups["volumes"])
    return manifest, names


def config_of(result, route, specs=None):
    """The Electrostatic config of a run: first Net as Terminal, second as Ground."""
    nets = sorted({r["net"] for r in result.groups["pec_surfaces"].values()})
    common = ConfigInput(
        problem="Electrostatic",
        groups=result.groups,
        mesh_path=result.mesh_path,
        materials=MATERIALS,
        surface_epr_specs=SurfaceEprSpecs.normalized(specs or {}, materials=MATERIALS),
        numerical=NumericalControls().with_defaults(
            ElectrostaticSim.numerical_defaults
        ),
    )
    return build_electrostatic_config(
        common,
        terminals=[TerminalBinding("t", nets[0])],
        ground_net_ids=nets[1:],
        electrostatic=ElectrostaticSettings(unassigned_conductor_policy="error"),
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


def assert_config_binds_the_mesh(config, names):
    """Every attribute of the config is a physical group of the mesh, in its dimension."""
    surfaces, volumes = config_attributes(config)
    assert (surfaces | volumes) <= set(names)


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
    assert banded_groups_of(result) and banded_groups_of(result) <= set(mesh.values())
    config = config_of(result, route, SPECS)
    assert_config_binds_the_mesh(config.config, mesh)
    assert manifest["route"] == route
    assert manifest["route_policy"]["conductor_sheets"] is (route == "A")
    # The unbanded build used to find the supports is gone, and only the final model is on disk.
    assert manifest["insets"]
    assert_readers_accept_the_run(result, route, manifest, config)


def sidecar_of_run(result):
    """The decoded sidecar of a mesh run."""
    return json.loads(result.sidecar_path.read_text(encoding="utf-8"))


def banded_groups_of(result):
    """The surface groups SGB cut into bands, read from the sidecar's ``band`` field."""
    return {
        name
        for name, group in sidecar_of_run(result)["group"].items()
        if group.get("band") is not None
    }


def band_families(sidecar):
    """Every band family of a sidecar once, each with its id as ``request_id``.

    A Route A sheet serves an MA and an MS request, so both requests list the same family.
    """
    return [
        {**family, "request_id": family_id}
        for family_id, family in sidecar["band_family"].items()
    ]


def expected_margin_attributes(result, entry):
    """The attributes of bands first_band..K of the family that serves a margin entry, from the sidecar alone."""
    first_band = entry["inset"]["first_band"]
    group = entry["metadata"]["rows"][0]["group"]
    (family,) = (
        f
        for f in band_families(sidecar_of_run(result))
        if any(group in names for names in f["bands"].values())
    )
    wanted = [
        name
        for band, names in family["bands"].items()
        if int(band) >= first_band
        for name in names
    ]
    return sorted(
        a
        for name in wanted
        for a in result.groups["boundary_surfaces"][name]["attributes"]
    )


def assert_margin_rows_select_the_bands_of_their_families(result, config):
    """Each margin ``Dielectric`` row lists the attributes of the family's bands from its margin inwards."""
    rows = {
        row["Index"]: row
        for row in config.config["Boundaries"]["Postprocessing"]["Dielectric"]
    }
    margin_entries = [
        e.to_payload()
        for e in config.index_entries
        if isinstance(e, SurfaceEprEntry) and e.inset is not None
    ]
    assert margin_entries
    for entry in margin_entries:
        wanted = expected_margin_attributes(result, entry)
        assert rows[entry["index"]]["Attributes"] == wanted
        assert entry["attributes"] == wanted
    return {e["entry_name"]: e["inset"]["margin_um"] for e in margin_entries}


def assert_readers_accept_the_run(result, route, manifest, config):
    """The manifest the build wrote and the index map of its config satisfy the readers, and render."""
    index = index_map(
        config.index_entries, ground_boundary=config.ground_boundary_resolution
    )
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


def test_route_b_sends_one_request_per_spec_and_gets_a_band_family_per_support(
    route_b,
):
    """The bands requested from the specs: one type-level request each, expanded to MA tops, MS bottoms and the SA gap."""
    insets = sidecar_of_run(route_b)["inset_request"]
    assert set(insets) == {"epr_MA", "epr_MS", "epr_SA"}
    families = band_families(sidecar_of_run(route_b))
    assert len(families) == 5
    assert any("VACUUM_REGION" in f["request_id"] for f in families)


def test_the_default_path_margin_rows_select_the_bands_of_their_families(
    route_a, route_b
):
    """Routes A and B: the config's margin rows list exactly the band attributes the sidecar's families give."""
    for route, result in (("A", route_a), ("B", route_b)):
        config = config_of(result, route, SPECS)
        margins = assert_margin_rows_select_the_bands_of_their_families(result, config)
        assert set(margins.values()) == {0.5, 1.0}


def test_route_a_attributes_split_where_the_row_signature_changes(route_a):
    """Route A sheet bands carry the MA row (above) and the MS row (below) on one attribute each."""
    left_bands = [
        record
        for record in route_a.groups["boundary_surfaces"].values()
        if record["interface_type"] == "SHEET"
        and record["entities"] == ["chip/left"]
        and record["band"] is not None
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
            {"L": 1e-9},
        )
    ]
    materials = {
        **MATERIALS,
        "Si": {"kind": "dielectric", "permittivity": 4, "loss_tangent": 0},
    }
    common = ConfigInput(
        problem="Eigenmode",
        groups=result.groups,
        mesh_path=result.mesh_path,
        materials=materials,
        surface_epr_specs=SurfaceEprSpecs.normalized(
            surface_epr_specs(MA=None, MS=None), materials=materials
        ),
        numerical=NumericalControls(amr_max_passes=0).with_defaults(
            EigenmodeSim.numerical_defaults
        ),
    )
    config = build_eigenmode_config(
        common, ports=ports, eigenmode=EigenmodeSettings(num_modes=2)
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
    assert json.loads(result.sidecar_path.read_text())["inset_request"] == {}
    assert not banded_groups_of(result)


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


CUSTOM_SPECS = {
    # route -> (conductor with its own margins, its margins, the spec's own margins)
    "A": ("chip/left", [0.25], [0.5, 1.0]),
    "B": ("pad", [0.25, 0.5], [0.5, 1.0]),
}


@pytest.mark.parametrize("route", ["A", "B"])
def test_the_custom_path_bands_one_conductor_with_its_own_margins(route, tmp_path):
    """One conductor customised: two builds, concrete request for it, type-level request for the rest."""
    entity, own, rest = CUSTOM_SPECS[route]
    specs = surface_epr_specs(MA=rest, MS=rest, SA=[0.5])
    specs["MA"]["inset_margins_by_entity_um"] = {entity: own}
    if (
        route == "A"
    ):  # the sheet is the face of both MA and MS, so MS takes the conductor's margins too
        specs["MS"]["inset_margins_by_entity_um"] = {entity: own}
    result = mesh_run(effective_input(route, tmp_path), route, tmp_path, specs=specs)
    check_run(result, route)
    sidecar = sidecar_of_run(result)
    requests = sidecar["inset_request"]
    concrete = [r for r in requests.values() if r["scope"] == "face"]
    assert [r["support"]["entity"] for r in concrete] == [entity]
    type_ma = requests["epr_MA"]
    assert type_ma["support"]["exclude_entity_ids"] == [entity]
    assert entity not in {
        sidecar["band_family"][f]["support"]["entity"] for f in type_ma["families"]
    }

    config = config_of(result, route, specs)
    margins = assert_margin_rows_select_the_bands_of_their_families(result, config)
    ma_margins = {
        name.split("__inset_")[0]: [] for name in margins if name.startswith("MA::")
    }
    for name, margin in sorted(margins.items()):
        if name.startswith("MA::"):
            ma_margins[name.split("__inset_")[0]].append(margin)
    own_rows = [m for n, m in ma_margins.items() if entity in n]
    other_rows = [m for n, m in ma_margins.items() if entity not in n]
    assert own_rows == [own]
    assert other_rows and all(m == rest for m in other_rows)
    assert_config_binds_the_mesh(config.config, physical_names(result.mesh_path))


def test_a_sheet_asked_the_same_margins_in_different_spellings_by_ma_and_ms_meshes(
    tmp_path,
):
    """MA and MS write one sheet's margins as 0.3 and 0.1 + 0.2: the same multiple of the grid, one family."""
    specs = surface_epr_specs(MA=None, MS=None)
    specs["MA"]["inset_margins_by_entity_um"] = {"chip/left": [0.3]}
    specs["MS"]["inset_margins_by_entity_um"] = {"chip/left": [0.1 + 0.2]}
    result = mesh_run(effective_input("A", tmp_path), "A", tmp_path, specs=specs)
    check_run(result, "A")
    sidecar = sidecar_of_run(result)
    ((family_id,),) = (
        r["families"] for r in sidecar["inset_request"].values() if r["scope"] == "face"
    )
    family = sidecar["band_family"][family_id]
    assert family["support"]["entity"] == "chip/left"
    assert family["margins_grid"] == [300]
    config = config_of(result, "A", specs)
    margins = assert_margin_rows_select_the_bands_of_their_families(result, config)
    assert {name.split("::")[0] for name in margins} == {"MA", "MS"}
    assert_config_binds_the_mesh(config.config, physical_names(result.mesh_path))
