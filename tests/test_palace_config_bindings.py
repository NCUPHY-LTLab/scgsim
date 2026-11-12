"""Generated Palace config bindings over the attributes of real SGB sidecars (Routes A and B; no Gmsh)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from palace_sgb_support import (
    MATERIALS,
    effective_input,
    sidecar_of,
    surface_epr_specs,
)

from scgsim.palace import EigenmodeSim
from scgsim.palace.mesh.physical_groups import groups_from_sidecar
from scgsim.palace.preparation.config import (
    LayoutPortBinding,
    TerminalBinding,
    build_eigenmode_config,
    build_electrostatic_config,
    configure_numerical_controls,
)
from scgsim.palace.results.surface_contract import validate_surface_epr_config
from scgsim.palace.simulation.lifecycle import Stage

NETS = {"A": ("left", "right"), "B": ("P", "G")}
PORTS = {
    "A": LayoutPortBinding(
        1, "chip/o_junction_lumped", "metal", "port", (1.0, 0.0, 0.0), {"L": 1e-9}
    ),
    "B": LayoutPortBinding(1, "P1", "metal", "9/0", (1.0, 0.0, 0.0), {"L": 1e-9}),
}


@pytest.fixture(params=["A", "B"])
def case(request, tmp_path):
    """(route, groups) of the plain build of each route."""
    route = request.param
    sidecar = sidecar_of(effective_input(route, tmp_path), route, tmp_path)
    return route, groups_from_sidecar(sidecar)


def electrostatic(groups, terminals, **overrides):
    """The Electrostatic config of ``groups`` with the usual defaults."""
    settings = {
        "ground_net_ids": (),
        "unassigned_conductor_policy": "error",
        "exterior_boundary_policy": "none",
        "surface_epr_specs": {},
    }
    settings.update(overrides)
    return build_electrostatic_config(
        groups=groups,
        terminals=terminals,
        materials=MATERIALS,
        save_fields=0,
        mesh_path=Path("palace.msh"),
        numerical=configure_numerical_controls(),
        **settings,
    )


def attributes_of(groups, net):
    """The attributes of every conductor surface of ``net``."""
    return sorted(
        a
        for record in groups["pec_surfaces"].values()
        if record["net"] == net
        for a in record["attributes"]
    )


def test_terminal_is_the_attributes_of_the_exact_net(case):
    """A Terminal lists the attributes of the conductor surfaces of its Net, and nothing else."""
    route, groups = case
    first, second = NETS[route]
    result = electrostatic(
        groups,
        [TerminalBinding("readout", first)],
        ground_net_ids=[second],
    )
    boundaries = result.config["Boundaries"]
    assert boundaries["Terminal"] == [
        {"Index": 1, "Attributes": attributes_of(groups, first)}
    ]
    assert boundaries["Ground"] == {"Attributes": attributes_of(groups, second)}
    entry = result.terminal_index_map[0]
    assert entry["attributes"] == attributes_of(groups, first)
    assert entry["net_id"] == first
    assert set(entry["physical_names"]) == {
        n for n, r in groups["pec_surfaces"].items() if r["net"] == first
    }


def test_terminal_binding_uses_the_entity_ids_as_conductor_components(case):
    """PEC, Terminal and Ground bind the conductors of the component (open item 7)."""
    route, groups = case
    first, second = NETS[route]
    result = electrostatic(
        groups, [TerminalBinding("t", first)], ground_net_ids=[second]
    )
    expected = {
        "A": {"left": "chip/left", "right": "chip/right"},
        "B": {"P": "pad", "G": "ground_0000"},
    }[route]
    assert result.terminal_index_map[0]["conductor_component_ids"] == [expected[first]]
    assert result.ground_boundary_resolution["conductor_component_ids"] == [
        expected[second]
    ]


def test_unknown_terminal_net_and_unassigned_conductors_are_errors(case):
    """A Net no conductor has fails; so does a conductor left unassigned under policy ``error``."""
    route, groups = case
    with pytest.raises(ValueError, match="has no exact net match"):
        electrostatic(groups, [TerminalBinding("t", "missing")])
    with pytest.raises(ValueError, match="Unassigned conductor components"):
        electrostatic(groups, [TerminalBinding("t", NETS[route][0])])


def test_exterior_ground_is_the_ext_attributes(case):
    """The exterior Ground is the attributes of the ``EXT`` groups and stays disjoint from the conductors."""
    route, groups = case
    first, second = NETS[route]
    result = electrostatic(
        groups,
        [TerminalBinding("t", first)],
        ground_net_ids=[second],
        exterior_boundary_policy="ground",
    )
    exterior = sorted(
        a
        for r in groups["boundary_surfaces"].values()
        if r["interface_type"] == "EXT"
        for a in r["attributes"]
    )
    resolution = result.ground_boundary_resolution
    assert resolution["exterior_attributes"] == exterior
    assert set(exterior) <= set(result.config["Boundaries"]["Ground"]["Attributes"])
    assert set(attributes_of(groups, first)).isdisjoint(exterior)


def test_domain_materials_come_from_the_volume_materials(case):
    """Each volume attribute gets the permittivity of its sidecar material in the stack."""
    route, groups = case
    result = electrostatic(
        groups, [TerminalBinding("t", NETS[route][0])], ground_net_ids=[NETS[route][1]]
    )
    rows = {
        tuple(r["Attributes"]): r["Permittivity"]
        for r in result.config["Domains"]["Materials"]
    }
    for volume in groups["volumes"].values():
        assert (
            rows[tuple(volume["attributes"])]
            == MATERIALS[volume["material"]]["permittivity"]
        )
    assert {
        e["metadata"]["material"]["kind"] for e in result.domain_energy_index_map
    } == {
        "dielectric",
        "vacuum",
    }


@pytest.mark.parametrize(
    ("films", "shared_rows_on_route_a"),
    [({}, False), ({"SA": None}, False), ({"MA": None, "MS": None}, True)],
)
def test_cracking_is_written_exactly_for_configured_shared_rows(
    case, films, shared_rows_on_route_a
):
    """Only a configured row on a Route A sheet cracks internal boundaries; the run's own check agrees."""
    route, groups = case
    first, second = NETS[route]
    result = electrostatic(
        groups,
        [TerminalBinding("t", first)],
        ground_net_ids=[second],
        surface_epr_specs=surface_epr_specs(**films),
    )
    model = result.config["Model"]
    if route == "A" and shared_rows_on_route_a:
        assert model["CrackInternalBoundaryElements"] is True
    else:
        assert "CrackInternalBoundaryElements" not in model
    entries = [*result.domain_energy_index_map, *result.surface_epr_index_map]
    validate_surface_epr_config(result.config, {"entries": entries})


def test_surface_rows_select_the_attributes_of_their_rows(case):
    """Dielectric rows list plain attributes; a sheet's MA and MS rows share them."""
    route, groups = case
    first, second = NETS[route]
    result = electrostatic(
        groups,
        [TerminalBinding("t", first)],
        ground_net_ids=[second],
        surface_epr_specs=surface_epr_specs(MA=None, MS=None),
    )
    rows = result.config["Boundaries"]["Postprocessing"]["Dielectric"]
    assert all(isinstance(a, int) and a > 0 for row in rows for a in row["Attributes"])
    if route == "A":
        by_type = {}
        for row in rows:
            by_type.setdefault(row["Type"], []).append(tuple(row["Attributes"]))
        assert sorted(by_type["MA"]) == sorted(by_type["MS"])


def eigenmode(groups, ports, **overrides):
    """The Eigenmode config of ``groups`` with the usual defaults."""
    numerical = configure_numerical_controls(amr_max_passes=0)
    return build_eigenmode_config(
        groups=groups,
        ports=ports,
        materials=MATERIALS,
        surface_epr_specs=overrides.get("surface_epr_specs", {}),
        numerical=numerical,
        num_modes=3,
        target_hz=None,
        eigenmode_tolerance=1e-6,
        save_fields=0,
        mesh_path=Path("palace.msh"),
    )


def test_eigenmode_pec_is_every_conductor_attribute_and_the_port_is_its_own(case):
    """PEC lists the conductor attributes; the lumped port lists its attributes and is not PEC."""
    route, groups = case
    result = eigenmode(groups, [PORTS[route]])
    boundaries = result.config["Boundaries"]
    pec = {a for r in groups["pec_surfaces"].values() for a in r["attributes"]}
    assert set(boundaries["PEC"]["Attributes"]) == pec
    (port,) = boundaries["LumpedPort"]
    assert port["Attributes"] == groups["port_surfaces"]["P1"]["attributes"]
    assert not set(port["Attributes"]) & pec
    (entry,) = [
        e for e in result.index_entries if e["section"] == "Boundaries.LumpedPort"
    ]
    assert (
        entry["metadata"]["host_entity_ids"]
        == groups["port_surfaces"]["P1"]["host_entity_ids"]
    )


def test_eigenmode_port_identity_must_match_the_sidecar(case):
    """A binding whose name or layers differ from the authored port is refused."""
    route, groups = case
    wrong = LayoutPortBinding(
        1, "other", "metal", PORTS[route].source_layer, (1.0, 0.0, 0.0), {"L": 1e-9}
    )
    with pytest.raises(ValueError, match="does not match SGB-authored identity"):
        eigenmode(groups, [wrong])
    with pytest.raises(ValueError, match="no SGB group P2"):
        eigenmode(
            groups,
            [
                PORTS[route],
                LayoutPortBinding(2, "x", "metal", "9/0", (1.0, 0.0, 0.0), {"L": 1e-9}),
            ],
        )


def test_eigenmode_surface_rows_do_not_overlap_the_port(case):
    """Surface-EPR attributes and the port attributes stay disjoint (the config refuses an overlap)."""
    route, groups = case
    result = eigenmode(
        groups, [PORTS[route]], surface_epr_specs=surface_epr_specs(MA=None, SA=None)
    )
    rows = result.config["Boundaries"]["Postprocessing"]["Dielectric"]
    port = set(result.config["Boundaries"]["LumpedPort"][0]["Attributes"])
    assert port.isdisjoint({a for row in rows for a in row["Attributes"]})


def test_eigenmode_epr_request_is_checked_against_the_rows_of_the_index(case, tmp_path):
    """``set_epr_request`` accepts the interface types some indexed row has, and refuses the others."""
    route, groups = case
    (tmp_path / "metadata").mkdir()
    (tmp_path / "palace.msh").write_text("$MeshFormat\n")
    manifest = tmp_path / "metadata" / "mesh_manifest.json"
    manifest.write_text("{}")
    sim = EigenmodeSim()
    sim._materials = MATERIALS  # noqa: SLF001 -- stands for set_plan()
    sim._resolved_ports = [PORTS[route]]  # noqa: SLF001 -- stands for mesh()
    sim._mesh_result = SimpleNamespace(  # noqa: SLF001 -- stands for mesh()
        groups=groups,
        mesh_path=tmp_path / "palace.msh",
        output_dir=tmp_path,
        mesh_manifest_path=manifest,
    )
    sim._stage = Stage.MESHED  # noqa: SLF001 -- stands for set_plan() and mesh()
    sim.surface_epr_specs = surface_epr_specs(MA=None)
    sim.set_numerical(amr_max_passes=0)
    sim.set_epr_request(surface_interfaces=["MA"])
    assert sim.write_config() == tmp_path / "config.json"
    sim.set_epr_request(surface_interfaces=["MS"])
    with pytest.raises(ValueError, match="unknown surface_interfaces"):
        sim.write_config()
