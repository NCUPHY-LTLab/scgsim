"""The meshing stage names the layer of each Entity a finding comes from, read from the sidecar's ``entity`` table.

A located refusal names the object, its layer, the place in um and who must act (#90).  The meshing stage reads
only the sidecar, so the layer is the ``layer`` of the Entities its finding comes from (#170, slice C).
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace

import pytest
import sgb_supported_cells as disk_qubit
from sgb_supported_kit import catalogue, type_bands
from test_simulation_drc import SIMULATIONS
from test_simulation_drc_meshing_stage import JOG_WALL_UM, PADS, refusal, scene

from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.meshing import check_fine_structure

# "at (x, y, z) um, <entity> in <groups> on layer <layers>; from": the place, the object and its layers.
LOCATED = re.compile(
    r"at \((?P<x>-?[\d.]+), (?P<y>-?[\d.]+), (?P<z>-?[\d.]+)\) um, (?P<entity>\S+) in "
    r"(?P<groups>.+?) on layers? (?P<layers>[^;]+); from"
)
LAYERS = {"disk": "M1", "ground": "M1/0"}


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_the_meshing_refusal_names_the_layer_the_refused_entity_was_drawn_on(
    monkeypatch, simulation
):
    """The jog of the pad is refused on the pad's layer: the message parses back to its place, face and layer."""
    error = refusal(simulation, scene(monkeypatch, PADS["jog"]))
    (finding,) = [f for f in error.report.meshing.findings if f.action == "refuse"]
    assert finding.entities == ("chip/pad_0000",)
    # The pad's level, as its conductor region declares it.
    assert finding.layers == ("metal",)
    match = LOCATED.search(str(error))
    assert match is not None, str(error)
    place = (float(match["x"]), float(match["y"]), float(match["z"]))
    assert place == pytest.approx(JOG_WALL_UM, abs=1e-4)
    assert match["entity"] == finding.entity
    assert match["groups"] == "MA::chip/pad_0000__sidewall"
    assert match["layers"].split(", ") == ["metal"]


def layered_junction_sidecar(tmp_path):
    """The sidecar of ``junction_fan`` with its two conductors drawn on layers M1 and M1/0."""
    prisms = tuple(
        replace(p, layer=LAYERS.get(p.entity_id))
        for p in catalogue(disk_qubit.SCENES["junction_fan"]()).prisms
    )
    layered = replace(catalogue(disk_qubit.SCENES["junction_fan"]()), prisms=prisms)
    model = build_model_from_catalogue(layered, "B", insets=type_bands())
    return read_sidecar(emit_model(model, tmp_path).sidecar)


def test_each_finding_names_the_layers_of_its_entities_from_the_entity_table(tmp_path):
    """The layers are those the ``entity`` table gives the finding's Entities; change the table, change the text."""
    sidecar = layered_junction_sidecar(tmp_path)
    assert {
        e: {p["layer"] for p in sidecar["entity"][e]["prisms"]} for e in LAYERS
    } == {e: {layer} for e, layer in LAYERS.items()}
    report = check_fine_structure(sidecar)
    assert report.findings
    for finding in report.findings:
        expected = sorted(
            {LAYERS[e] for e in finding.entities if e in LAYERS}
        )  # the vacuum, the substrate and a band family name no layer
        assert list(finding.layers) == expected
        match = LOCATED.search(finding.describe())
        if expected:
            assert match is not None and match["layers"].split(", ") == expected
        else:
            assert " on layer" not in finding.describe()
    renamed = copy.deepcopy(sidecar)
    for prism in renamed["entity"]["ground"]["prisms"]:
        prism["layer"] = "D0_GROUND"
    for before, after in zip(
        report.findings, check_fine_structure(renamed).findings, strict=True
    ):
        swapped = tuple(
            sorted("D0_GROUND" if layer == "M1/0" else layer for layer in before.layers)
        )
        assert after.layers == swapped
        assert (after.entity, after.location_um, after.owner) == (
            before.entity,
            before.location_um,
            before.owner,
        )
