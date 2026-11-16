"""``scripts/aedt_acceptance_67.py``: the part of ``build`` before AEDT, on the real SGB model; no AEDT is started."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("OCP")
pytest.importorskip("gdsfactory")

from scgsim.aedt import EigenmodeSim  # noqa: E402
from scgsim.aedt.specs.parse import parse_aedt_spec  # noqa: E402
from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aedt_acceptance_67.py"


@pytest.fixture(scope="module")
def acceptance():
    """The script, imported as a module; the PDK its layout activates is restored afterwards."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    spec = importlib.util.spec_from_file_location("aedt_acceptance_67", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    active = gf.pdk._ACTIVE_PDK  # noqa: SLF001 -- preserve the process-local PDK around the script's layout
    yield module
    gf.pdk._ACTIVE_PDK = active  # noqa: SLF001 -- restore the process-local PDK


def test_the_handoff_is_the_route_c_epr_model_of_the_pad_pair(acceptance, tmp_path):
    """One junction on the port sheet between the pads, and MA, MS and SA rows with their inset bands."""
    snapshot, junction = acceptance._plan()  # noqa: SLF001 -- the script's own preparation step
    handoff = acceptance._handoff(  # noqa: SLF001 -- the script's own preparation step
        tmp_path / "out", snapshot, junction, acceptance.AedtResources(2, 50)
    )
    spec = parse_aedt_spec(
        json.loads(handoff.spec_path.read_text(encoding="utf-8")),
        base_dir=handoff.run_dir,
    )
    assert spec.modeling == "solid"
    assert [item.junction.junction_id for item in spec.model.junctions] == ["jj"]
    sidecar = read_sidecar(spec.model.path("sidecar"))
    assert sidecar["model"]["route"] == "C"
    banded = {
        row["interface_type"]
        for row in sidecar["surface_epr_row"].values()
        if row["band"] is not None
    }
    assert banded == {"MA", "MS", "SA"}


def test_the_acceptance_layout_passes_the_aedt_eigenmode_check(acceptance):
    """``EigenmodeSim.simulation_drc`` accepts the plan the script builds, junction included, on the route it builds."""
    report = EigenmodeSim.simulation_drc(
        acceptance._geometry_plan(),  # noqa: SLF001 -- the script's own plan, its junction declared
        surface_epr_specs=acceptance.SURFACE_EPR_SPECS,
        modeling="solid",
    )
    assert report.passed


def test_a_scratch_folder_inside_the_repository_is_refused(acceptance):
    """Scratch output never goes into the repository; the refusal comes before any preparation."""
    inside = SCRIPT.parents[1] / "scratch-acceptance"
    with pytest.raises(SystemExit, match="inside the repository"):
        acceptance.main(
            ["build", str(inside), "--cores", "2", "--ram-limit-percent", "50"]
        )
    assert not inside.exists()
