"""Protocol-unit ES outputs; these fixtures do not represent native solver runs."""

import json
from pathlib import Path

import pytest

from scgsim.palace.results import resolve
from scgsim.palace.run import returned_receipt

_ES_TABLES = {
    "terminal-C": "i,C[i][1] (F)\n1,1e-15\n",
    "terminal-Cm": "i,C_m[i][1] (F)\n1,1e-15\n",
    "terminal-Cinv": "i,C⁻¹[i][1] (1/F)\n1,1e15\n",
    "terminal-V": "i,V_inc[i] (V)\n1,1\n",
    "domain-E": (
        "i,E_elec (J),E_mag (J),E_cap (J),E_ind (J),"
        "E_elec[1] (J),p_elec[1],E_mag[1] (J),p_mag[1]\n"
        "1,1,0,0,0,1,1,0,0\n"
    ),
    "error-indicators": "Norm,Minimum,Maximum,Mean\n0.1,0,0.1,0.05\n",
}


def _unit_outputs(root: Path, boundaries: dict) -> dict:
    """Write real protocol files without mocking the recorder or CSV validators."""
    results = root / "results/palace"
    results.mkdir(parents=True)
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "palace_index_map.json").write_text(
        '{"entries": [{"section": "Boundaries.Terminal", "index": 1}]}\n'
    )
    (root / "logs").mkdir()
    (root / "logs/unit.log").write_text("protocol-unit output fixture only\n")
    config = {"Problem": {"Type": "Electrostatic"}, "Boundaries": boundaries}
    (root / "config.json").write_text(json.dumps(config))
    (results / "palace.json").write_text("{}\n")
    for name, text in _ES_TABLES.items():
        (results / f"{name}.csv").write_text(text)
    return config


def _record(root: Path) -> dict:
    """Exercise receipt identity/completeness against actual fixture file hashes."""
    request = returned_receipt._ReturnRequest(  # noqa: SLF001 -- isolate the receipt output contract
        root=root,
        exit_code=0,
        log_path="logs/unit.log",
        job={"job_id": "protocol-unit-only", "job_name": None},
        execution_stage="solver",
        preflight_exit_code=0,
        solver_exit_code=0,
        tee_exit_code=0,
        diagnostic_after_the_fact=False,
        receipt_path="metadata/palace_returned_run_receipt.json",
    )
    handoff = returned_receipt._Handoff(  # noqa: SLF001 -- isolate the receipt output contract
        metadata={"palace_identity": {}},
        route="B",
        problem="Electrostatic",
        input_hashes=[dict(returned_receipt.record_for_path(root, "config.json"))],
        handoff_id="protocol-unit-only",
    )
    return dict(returned_receipt._receipt(request, handoff))  # noqa: SLF001 -- no native execution in this unit test


def _read(root: Path, config: dict, surfaces: int = 0) -> dict:
    """Use the strict CSV discovery/validation boundary after identity validation."""
    return resolve._discover_results(  # noqa: SLF001 -- exercise the strict table boundary
        root,
        "Electrostatic",
        {"terminal": 1, "domain": 1, "lumped_port": 0, "surface": surfaces},
        config,
    )


@pytest.mark.parametrize(
    "boundaries",
    ({}, {"Postprocessing": {}}, {"Postprocessing": {"Dielectric": []}}),
)
def test_es_without_surface_configuration_records_and_reads_without_surface_q(
    tmp_path, boundaries
):
    """A successful ES protocol requires only the outputs its setup requests."""
    config = _unit_outputs(tmp_path, boundaries)
    receipt = _record(tmp_path)
    assert receipt["status"] == "completed"
    assert receipt["identity_verified"] is True
    assert "results/palace/surface-Q.csv" not in {
        record["path"] for record in receipt["output_files"]
    }
    assert set(_read(tmp_path, config)) == set(_ES_TABLES)


def test_es_configured_surface_q_remains_required_and_validated(tmp_path):
    """A missing requested surface table still fails both return and read paths."""
    config = _unit_outputs(
        tmp_path,
        {"Postprocessing": {"Dielectric": [{"Index": 1}]}},
    )
    receipt = _record(tmp_path)
    assert receipt["status"] == "failed"
    assert receipt["identity_verified"] is False
    assert any(
        record["path"] == "results/palace/surface-Q.csv" and not record["present"]
        for record in receipt["output_files"]
    )
    with pytest.raises(FileNotFoundError, match="surface-Q"):
        _read(tmp_path, config, surfaces=1)

    path = tmp_path / "results/palace/surface-Q.csv"
    path.write_text("i,p_surf[1],Q_surf[1]\n1,0.1,100\n")
    assert _record(tmp_path)["status"] == "completed"
    assert _read(tmp_path, config, surfaces=1)["surface-Q"].rows[0]["p_surf[1]"] == 0.1
    path.write_text("i,p_surf[1],Q_surf[1]\n1,NaN,100\n")
    with pytest.raises(ValueError, match="non-finite"):
        _read(tmp_path, config, surfaces=1)


@pytest.mark.parametrize("missing", ("terminal-C", "domain-E"))
def test_es_other_mandatory_outputs_still_fail_when_missing(tmp_path, missing):
    """Optional surface output must not weaken the remaining output contract."""
    config = _unit_outputs(tmp_path, {})
    (tmp_path / f"results/palace/{missing}.csv").unlink()
    assert _record(tmp_path)["status"] == "failed"
    with pytest.raises(FileNotFoundError, match=missing):
        _read(tmp_path, config)
