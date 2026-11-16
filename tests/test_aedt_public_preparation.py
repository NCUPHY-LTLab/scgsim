"""Public file-only HFSS Eigenmode handoffs of a real SGB model; no native project or success evidence.

``produce_request`` writes the handoff of the rectangular junction pair's Route C model (a ``solid``
request), built once per session.  The handoff carries the model's STEP, BREP, sidecar and guard in its
``model/`` folder.  Skipped where OCP (the ``occt`` group) is missing.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("OCP")

from sgb_inputs import rectangular_junction_source  # noqa: E402

from scgsim.aedt import (  # noqa: E402
    AedtResources,
    EigenmodeRunControl,
    PlanarJunction,
    prepare_hfss_eigenmode_from_geometry,
    prepare_sgb_model,
)
from scgsim.aedt.preparation.sgb_model import PreparedSgbModel  # noqa: E402
from scgsim.aedt.specs.parse import parse_aedt_spec  # noqa: E402

pytestmark = pytest.mark.occt

JUNCTION = PlanarJunction(
    "junction", "sheet_polygon", "A", "B", (1, 0), 2.0, 10e-9, 2e-15
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@functools.cache
def junction_model() -> PreparedSgbModel:
    """The Route C model of the junction pair, built once into a folder that lives for the session."""
    source, stack = rectangular_junction_source("VACUUM_REGION")
    folder = Path(tempfile.mkdtemp(prefix="scgsim-test-sgb-model-"))
    atexit.register(shutil.rmtree, folder, True)
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling="solid",
        output_dir=folder,
        junctions=(JUNCTION,),
    )


def produce_request(root: Path, version="2024.2"):
    """Produce a real file-only eigenmode handoff with explicit version and controls."""
    return prepare_hfss_eigenmode_from_geometry(
        aedt_version=version,
        model=junction_model(),
        project_name="ProtocolSource",
        design_name="Design",
        output_dir=root,
        run_control=EigenmodeRunControl("Setup", 1.0, 1, 3, 1.0),
        resources=AedtResources(2, 8),
    )


@pytest.mark.parametrize("version", ("2023.2", "2024.2"))
def test_both_versions_produce_real_not_run_handoff_and_archive(tmp_path, version):
    """Matching not-run handoffs and exact archives, the model's files among them."""
    plan = produce_request(tmp_path / version, version)
    payload = json.loads(plan.spec_path.read_text())
    parsed = parse_aedt_spec(payload, base_dir=plan.run_dir)
    assert parsed.aedt_version == version
    assert parsed.modeling == "solid"
    assert parsed.to_payload() == payload
    assert payload["model"]["model_dir"] == "model"
    receipt = json.loads(plan.receipt_path.read_text())
    assert receipt["status"] == "not_run"
    assert receipt["requested"]["aedt_version"] == version
    assert receipt["source"]["sgb_model_sha256"] == junction_model().identity_sha256
    assert not (plan.run_dir / "ProtocolSource.aedt").exists()
    manifest = json.loads(
        (plan.run_dir / "metadata/aedt_handoff_manifest.json").read_text()
    )
    assert {
        "model/model.step",
        f"model/{junction_model().files['sidecar']['path']}",
        f"model/{junction_model().files['guard']['path']}",
    } <= set(manifest["allowed_paths"])
    for item in manifest["members"]:
        body = (plan.run_dir / item["path"]).read_bytes()
        assert len(body) == item["bytes"]
        assert hashlib.sha256(body).hexdigest() == item["sha256"]
    with tarfile.open(plan.archive_path) as archive:
        assert set(archive.getnames()) == set(manifest["allowed_paths"])


def test_a_moved_handoff_binds_its_own_model(tmp_path):
    """The request decodes against the folder that holds it, wherever that folder is."""
    plan = produce_request(tmp_path / "request")
    moved = tmp_path / "moved"
    shutil.move(plan.run_dir, moved)
    parsed = parse_aedt_spec(
        json.loads((moved / "aedt_spec.json").read_text()), base_dir=moved
    )
    assert parsed.model.model_dir == moved / "model"
    (moved / "model" / "model.step").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        parse_aedt_spec(
            json.loads((moved / "aedt_spec.json").read_text()), base_dir=moved
        )


def test_public_preparation_preserves_fresh_output_failure(tmp_path):
    """Reject an existing output directory without changing the original request bytes."""
    root = tmp_path / "run"
    produce_request(root)
    before = (root / "aedt_spec.json").read_bytes()
    with pytest.raises(FileExistsError):
        produce_request(root)
    assert (root / "aedt_spec.json").read_bytes() == before
