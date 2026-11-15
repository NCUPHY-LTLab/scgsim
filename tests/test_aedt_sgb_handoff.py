"""A prepared SGB model travels with its handoff and is checked again where AEDT runs (issue #67).

Each test prepares a real SGB model, attaches it to a handoff folder, moves that folder, and binds it back,
the way a handoff moves to the AEDT machine.  Skipped where OCP (the ``occt`` group) is missing.
"""

from __future__ import annotations

import shutil

import pytest

pytest.importorskip("OCP")

from sgb_inputs import rectangular_junction_source  # noqa: E402

from scgsim.aedt.preparation.sgb_handoff import (  # noqa: E402
    attach_sgb_model,
    bind_sgb_model,
    check_sgb_model_source,
    sgb_model_paths,
    sgb_model_source,
)
from scgsim.aedt.preparation.sgb_model import prepare_sgb_model  # noqa: E402

pytestmark = pytest.mark.occt


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    source, stack = rectangular_junction_source("VACUUM_REGION")
    return prepare_sgb_model(
        source,
        prepared_stack=stack,
        modeling="solid",
        output_dir=tmp_path_factory.mktemp("prepared"),
        ground_net_ids=("B",),
    )


def _moved(model, tmp_path):
    """Attach ``model`` to a handoff folder, then move the folder, as a handoff is moved to AEDT."""
    run_dir = tmp_path / "handoff"
    payload = attach_sgb_model(model, run_dir)
    moved = tmp_path / "elsewhere"
    shutil.move(run_dir, moved)
    return payload, moved


def test_a_moved_handoff_binds_its_own_copy_of_the_model(model, tmp_path):
    """The payload names only the handoff's ``model/`` folder; bound back, it is the same model."""
    payload, moved = _moved(model, tmp_path)
    assert payload["model_dir"] == "model"
    assert sorted(path.split("/")[0] for path in sgb_model_paths(model)) == ["model"] * 4
    for path in sgb_model_paths(model):
        assert (moved / path).is_file()
    bound = bind_sgb_model(payload, moved)
    assert bound.model_dir == moved / "model"
    assert bound.identity_sha256 == model.identity_sha256
    check_sgb_model_source(sgb_model_source(bound), model)


def test_a_changed_model_file_is_refused_when_bound(model, tmp_path):
    """A file of the handoff's model that changed after preparation is refused before any import."""
    payload, moved = _moved(model, tmp_path)
    sidecar = moved / "model" / model.files["sidecar"]["path"]
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar file .* is missing or changed"):
        bind_sgb_model(payload, moved)


def test_a_payload_naming_a_folder_outside_the_handoff_is_refused(model, tmp_path):
    """A handoff can only name the model files it carries."""
    payload = {**model.to_payload()}
    with pytest.raises(ValueError, match="lives in its 'model' folder"):
        bind_sgb_model(payload, tmp_path)


def test_a_source_record_without_an_sgb_model_is_refused(model):
    """A natively built handoff's source record binds no SGB model: refused (decision 6)."""
    native = {"planar_source_sha256": "0" * 64, "planar_model_sha256": "0" * 64}
    with pytest.raises(ValueError, match="binds no SGB model"):
        check_sgb_model_source(native, model)
    other = {**sgb_model_source(model), "sgb_model_sha256": "0" * 64}
    with pytest.raises(ValueError, match="sgb_model_sha256 differs"):
        check_sgb_model_source(other, model)
