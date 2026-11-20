"""An install without the ``aedt`` extras imports every public module, and AEDT preparation names that extra (#291).

OCP (``cadquery-ocp-novtk``) and PyAEDT come only with the ``aedt`` and ``aedt-epr`` extras, yet a
``[palace,visualization]`` install imports ``scgsim.visualization``, whose AEDT loader imports ``scgsim.aedt``.
Each import runs in a fresh interpreter with OCP and PyAEDT hidden: this environment has both installed, and
another test may already have imported them.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from sgb_inputs import rectangular_junction_source

from scgsim.aedt.preparation.sgb_model import prepare_sgb_model
from scgsim.geometry.sgb.output.verification import ocp

# A name that is None in sys.modules fails to import, as a package that is not installed does.
WITHOUT_AEDT_EXTRAS = 'import sys; sys.modules.update({"OCP": None, "ansys": None}); '


@pytest.mark.parametrize(
    "statement",
    [
        "import scgsim.aedt",
        "from scgsim.visualization import inspect_aedt_geometry, inspect_palace_geometry",
    ],
)
def test_a_public_module_imports_without_ocp_or_pyaedt(statement):
    """The import in #291 failed on OCP at ``scgsim.aedt``'s import of the STEP exporter."""
    run = subprocess.run(
        [sys.executable, "-c", WITHOUT_AEDT_EXTRAS + statement],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_preparing_an_aedt_model_without_ocp_names_the_extra_before_building(tmp_path, monkeypatch):
    """The refusal comes before the SGB build writes anything, and says which extra installs OCP."""
    monkeypatch.setattr(ocp, "ocp_available", lambda: False)
    source, stack = rectangular_junction_source("VACUUM_REGION")
    with pytest.raises(ocp.OcctUnavailable, match=r"`aedt` or `aedt-epr` extra"):
        prepare_sgb_model(
            source, prepared_stack=stack, modeling="solid", output_dir=tmp_path / "model"
        )
    assert not (tmp_path / "model").exists()
