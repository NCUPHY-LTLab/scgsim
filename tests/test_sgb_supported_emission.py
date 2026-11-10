"""The Supported Geometry sample set: every scene on the supported list is written as BREP and XAO.

``test_sgb_supported_geometry`` builds each scene and checks its bands.  A model that builds can still be
refused when its files are planned, so every scene of the same list is also taken through ``emit_model``, the
second half of ``build_sgb_geometry``: it plans the BREP and writes the BREP and XAO files, the reports and the
sidecar.  A scene that builds but cannot be written is not on the list; it is pinned here at its refusal, so
that a change which lets it be written is noticed.
"""

from __future__ import annotations

import pytest
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import Scene, assert_exact_bands, catalogue, type_bands
from test_sgb_supported_geometry import SCENES, scene_params

from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue


def sample_model(scene: Scene) -> ConformalModel:
    """The model of a scene as the sample set builds it: Route B, with MA, MS and SA bands asked for by type."""
    return build_model_from_catalogue(catalogue(scene), "B", insets=type_bands())


@pytest.mark.parametrize("name", scene_params())
def test_a_supported_cell_is_written_as_brep_and_xao(name, tmp_path):
    """The model of the scene is planned and both files are written; a scene that cannot be written fails here."""
    artifacts = emit_model(sample_model(SCENES[name]()), tmp_path)
    assert artifacts.brep.stat().st_size > 0
    assert artifacts.xao.stat().st_size > 0


@pytest.mark.parametrize("name", sorted(pdk.NOT_WRITTEN))
def test_a_scene_kept_off_the_list_builds_but_is_refused_when_written(name, tmp_path):
    """The lumped sheet across a 32-point pad builds with exact bands; planning its BREP refuses a wall face.

    Metal side walls beside the sheet are neither planar nor cylindrical, and the writer has no record for such
    a face.  When this scene is written, move it back to ``SCENES`` and to the Supported Geometry page.
    """
    scene = pdk.NOT_WRITTEN[name]()
    assert_exact_bands(scene)
    model = sample_model(scene)
    with pytest.raises(SgbError, match="non-planar, non-cylindrical face"):
        emit_model(model, tmp_path)
