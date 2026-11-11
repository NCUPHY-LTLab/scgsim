"""The Supported Geometry sample set: every scene on the supported list is written as BREP and XAO.

``test_sgb_supported_geometry`` builds each scene and checks its bands.  A model that builds can still be
refused when its files are planned, so every scene of the same list is also taken through ``emit_model``, the
second half of ``build_sgb_geometry``: it plans the BREP and writes the BREP and XAO files, the reports and the
sidecar.  A scene that builds but cannot be written is not on the list; it is pinned here at its refusal, so
that a change which lets it be written is noticed.  So is a scene whose written model the meshing stage of
``simulation_drc`` refuses (``scgsim.meshing.check_fine_structure`` on its sidecar, as ``mesh()`` runs it).
"""

from __future__ import annotations

import pytest
import sgb_supported_cells as disk_qubit
import sgb_supported_pdk_cells as pdk
from sgb_supported_kit import (
    Scene,
    assert_exact_bands,
    catalogue,
    type_bands,
)
from sgb_unsupported_cells import straight_lead
from test_sgb_supported_geometry import SCENES, scene_params

from scgsim.geometry.sgb.build import emit_model
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.meshing import check_fine_structure

# Where the core of the straight lead lies, from the disk end to the tip (um): the lead runs at -45 degrees.
LEAD_CORE_X_UM = (84.670, 165.074)
LEAD_CORE_Y_UM = (-165.074, -84.670)


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


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(disk_qubit.REFUSED_BEFORE_MESHING))
def test_a_lead_twice_a_margin_wide_is_refused_before_meshing_as_the_erosion(
    name, tmp_path
):
    """The straight junction lead is 2 um wide, twice the 1 um margin; the meshing stage refuses its core (Issue #128).

    The bands are those of the exact erosion, which keeps a core 0.7 nm wide down the 113 um lead.  The meshing
    stage refuses it as a class B face, owned by the erosion, located on the lead, with the fix to change the
    margins or the width.  When such a lead is no longer refused, move the scene back to ``SCENES``.
    """
    model = assert_exact_bands(disk_qubit.REFUSED_BEFORE_MESHING[name]())
    report = check_fine_structure(emit_model(model, tmp_path).sidecar)
    assert not report.passed
    assert {(f.rule, f.owner) for f in report.refusals} == {("B", "erosion")}
    cores = [f for f in report.refusals if f.measure == "face width 2A/P"]
    assert sorted(f.groups for f in cores) == [
        ("MA::disk__top::band=2",),
        ("MS::disk__bottom::band=2",),
    ]
    for finding in cores:
        (x0, y0, _), (x1, y1, _) = finding.bounds_um
        assert (x0, x1) == pytest.approx(LEAD_CORE_X_UM, abs=0.001)
        assert (y0, y1) == pytest.approx(LEAD_CORE_Y_UM, abs=0.001)
        assert finding.value_steps is not None and finding.value_steps < 1
        assert "change the margins" in finding.fix


@pytest.mark.parametrize(
    ("width_um", "passed"),
    [(2.0, True), (2.004, False)],
    ids=["twice_the_margin", "four_steps_wider"],
)
def test_a_lead_exactly_twice_the_margin_has_no_core_and_a_little_wider_is_refused(
    width_um, passed, tmp_path
):
    """At exactly twice the 1 um margin the bands meet down the middle of the lead and leave no core.

    The bands are the exact erosion and the meshing stage passes the model.  Four grid steps wider, the
    core is a face four steps wide, and the meshing stage refuses it, owned by the erosion.
    """
    model = assert_exact_bands(straight_lead(width_um))
    report = check_fine_structure(emit_model(model, tmp_path).sidecar)
    assert report.passed is passed
    if not passed:
        assert {(f.rule, f.owner) for f in report.refusals} == {("B", "erosion")}
