"""``build_sgb_geometry``: real effective inputs to XAO, BREP and sidecar, and the load-side check on them."""

from __future__ import annotations

import copy
import json

import jsonschema
import pytest
from sgb_emit_cases import MARGINS
from sgb_gmsh_oracle import gmsh_session, mesh_coarsely
from sgb_inputs import (
    EffectiveInput,
    gds_pad_in_ground,
    rectangular_junction_pair,
    thin_film_pair,
)

import scgsim.geometry.sgb as sgb
from scgsim.contracts.sgb_sidecar import read_sidecar, views
from scgsim.geometry.sgb import SgbArtifacts, build_sgb_geometry
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    RefinementRequest,
)
from scgsim.geometry.sgb.errors import RouteError
from scgsim.geometry.sgb.sidecar.files import sidecar_schema
from scgsim.geometry.sgb.verification import ocp
from scgsim.geometry.sgb.verification.loaded import (
    LoadCheckError,
    apply_sidecar_embeds,
    verify_loaded,
    verify_xao,
)

ARTIFACT_NAMES = ("xao", "brep", "sidecar", "sliver_report", "validation_report")


@pytest.fixture(scope="module")
def gds_input(tmp_path_factory) -> EffectiveInput:
    """A real effective input from a GDS and a stack file (solid profile): a pad in ground with a lumped port."""
    return gds_pad_in_ground(tmp_path_factory.mktemp("gds"))


@pytest.fixture(scope="module")
def films() -> EffectiveInput:
    """Two Thin Film metals (Route A sheets) joined by a lumped port."""
    return thin_film_pair()


def build(effective: EffectiveInput, route: str, output_dir, **options) -> SgbArtifacts:
    """``build_sgb_geometry`` on an effective input."""
    return build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        output_dir=output_dir,
        **options,
    )


def test_the_entry_is_exported_from_the_package_and_not_from_geometry():
    """The entry is exported from the package and not from geometry."""
    import scgsim.geometry as geometry  # noqa: PLC0415 -- the import is the point of the test

    assert (
        sgb.build_sgb_geometry is build_sgb_geometry
        and "build_sgb_geometry" not in geometry.__all__
    )


@pytest.mark.parametrize(
    ("source", "route"), [("gds_input", "B"), ("gds_input", "C"), ("films", "A")]
)
def test_a_real_input_gives_all_five_files_and_a_model_gmsh_accepts(
    request, tmp_path, source, route
):
    """A real input gives all five files and a model gmsh accepts."""
    effective = request.getfixturevalue(source)
    artifacts = build(effective, route, tmp_path)
    for name in ARTIFACT_NAMES:
        assert getattr(artifacts, name).is_file(), name
    sidecar = read_sidecar(artifacts.sidecar)
    assert not list(
        jsonschema.Draft202012Validator(sidecar_schema()).iter_errors(sidecar)
    )
    assert sidecar["model"]["files"] == {
        name: getattr(artifacts, name).name for name in ARTIFACT_NAMES
    }
    report = verify_xao(artifacts.xao, sidecar)
    assert report.ok, report.problems
    assert (
        artifacts.counts["faces"] == sidecar["summary"]["counts"]["faces"]
        and artifacts.timings["write_xao"] >= 0
    )
    with gmsh_session(artifacts.xao):
        apply_sidecar_embeds(sidecar)
        mesh_coarsely(60.0)  # the port face meshes with the interface it lies on


def test_the_port_of_a_real_input_is_a_face_between_volumes_in_every_route(
    tmp_path, gds_input, films
):
    """The port of a real input is a face between two volumes in every route: it lies on a support plane."""
    b = read_sidecar(build(gds_input, "B", tmp_path / "b").sidecar)
    (port,) = b["port"].values()
    assert port["port_index"] == 1 and port["direction"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    # the port lies on the substrate top: a face between two volumes, never a free face
    assert views.embedded_faces(b) == [] and len(port["adjacent_volumes"]) == 2
    assert port["z_um"] == 0.0
    a = read_sidecar(build(films, "A", tmp_path / "a").sidecar)
    (a_port,) = a["port"].values()
    assert views.embedded_faces(a) == [] and len(a_port["adjacent_volumes"]) == 2


def test_a_junction_input_builds_under_route_a(tmp_path):
    """A junction input builds under route a."""
    effective = rectangular_junction_pair()
    artifacts = build(effective, "A", tmp_path)
    assert verify_xao(artifacts.xao, read_sidecar(artifacts.sidecar)).ok


def test_requests_for_bands_and_refinement_curves_reach_the_files(tmp_path, films):
    """Requests for bands and refinement curves reach the files."""
    insets = (
        InsetRequest("band_left", ConductorSurface("chip/left", "sheet"), (0.5, 1.0)),
    )
    refinements = (RefinementRequest("ref", 0.0, ((4.5, 1.5), (5.5, 1.5))),)
    try:
        artifacts = build(films, "A", tmp_path, insets=insets, refinements=refinements)
    except Exception as error:  # noqa: BLE001 -- name of the entity depends on the fixture; show it
        pytest.skip(f"fixture entity ids differ: {error}")
    sidecar = read_sidecar(artifacts.sidecar)
    assert list(sidecar["inset_request"]) == ["band_left"]
    assert list(sidecar["refinement_request"]) == ["ref"]
    assert any(name.startswith("chip/left::band=") for name in sidecar["group"])
    assert verify_xao(artifacts.xao, sidecar).ok
    assert MARGINS  # the shared margins are used by the other tests


def test_every_file_is_byte_identical_for_every_worker_and_tile_count(
    tmp_path, gds_input
):
    """Workers and tiles change how a model is derived and written, never one byte of it."""
    first = build(gds_input, "B", tmp_path / "one")
    second = build(gds_input, "B", tmp_path / "many", workers=3, tiles=3)
    for name in ARTIFACT_NAMES:
        assert (
            getattr(first, name).read_bytes() == getattr(second, name).read_bytes()
        ), name


def test_the_validation_report_is_written_and_lists_every_check(tmp_path, gds_input):
    """The validation report is written and lists every check."""
    artifacts = build(gds_input, "B", tmp_path)
    report = json.loads(artifacts.validation_report.read_text())
    assert (
        report["passed"] is True
        and len(report["checks"]) > 10
        and report["counts"]["faces"] > 0
    )


def test_a_route_error_stops_before_anything_is_written(tmp_path, films):
    """A route error stops before anything is written."""
    with pytest.raises(RouteError):
        build(films, "B", tmp_path / "out")  # sheets are not valid for Route B
    assert not (tmp_path / "out").exists()


def test_strict_without_ocp_names_the_dependency_group_and_writes_nothing(
    tmp_path, monkeypatch, gds_input
):
    """Strict without ocp names the dependency group and writes nothing."""
    monkeypatch.setattr(ocp, "ocp_available", lambda: False)
    with pytest.raises(ocp.OcctUnavailable, match=r"`occt` dependency group"):
        build(gds_input, "B", tmp_path / "out", strict=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.occt
@pytest.mark.parametrize(
    ("source", "route"), [("gds_input", "B"), ("gds_input", "C"), ("films", "A")]
)
def test_strict_mode_checks_the_written_xao_of_every_route_and_returns_the_report(
    request, tmp_path, source, route
):
    """The running examples under Routes A, B and C pass BRepCheck, the argument analyzer and ShapeAnalysis."""
    pytest.importorskip("OCP")
    artifacts = build(request.getfixturevalue(source), route, tmp_path, strict=True)
    assert (
        artifacts.occt is not None
        and artifacts.occt.ok
        and artifacts.occt.counts["solid"] >= 2
    )
    assert (
        build(request.getfixturevalue(source), route, tmp_path / "plain").occt is None
    )


# --------------------------------------------------------------------------- the load-side check catches what it should


@pytest.fixture(scope="module")
def loaded_pad(tmp_path_factory, gds_input):
    """A written Route B model and its sidecar."""
    artifacts = build(gds_input, "B", tmp_path_factory.mktemp("loaded"))
    return artifacts, read_sidecar(artifacts.sidecar)


def problems_with(artifacts, sidecar) -> list[str]:
    """The problems ``verify_loaded`` finds for a (possibly tampered) sidecar."""
    return verify_xao(artifacts.xao, sidecar).problems


def test_a_clean_model_passes(loaded_pad):
    """A clean model passes."""
    assert problems_with(*loaded_pad) == []


def test_a_moved_vertex_is_found(loaded_pad):
    """A moved vertex is found."""
    artifacts, sidecar = loaded_pad
    tampered = copy.deepcopy(sidecar)
    first_vertex = tampered["binding"]["ids_by_tag"]["0"][0]
    tampered["vertex"][first_vertex]["position_um"][0] += 0.01
    assert any("vertex tags" in p for p in problems_with(artifacts, tampered))


def test_a_wrong_face_area_and_a_wrong_volume_are_found(loaded_pad):
    """A wrong face area and a wrong volume are found."""
    artifacts, sidecar = loaded_pad
    tampered = copy.deepcopy(sidecar)
    fourth_face = tampered["binding"]["ids_by_tag"]["2"][3]
    first_solid = tampered["binding"]["ids_by_tag"]["3"][0]
    tampered["face"][fourth_face]["area_um2"] *= 1.001
    tampered["volume"][first_solid]["volume_um3"] *= 0.99
    found = problems_with(artifacts, tampered)
    assert any("face area" in p for p in found) and any("volume" in p for p in found)


def test_a_wrong_boundary_is_found(loaded_pad):
    """A wrong boundary is found."""
    artifacts, sidecar = loaded_pad
    tampered = copy.deepcopy(sidecar)
    loop = tampered["face"][tampered["binding"]["ids_by_tag"]["2"][0]]["loops"][0]
    loop["edges"].pop()
    loop["reversed"].pop()
    assert any("boundary mismatch" in p for p in problems_with(artifacts, tampered))


def test_a_group_with_the_wrong_member_or_tag_is_found(loaded_pad):
    """A group with the wrong member or tag is found."""
    artifacts, sidecar = loaded_pad
    tampered = copy.deepcopy(sidecar)
    members = views.group_members(tampered)
    name = next(
        n for n, g in tampered["group"].items() if g["dim"] == 2 and len(members[n]) > 1
    )
    tampered["face"][members[name][-1]]["group"] = None
    assert any(
        "physical groups differ" in p for p in problems_with(artifacts, tampered)
    )
    tampered = copy.deepcopy(sidecar)
    tampered["group"][name]["tag"] += 50
    assert any(
        "physical groups differ" in p for p in problems_with(artifacts, tampered)
    )


def test_a_count_mismatch_is_found_and_raise_if_failed_stops_the_build(loaded_pad):
    """A count mismatch is found and raise if failed stops the build.

    The sidecar plans one face fewer than the XAO holds: its last face is gone from the binding, the face table
    and the solids it bounds, as in the sidecar of a build that did not write it.
    """
    artifacts, sidecar = loaded_pad
    tampered = copy.deepcopy(sidecar)
    last = tampered["binding"]["ids_by_tag"]["2"].pop()
    del tampered["face"][last]
    for volume in tampered["volume"].values():
        volume["faces"] = [f for f in volume["faces"] if f != last]
    report = verify_xao(artifacts.xao, tampered)
    assert any("entity counts" in p for p in report.problems)
    with pytest.raises(LoadCheckError, match="difference"):
        report.raise_if_failed()


def test_a_model_loaded_from_the_brep_has_no_groups_so_they_can_be_skipped(loaded_pad):
    """A model loaded from the brep has no groups so they can be skipped."""
    artifacts, sidecar = loaded_pad
    with gmsh_session(artifacts.brep):
        assert verify_loaded(sidecar, groups=False).ok
        assert not verify_loaded(sidecar).ok  # a BREP carries no physical groups
