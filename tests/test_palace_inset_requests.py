"""The SGB inset requests Palace sends for its Surface-EPR specs (file only, no Gmsh)."""

from __future__ import annotations

import pytest
from palace_sgb_support import effective_input, mesh_run, sidecar_of, surface_epr_specs

import scgsim.palace.mesh.build as build_module
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    InterfaceSurfaces,
)
from scgsim.palace.mesh.insets import (
    inset_requests,
    needs_unbanded_build,
    requests_wanted,
)
from scgsim.palace.preparation.surface_epr import normalize_surface_epr_specs

FILM = {"film_thickness_m": 2e-9, "permittivity": 4}


class StopAfterGeometry(Exception):
    """Raised in place of meshing, so a test sees the geometry builds without needing Gmsh."""


def geometry_builds(monkeypatch, route, directory, specs):
    """The ``insets`` of every SGB build ``build_route_mesh`` makes for ``specs`` before it would mesh."""
    builds: list[tuple[InsetRequest, ...]] = []
    real = build_module.build_sgb_geometry

    def recording(*args, **kwargs):
        builds.append(kwargs.get("insets", ()))
        return real(*args, **kwargs)

    def stop(*_args, **_kwargs):
        raise StopAfterGeometry

    monkeypatch.setattr(build_module, "build_sgb_geometry", recording)
    monkeypatch.setattr(build_module, "mesh_sgb", stop)
    with pytest.raises(StopAfterGeometry):
        mesh_run(effective_input(route, directory), route, directory, specs=specs)
    return builds


def test_the_default_path_sends_one_type_level_request_per_spec():
    """Each spec with margins is one ``InterfaceSurfaces`` request; a spec without margins sends nothing."""
    requests = inset_requests(surface_epr_specs(MA=[0.5, 1.0], MS=[2.0], SA=None))
    assert requests == (
        InsetRequest("epr_MA", InterfaceSurfaces("MA"), (0.5, 1.0)),
        InsetRequest("epr_MS", InterfaceSurfaces("MS"), (2.0,)),
    )
    assert not needs_unbanded_build(surface_epr_specs(MA=[0.5]))


def test_no_margins_request_no_insets():
    """Specs without margins, or no specs, request nothing."""
    assert not requests_wanted(surface_epr_specs(MA=None, MS=None))
    assert not requests_wanted({})
    assert inset_requests(surface_epr_specs(MA=None, MS=None)) == ()
    assert inset_requests({}) == ()


def test_the_spec_face_kinds_go_to_sgb_as_its_own_face_kinds():
    """Palace's face kinds are SGB's (sheet, top, bottom, interface); they are sent sorted and unchanged."""
    specs = surface_epr_specs(MA=[0.5], SA=[0.5])
    specs["MA"]["face_kinds"] = ["bottom", "top"]
    specs["SA"]["face_kinds"] = ["interface"]
    ma, sa = inset_requests(specs)
    assert ma.support == InterfaceSurfaces("MA", face_kinds=("bottom", "top"))
    assert sa.support == InterfaceSurfaces("SA", face_kinds=("interface",))


@pytest.mark.parametrize("key", ["inset_margins_um", "inset_margins_by_entity_um"])
def test_a_face_kind_sgb_cannot_band_is_rejected_where_margins_are_set(key):
    """``sidewall`` has no SGB equivalent, because SGB never bands sidewalls."""
    margins = [0.5] if key == "inset_margins_um" else {"pad": [0.5]}
    with pytest.raises(ValueError, match=r"\['sidewall'\].*never banded"):
        normalize_surface_epr_specs(
            {"MA": {**FILM, "face_kinds": ["top", "sidewall"], key: margins}},
            materials={},
        )


def test_a_sidewall_face_kind_stays_valid_for_baseline_rows_only():
    """Without margins no band is asked for, so a sidewall baseline row is still selectable."""
    specs = normalize_surface_epr_specs(
        {"MA": {**FILM, "face_kinds": ["sidewall"]}}, materials={}
    )
    assert specs["MA"]["face_kinds"] == ["sidewall"]


def test_the_default_path_builds_the_geometry_once(monkeypatch, tmp_path):
    """No unbanded build: the one build already carries the type-level requests."""
    specs = surface_epr_specs(MA=[0.5, 1.0], MS=[0.5], SA=[0.5])
    (only,) = geometry_builds(monkeypatch, "B", tmp_path, specs)
    assert {request.request_id for request in only} == {"epr_MA", "epr_MS", "epr_SA"}
    assert all(isinstance(r.support, InterfaceSurfaces) for r in only)


def test_specs_without_margins_build_the_geometry_once_without_bands(
    monkeypatch, tmp_path
):
    """No margins: one build and no request."""
    assert geometry_builds(monkeypatch, "B", tmp_path, surface_epr_specs(MA=None)) == [
        ()
    ]


@pytest.fixture(scope="module")
def route_b_sidecar(tmp_path_factory):
    """The sidecar of the plain (unbanded) Route B build."""
    directory = tmp_path_factory.mktemp("unbanded_b")
    return sidecar_of(effective_input("B", directory), "B", directory)


@pytest.fixture(scope="module")
def route_a_sidecar(tmp_path_factory):
    """The sidecar of the plain (unbanded) Route A build."""
    directory = tmp_path_factory.mktemp("unbanded_a")
    return sidecar_of(effective_input("A", directory), "A", directory)


def customised(by_entity, **margins):
    """Specs whose MA spec customises ``by_entity`` (and any other spec has its own margins)."""
    specs = surface_epr_specs(**margins)
    specs["MA"]["inset_margins_by_entity_um"] = by_entity
    return specs


def test_the_custom_path_sends_concrete_requests_and_excludes_them_from_the_type(
    route_b_sidecar,
):
    """The named conductor gets a request for its faces of the type; the rest of the type excludes it."""
    specs = customised({"pad": [0.25, 0.5]}, MA=[0.5, 1.0])
    assert needs_unbanded_build(specs)
    assert inset_requests(specs, route_b_sidecar) == (
        InsetRequest("epr::pad::top", ConductorSurface("pad", "top"), (0.25, 0.5)),
        InsetRequest(
            "epr_MA",
            InterfaceSurfaces("MA", exclude_entity_ids=("pad",)),
            (0.5, 1.0),
        ),
    )


def test_a_spec_with_only_customised_conductors_bands_only_those(route_b_sidecar):
    """Without ``inset_margins_um`` there is no type-level request: the other conductors stay unbanded."""
    specs = surface_epr_specs(MA=None)
    specs["MA"]["inset_margins_by_entity_um"] = {"pad": [0.5]}
    (request,) = inset_requests(specs, route_b_sidecar)
    assert request == InsetRequest(
        "epr::pad::top", ConductorSurface("pad", "top"), (0.5,)
    )


def test_a_customised_conductor_is_banded_on_each_face_that_has_a_row_of_the_type(
    route_b_sidecar,
):
    """MS rows lie on the bottom of a finite conductor, MA rows on its top."""
    specs = surface_epr_specs(MS=[1.0])
    specs["MS"]["inset_margins_by_entity_um"] = {"ground_0000": [0.5]}
    requests = inset_requests(specs, route_b_sidecar)
    assert requests[0].support == ConductorSurface("ground_0000", "bottom")
    assert requests[1].support == InterfaceSurfaces(
        "MS", exclude_entity_ids=("ground_0000",)
    )


def test_one_sheet_named_by_two_specs_is_one_request_with_its_one_set_of_margins(
    route_a_sidecar,
):
    """Route A: the MA and MS rows of a sheet lie on one face, and a face is named once."""
    specs = customised({"chip/left": [0.5]}, MA=None, MS=[2.0])
    specs["MS"]["inset_margins_by_entity_um"] = {"chip/left": [0.5]}
    requests = inset_requests(specs, route_a_sidecar)
    (sheet,) = [r for r in requests if isinstance(r.support, ConductorSurface)]
    assert sheet.support == ConductorSurface("chip/left", "sheet")
    assert sheet.margins_um == (0.5,)
    (ms,) = [r for r in requests if r.request_id == "epr_MS"]
    assert ms.support == InterfaceSurfaces("MS", exclude_entity_ids=("chip/left",))


@pytest.mark.parametrize("entity", ["nobody", "VACUUM_REGION", "substrate"])
def test_a_named_entity_without_a_face_of_the_type_is_an_error(route_b_sidecar, entity):
    """Unknown entities, and regions that are not conductors, have no MA face to band."""
    with pytest.raises(ValueError, match=f"names '{entity}', which has no MA face"):
        inset_requests(customised({entity: [0.5]}, MA=None), route_b_sidecar)


def test_a_named_entity_whose_faces_the_face_kinds_leave_out_is_an_error(
    route_b_sidecar,
):
    """The pad's MA face is its top, so ``face_kinds=['bottom']`` leaves it nothing to band."""
    specs = customised({"pad": [0.5]}, MA=None)
    specs["MA"]["face_kinds"] = ["bottom"]
    with pytest.raises(ValueError, match=r"no MA face to band among face_kinds"):
        inset_requests(specs, route_b_sidecar)


def test_customised_conductors_need_the_unbanded_sidecar():
    """The faces of named conductors are a fact of the built model."""
    with pytest.raises(ValueError, match="unbanded build"):
        inset_requests(customised({"pad": [0.5]}, MA=[1.0]))


def test_the_custom_path_builds_once_without_bands_and_once_with(monkeypatch, tmp_path):
    """One unbanded build finds the named faces; one banded build serves every type."""
    specs = customised({"pad": [0.25, 0.5]}, MA=[0.5, 1.0], MS=[0.5], SA=[0.5])
    unbanded, banded = geometry_builds(monkeypatch, "B", tmp_path, specs)
    assert unbanded == ()
    assert {request.request_id for request in banded} == {
        "epr::pad::top",
        "epr_MA",
        "epr_MS",
        "epr_SA",
    }
    (type_ma,) = (r for r in banded if r.request_id == "epr_MA")
    assert type_ma.support == InterfaceSurfaces("MA", exclude_entity_ids=("pad",))


def test_one_unbanded_build_serves_all_types_with_per_entity_margins(
    monkeypatch, tmp_path
):
    """Two types customise conductors; there is still one unbanded build."""
    specs = customised({"pad": [0.5]}, MA=[1.0], MS=[1.0])
    specs["MS"]["inset_margins_by_entity_um"] = {"ground_0000": [0.5]}
    builds = geometry_builds(monkeypatch, "B", tmp_path, specs)
    assert [len(insets) for insets in builds] == [0, 4]


@pytest.mark.parametrize(
    ("margins", "message"),
    [
        ({"pad": [0.0, 0.5]}, "finite positive"),
        ({"pad": [-1.0]}, "finite positive"),
        ({"pad": [1.0, 0.5]}, "strictly increasing"),
        ({"pad": [0.5, 0.5]}, "strictly increasing"),
        ({"pad": []}, "non-empty"),
    ],
)
def test_per_entity_margins_are_validated_like_the_spec_margins(margins, message):
    """Positive and strictly increasing, for every named conductor."""
    with pytest.raises(ValueError, match=message):
        normalize_surface_epr_specs(
            {"MA": {**FILM, "inset_margins_by_entity_um": margins}}, materials={}
        )


def test_per_entity_margins_are_normalized_and_kept_in_the_spec():
    """Floats, sorted by entity."""
    specs = normalize_surface_epr_specs(
        {"MA": {**FILM, "inset_margins_by_entity_um": {"pad": [1, 2], "a": (0.5,)}}},
        materials={},
    )
    assert specs["MA"]["inset_margins_by_entity_um"] == {
        "a": [0.5],
        "pad": [1.0, 2.0],
    }


@pytest.mark.parametrize("raw", [{}, ["pad"], {"": [0.5]}, {3: [0.5]}])
def test_per_entity_margins_must_be_a_mapping_of_entity_ids(raw):
    """A non-empty mapping with non-empty text keys."""
    with pytest.raises((TypeError, ValueError)):
        normalize_surface_epr_specs(
            {"MA": {**FILM, "inset_margins_by_entity_um": raw}}, materials={}
        )


def test_sa_interfaces_cannot_be_customised_by_conductor():
    """An SA face lies between solution regions, not on a conductor."""
    with pytest.raises(ValueError, match="use inset_margins_um for SA"):
        normalize_surface_epr_specs(
            {"SA": {**FILM, "inset_margins_by_entity_um": {"pad": [0.5]}}},
            materials={},
        )
