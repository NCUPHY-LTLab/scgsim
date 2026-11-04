"""Lumped ports lie on the hosts' physical support plane, from real effective inputs.

The plane is what the source authored (the film attachment face, projected through the common
physical plan), never the middle of the metal.  A port on a substrate surface therefore occupies
part of that surface: it is a face between the two volumes there, and the surface under it is no
longer part of the solution-interface (SA) bands.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
from sgb_inputs import (
    EffectiveInput,
    gds_pad_in_ground,
    rectangular_junction_pair,
    thin_film_pair,
)

from scgsim.geometry.sgb import build_sgb_geometry
from scgsim.geometry.sgb.cutfamilies.requests import InsetRequest, SolutionInterface
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import build_conformal_model
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.sgb.sidecar.files import read_sidecar
from scgsim.geometry.sgb.verification.loaded import verify_xao

PORT_GROUP = "PORT::PORT_SHEET__P1__0000"
SA_BANDS = "SA::substrate|VACUUM_REGION::interface::band="
SA_INTERFACE = SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0)
PORT_AREA_UM2 = 40.0  # 12 x 4 um region minus a 1 x 4 um landing on each host


def port_prism(effective: EffectiveInput):
    """The only port prism of an effective input's catalogue."""
    catalogue = build_prism_catalogue(effective.build_input, effective.representation)
    (port,) = [p for p in catalogue.prisms if p.port is not None]
    return port


def sa_band_areas(sidecar: dict) -> dict[int, float]:
    """The area of every SA band group at z = 0, by band index."""
    return {
        int(name.removeprefix(SA_BANDS)): group["area_um2"]
        for name, group in sidecar["physical_groups"].items()
        if name.startswith(SA_BANDS)
    }


@pytest.mark.parametrize("route", ["B", "C"])
def test_the_port_lies_on_the_substrate_top_and_is_a_face_between_two_volumes(
    tmp_path, route
):
    """z = 0 (the film's lower face), not the metal's mid-plane 0.1: the port has no free face."""
    effective = gds_pad_in_ground(tmp_path)
    assert port_prism(effective).z0 == port_prism(effective).z1 == 0
    model = build_conformal_model(
        effective.build_input, effective.representation, route
    )
    volumes = model.topology.volumes
    (port_id,) = model.routed.groups[PORT_GROUP].ids
    port = model.topology.faces[port_id]
    assert port.neg_volume is not None and port.pos_volume is not None
    assert port.neg_volume != port.pos_volume
    assert {
        volumes[port.neg_volume].domain.name,
        volumes[port.pos_volume].domain.name,
    } == {
        "substrate",
        "VACUUM_REGION",
    }
    assert not any(v.embedded_faces for v in volumes.values())
    assert not model.report.failures


@pytest.mark.parametrize("route", ["B", "C"])
def test_the_files_agree_the_port_is_on_the_interface(tmp_path, route):
    """Sidecar: z = 0, two adjacent volumes, nothing embedded; the XAO passes its own load-side check."""
    effective = gds_pad_in_ground(tmp_path)
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        output_dir=tmp_path / "out",
    )
    sidecar = read_sidecar(artifacts.sidecar)
    (port,) = sidecar["ports"]
    assert port["z_um"] == 0.0 and len(port["adjacent_volumes"]) == 2
    assert sidecar["embedded_faces"] == [] and port["host_volumes"] == []
    assert verify_xao(artifacts.xao, sidecar).ok


@pytest.mark.parametrize("route", ["B", "C"])
def test_a_port_removes_its_footprint_from_the_sa_bands_without_moving_a_band(
    tmp_path, route
):
    """SA band areas + the port area equal the SA band areas of the same geometry without the port."""
    insets = (InsetRequest("sa", SA_INTERFACE, (0.5, 1.0)),)

    def sidecar_of(with_port: bool) -> dict:
        directory = tmp_path / ("with" if with_port else "without")
        directory.mkdir()
        effective = gds_pad_in_ground(directory, with_port=with_port)
        artifacts = build_sgb_geometry(
            effective.build_input,
            effective.representation,
            route=route,
            output_dir=directory / "out",
            insets=insets,
        )
        return read_sidecar(artifacts.sidecar)

    with_port, without = sidecar_of(True), sidecar_of(False)
    bands_with, bands_without = sa_band_areas(with_port), sa_band_areas(without)
    assert set(bands_with) == set(bands_without) == {0, 1, 2}
    port_area = with_port["physical_groups"][PORT_GROUP]["area_um2"]
    assert port_area == pytest.approx(PORT_AREA_UM2)
    assert sum(bands_with.values()) + port_area == pytest.approx(
        sum(bands_without.values()), rel=1e-9
    )
    # the support is the ring around the pad: 40 x 40 minus 20 x 20, port or not
    assert sum(bands_without.values()) == pytest.approx(1200.0)
    for (
        index
    ) in bands_with:  # the port only ever takes area away, in every band it touches
        assert bands_with[index] <= bands_without[index] + 1e-9
    # the bands are measured from the metal edges, so the port moved none of them
    assert with_port["insets"][0]["support_area_um2"] == pytest.approx(
        without["insets"][0]["support_area_um2"]
    )
    assert with_port["insets"][0]["margins_um"] == [0.5, 1.0]
    # one PORT group, possibly several faces (band seams split it)
    assert [n for n in with_port["physical_groups"] if n.startswith("PORT::")] == [
        PORT_GROUP
    ]


def test_a_host_attached_at_its_upper_face_puts_the_port_on_the_metal_top():
    """The plane is the authored attachment face of the film, whichever face that is."""
    import tempfile  # noqa: PLC0415 -- a throwaway directory for the GDS and stack files
    from pathlib import Path  # noqa: PLC0415 -- used only here

    with tempfile.TemporaryDirectory() as directory:
        effective = gds_pad_in_ground(Path(directory), attachment=("upper", "upper"))
    assert port_prism(effective).z0 == Fraction(1, 5)


@pytest.mark.parametrize(
    ("attachment", "host"),
    [
        ((None, None), "ground_0000"),
        (("lower", None), "pad"),
        (("lower", "finite_volume"), "pad"),
    ],
)
def test_a_host_without_a_declared_attachment_face_fails_closed(
    tmp_path, attachment, host
):
    """No declaration, or a body declaration, has no support plane: the error names the host and the PDK duty."""
    effective = gds_pad_in_ground(tmp_path, attachment=attachment)
    with pytest.raises(InputError, match=rf"host '{host}'.*layer_attachment_face"):
        build_prism_catalogue(effective.build_input, effective.representation)


def test_hosts_with_different_authored_planes_fail_closed(tmp_path):
    """The ground attached at its lower face (z = 0) and the pad at its upper face (z = 0.2) disagree."""
    effective = gds_pad_in_ground(tmp_path, attachment=("lower", "upper"))
    with pytest.raises(InputError, match="do not share one authored support plane"):
        build_prism_catalogue(effective.build_input, effective.representation)


def test_a_port_needs_the_common_physical_plan(tmp_path):
    """Without the plan nothing projects the support plane, and nothing is guessed."""
    effective = gds_pad_in_ground(tmp_path)
    metadata = {
        k: v
        for k, v in effective.build_input.metadata.items()
        if k != "physical_representation"
    }
    bare = replace(effective.build_input, metadata=metadata)
    with pytest.raises(InputError, match="physical representation plan"):
        build_prism_catalogue(bare, effective.representation)


def test_route_a_sheets_are_their_own_support_plane():
    """A sheet host's anchor is the plane: the port stays at z = 0, for plain and junction inputs alike."""
    for effective in (thin_film_pair(0.5), rectangular_junction_pair()):
        assert port_prism(effective).z0 == 0
    effective = thin_film_pair(0.5)
    model = build_conformal_model(effective.build_input, effective.representation, "A")
    (port_id,) = model.routed.groups["PORT::port_sheet"].ids
    port = model.topology.faces[port_id]
    assert port.neg_volume != port.pos_volume
