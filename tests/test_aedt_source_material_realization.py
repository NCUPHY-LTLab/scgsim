"""Offline source-material contracts; no native application or simulated readback."""

from dataclasses import replace
import hashlib
import json

import pytest

from scgsim.aedt.preparation.cohort import GEOMETRY_SOURCE_PATHS
from scgsim.aedt.runtime.families.q3d import _source_material_bindings
from scgsim.aedt.runtime.native.common import source_material_properties
from scgsim.aedt.specs.common import PdkMaterial
from scgsim.aedt.specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec


def _spec():
    outer = ((0, 0), (2, 0), (2, 2), (0, 2))
    return Q3dSpec(
        project_name="Materials",
        design_name="Design",
        aedt_version="2024.2",
        modeling="solid",
        materials={
            "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
            "si_a": PdkMaterial("si_a", "dielectric", False, "Silicon"),
            "si_b": PdkMaterial("si_b", "dielectric", False, "silicon"),
            "metal": PdkMaterial("metal", "superconductor", True, None),
        },
        vacuum_material_id="vacuum",
        bodies=(
            Q3dBodySpec("pad", outer, (), 0, 1, "metal", "signal", "Signal"),
            Q3dBodySpec("sub_a", outer, (), -2, -1, "si_a", "substrate", None),
            Q3dBodySpec("sub_b", outer, (), -1, 0, "si_b", "substrate", None),
        ),
        nets=(Q3dNetSpec("Signal", "Signal", ("pad",)),),
        run_control=None,
        solve_ac_rl=False,
        region_padding_um=(1, 1, 1, 1, 1, 1),
    )


def _source_spec(root, catalog):
    spec = _spec()
    path = root / GEOMETRY_SOURCE_PATHS["stack"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"materials": catalog}))
    stack_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    empty_sha = hashlib.sha256(b"").hexdigest()
    source = {
        "schema_version": "scgsim.aedt.q3d-geometry-source.v3",
        "source_gds_sha256": empty_sha,
        "source_geometry_sha256": empty_sha,
        "source_stack_sha256": stack_sha,
        "source_dbu_um": 0.001,
        "physical_ground_nets": [],
        "bodies": [body.to_payload() for body in spec.bodies],
        "files": {
            key: {"path": value, "sha256": stack_sha if key == "stack" else empty_sha}
            for key, value in GEOMETRY_SOURCE_PATHS.items()
        },
    }
    return replace(spec, geometry_source=source)


def test_projects_only_declared_properties_without_material_defaults():
    record = {
        "permittivity": 11.45,
        "permeability": 1,
        "loss_tangent": 0.0,
        "kind": "dielectric",
        "aedt_library_name": "Silicon",
        "metadata": {"note": "kept"},
    }
    assert source_material_properties(record) == {
        "permittivity": 11.45,
        "permeability": 1.0,
        "dielectric_loss_tangent": 0.0,
    }
    assert record["metadata"] == {"note": "kept"}
    assert source_material_properties({"kind": "vacuum"}) == {}


@pytest.mark.parametrize(
    "record, error",
    [
        ({"conductivity": 1.0}, TypeError),
        ({"permittivity": True}, TypeError),
        ({"permittivity": "11.45"}, TypeError),
        ({"permittivity": [11, 12, 13]}, TypeError),
        ({"permittivity": 0}, ValueError),
        ({"permittivity": float("nan")}, ValueError),
        ({"loss_tangent": -0.1}, ValueError),
        ({"loss_tangent": None}, TypeError),
    ],
)
def test_unsupported_declared_properties_fail_instead_of_using_library(record, error):
    with pytest.raises(error):
        source_material_properties(record)


def test_source_free_request_keeps_library_only_path(tmp_path):
    assert _source_material_bindings(tmp_path, _spec()) is None


def test_q3d_resolves_only_referenced_nonpec_source_materials(tmp_path):
    catalog = {
        "vacuum": {"permittivity": 1.0},
        "si_a": {"permittivity": 11.45},
        "si_b": {"loss_tangent": 0.0001},
        "metal": {"conductivity": "unsupported but PEC is separate"},
        "unused": {"permittivity": [1, 2, 3]},
    }
    bindings = _source_material_bindings(tmp_path, _source_spec(tmp_path, catalog))
    assert tuple(bindings) == ("si_a", "si_b", "vacuum")
    assert bindings["si_a"] == {"record": catalog["si_a"], "native_name": "Silicon"}
    assert bindings["si_b"]["record"] == {"loss_tangent": 0.0001}


def test_same_native_name_conflicts_reject_before_any_native_write(tmp_path):
    catalog = {
        "vacuum": {},
        "si_a": {"permittivity": 11.45},
        "si_b": {"permittivity": 12.0},
    }
    spec = _source_spec(tmp_path, catalog)
    with pytest.raises(ValueError, match="conflicting 'permittivity'"):
        _source_material_bindings(tmp_path, spec)


def test_same_native_name_matching_values_remain_separate_source_bindings(tmp_path):
    catalog = {
        "vacuum": {},
        "si_a": {"permittivity": 11.45},
        "si_b": {"permittivity": 11.45},
    }
    bindings = _source_material_bindings(tmp_path, _source_spec(tmp_path, catalog))
    assert bindings["si_a"]["record"] == bindings["si_b"]["record"]
    assert bindings["si_a"]["native_name"] == "Silicon"
    assert bindings["si_b"]["native_name"] == "silicon"


def test_source_normalization_preserves_declared_scalars_template_and_raw_conductivity():
    from scgsim.geometry.source._normalization import _materials

    declared = {
        "dielectric": {
            "material_kind": "dielectric",
            "relative_permittivity": 11.45,
            "permeability": 1.2,
            "aedt_library_name": "Silicon",
        },
        "conductor": {
            "material_kind": "conductive",
            "conductivity": {"value": 5.8e7, "unit": "S/m"},
        },
    }
    normalized = _materials(declared)
    assert normalized["dielectric"] == {
        "kind": "dielectric",
        "permittivity": 11.45,
        "permeability": 1.2,
        "aedt_library_name": "Silicon",
    }
    assert source_material_properties(normalized["dielectric"]) == {
        "permittivity": 11.45,
        "permeability": 1.2,
    }
    assert normalized["conductor"]["conductivity"] == {
        "value": 5.8e7,
        "unit": "S/m",
    }
    assert normalized["conductor"]["conductivity"] is not declared["conductor"]["conductivity"]
    with pytest.raises(TypeError, match="no established scalar unit contract"):
        source_material_properties(normalized["conductor"])


@pytest.mark.parametrize("permittivity", [True, [11.45, 12.0, 13.0]])
def test_source_normalization_keeps_unsupported_permittivity_for_backend_failure(permittivity):
    from scgsim.geometry.source._normalization import _materials

    normalized = _materials({
        "dielectric": {
            "kind": "dielectric",
            "relative_permittivity": permittivity,
            "aedt_library_name": "Silicon",
        },
    })["dielectric"]
    assert normalized["permittivity"] == permittivity
    assert type(normalized["permittivity"]) is type(permittivity)
    assert normalized["aedt_library_name"] == "Silicon"
    with pytest.raises(TypeError, match="requires a scalar number"):
        source_material_properties(normalized)
