"""Native Mask configuration is independent of SGB geometry partitioning."""
from dataclasses import replace
import pytest
from test_palace_surface_integration import _group
from test_geometry_context_propagation import conductor_input
from scgsim.geometry import GeometryPlan
from scgsim.geometry.compiler import build_route_construction_plan
from scgsim.palace.preparation.surface_epr import build_surface_epr_postprocessing, normalize_surface_epr_specs
from scgsim.palace.results.resolve import _validate_surface_mask_config

@pytest.mark.parametrize("l0", [1e-6, 1e-3])
def test_native_mask_baseline_zero_positive_and_scale(l0):
    specs = normalize_surface_epr_specs({"MA": {"film_thickness_m": 2e-9, "permittivity": 4, "inset_margins_um": [0, 0.1]}}, materials={})
    rows, entries = build_surface_epr_postprocessing({"boundary_surfaces": {"pad": _group("pad", [1])}}, specs, model_l0_m=l0)
    assert len(rows) == 3
    assert "Mask" not in rows[0]
    assert rows[1]["Mask"] == {"Type": "Inset", "Margin": 0.0}
    assert rows[2]["Mask"] == {"Type": "Inset", "Margin": 0.1e-6 / l0}
    assert [entry.get("baseline_index") for entry in entries] == [1, 1, 1]
    assert [entry["mask"]["margin_index"] for entry in entries[1:]] == [0, 1]
    _validate_surface_mask_config({"Model": {"L0": l0}, "Boundaries": {"Postprocessing": {"Dielectric": rows}}}, {"entries": entries})

def test_unsupported_positive_mask_keeps_side_guard():
    group = _group("pad", [1])
    group["source_provenance"] = {"mask_support": {"positive_inset_supported": False, "unsupported_reason": "aggregated sides", "sides": ["top", "wall"]}}
    specs = normalize_surface_epr_specs({"MA": {"thickness": 0.001, "permittivity": 4, "inset_margins_um": [0.1]}}, materials={})
    with pytest.raises(ValueError, match="aggregated side"):
        build_surface_epr_postprocessing({"boundary_surfaces": {"pad": group}}, specs)

def test_withdrawn_geometry_request_fails_before_planning():
    source, _ = conductor_input()
    requested = replace(source, metadata={"surface_distance_bands": [{"parent_surface_id": "old"}]})
    with pytest.raises(ValueError, match="withdrawn"):
        build_route_construction_plan(requested, route="B")
    assert not hasattr(GeometryPlan, "set_surface_distance_bands")
