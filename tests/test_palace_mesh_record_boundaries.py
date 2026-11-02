"""File-only mesh custody and exact structured-group mapping regressions."""

from pathlib import Path

import numpy as np
import pytest

from scgsim.palace.mesh.physical_groups import (
    augment_groups_from_live_records,
    build_groups_from_sgb_records,
)
from scgsim.palace.mesh.quality.reader import read_msh22
from scgsim.palace.mesh.source import build_mesh_manifest

_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
3 7 "authored vacuum"
$EndPhysicalNames
$Nodes
4
10 0 0 0
20 1 0 0
30 0 1 0
40 0 0 1
$EndNodes
$Elements
1
91 4 3 7 8 9 10 20 30 40
$EndElements
$Comments
retained optional section
$EndComments
"""


def test_streaming_mesh_reader_retains_authored_ids_tags_and_optional_sections(
    tmp_path: Path,
) -> None:
    """Preserve non-contiguous IDs and all tags without renumbering source data."""
    path = tmp_path / "source.msh"
    path.write_text(_MESH)
    result = read_msh22(path)
    np.testing.assert_array_equal(result.node_ids, [10, 20, 30, 40])
    np.testing.assert_array_equal(result.connectivity, [[10, 20, 30, 40]])
    np.testing.assert_array_equal(result.node_indices, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(result.raw_tags, [7, 8, 9])
    np.testing.assert_array_equal(result.tag_offsets, [[0, 3]])
    assert result.physical_names == {(3, 7): "authored vacuum"}
    assert result.optional_sections == {"Comments": 1}
    assert result.bytes_read == path.stat().st_size
    assert not result.coordinates.flags.writeable


@pytest.mark.parametrize(
    "before,after,message",
    (
        ("2.2 0 8", "2.2 1 8", "binary MSH"),
        ("40 0 0 1", "10 0 0 1", "duplicate node ID 10"),
        ("91 4 3 7 8 9", "91 4 -1 7 8 9", "invalid tag count"),
        ("$EndComments", "$EndUnexpected", "invalid delimiter"),
    ),
)
def test_streaming_mesh_reader_preserves_invalid_section_failures(
    tmp_path: Path, before: str, after: str, message: str
) -> None:
    """Reject malformed source formats at their original parse boundary."""
    path = tmp_path / "invalid.msh"
    path.write_text(_MESH.replace(before, after))
    with pytest.raises(ValueError, match=message):
        read_msh22(path)


def test_combined_interface_aliases_keep_one_physical_attribute() -> None:
    """MA/MS aliases retain their shared authored source identity and mesh tag."""
    record = {
        "dimension": 2,
        "physical_name": "authored-interface",
        "route": "A",
        "solver_use": "solver_active",
        "sgb_record": "final_physical_group",
        "role": "interface",
        "representation": "surface_sheet",
        "surface_id": "face-1",
        "interface_type": "MS_MA",
        "contact_kind": None,
        "face_kind": "INTERFACE",
        "owner_semantic_ids": ["metal-owner"],
        "adjacent_solution_volume_ids": ["substrate", "vacuum"],
        "conductor_component_id": "metal-component",
        "net_id": "metal-net",
        "equipotential_id": "metal-potential",
        "source_provenance": {"conductor_source_layer_name": "metal-source"},
        "physical_attribute": {"authored": "unchanged"},
    }
    groups = build_groups_from_sgb_records()
    augment_groups_from_live_records(
        records=[record],
        groups=groups,
        live={(2, "authored-interface"): (17, (23,))},
        route="A",
        solution_material_kinds={"substrate": "dielectric", "vacuum": "vacuum"},
    )
    assert set(groups["boundary_surfaces"]) == {"face-1__MA", "face-1__MS"}
    for group in groups["boundary_surfaces"].values():
        assert group["phys_group"] == 17
        assert group["tags"] == [23]
        assert group["surface_id"] == "face-1"
        assert group["source_id"] == "metal-source"
        assert group["physical_attribute"] == {"authored": "unchanged"}
        assert group["conductor_component_id"] == "metal-component"
    manifest = build_mesh_manifest(
        groups=groups,
        mesh_name="palace.msh",
        route="A",
        route_a_thin_film={"variant": "substrate_face"},
    )
    assert all(entry["attributes"] == [17] for entry in manifest["groups"])
    assert manifest["source"]["scgsim_sgb"] == "scgsim.geometry.compiler"


@pytest.mark.parametrize("role", (None, "domain_boundary"))
@pytest.mark.parametrize("solver_use", (None, "solver_active"))
def test_surface_source_record_metadata_is_read_before_role_dispatch(
    role: str | None, solver_use: str | None
) -> None:
    """Keep malformed source-record failure before fallback or role validation."""
    record = {
        "dimension": 2,
        "physical_name": "authored-surface",
        "route": "A",
        "role": role,
        "solver_use": solver_use,
        "metadata": {"source_record_ids": 3},
    }
    groups = build_groups_from_sgb_records()
    with pytest.raises(TypeError, match="not iterable"):
        augment_groups_from_live_records(
            records=[record],
            groups=groups,
            live={(2, "authored-surface"): (17, (23,))},
            route="A",
            solution_material_kinds={},
        )
    assert all(not entries for entries in groups.values())
