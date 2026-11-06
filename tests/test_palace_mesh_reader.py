"""The streaming MSH 2.2 reader keeps authored ids, tags and optional sections."""

from pathlib import Path

import numpy as np
import pytest

from scgsim.palace.mesh.quality.reader import read_msh22

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
