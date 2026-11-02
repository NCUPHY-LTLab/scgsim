"""Verify portable GDS preparation against real file codec behavior."""

from __future__ import annotations

import gdstk
import pytest

from scgsim.aedt.preparation.gds import gds_preflight, write_import_gds


def test_import_rewrite_preserves_source_geometry_and_original_bytes(tmp_path):
    """Numeric import layers change while cells, coordinates, and DBU stay fixed."""
    source = tmp_path / "source.gds"
    derived = tmp_path / "import.gds"
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    child = library.new_cell("Child")
    child.add(gdstk.rectangle((1, 2), (5, 8), layer=7, datatype=3))
    parent = library.new_cell("Parent")
    parent.add(gdstk.Reference(child, origin=(20, 30)))
    library.write_gds(source)
    original = source.read_bytes()

    write_import_gds(
        source, derived, [{"layer": 7, "datatype": 3, "native_import_layer": 42}]
    )
    rewritten = gdstk.read_gds(derived)
    observed = {cell.name: cell for cell in rewritten.cells}

    assert source.read_bytes() == original
    assert rewritten.unit == library.unit
    assert rewritten.precision == library.precision
    assert set(observed) == {"Child", "Parent"}
    assert (
        observed["Child"].polygons[0].points.tolist()
        == child.polygons[0].points.tolist()
    )
    assert observed["Child"].polygons[0].layer == 42
    assert observed["Child"].polygons[0].datatype == 0
    assert observed["Parent"].references[0].origin == (20, 30)
    assert gds_preflight(source)["polygon_layer_datatypes"] == {(7, 3)}
    assert gds_preflight(derived)["polygon_layer_datatypes"] == {(42, 0)}


def test_raw_path_is_rejected_before_gdstk_polygonizes_it(tmp_path):
    """A PATH input cannot silently become accepted polygon source geometry."""
    source = tmp_path / "path.gds"
    library = gdstk.Library()
    cell = library.new_cell("PathSource")
    cell.add(
        gdstk.FlexPath([(0, 0), (10, 0)], 1, layer=9, datatype=2, simple_path=True)
    )
    library.write_gds(source)

    with pytest.raises(ValueError, match=r"PATH layer/datatypes.*\(9, 2\)"):
        gds_preflight(source)
