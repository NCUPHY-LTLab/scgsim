"""Call-local reuse of genuine gdstk extraction; no solver or mesh work."""

from contextvars import copy_context
import json
from threading import Thread

import gdstk
import pytest

from scgsim.geometry.source import _occurrence as occurrence


@pytest.fixture
def cells():
    a = gdstk.Cell("A")
    a.add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    b = gdstk.Cell("B")
    b.add(gdstk.rectangle((2, 0), (3, 1), layer=2))
    parent = gdstk.Cell("PARENT")
    parent.add(gdstk.rectangle((4, 0), (5, 1), layer=3))
    parent.add(gdstk.Reference(a), gdstk.Reference(b))
    return parent, a, b


@pytest.fixture
def extractions(monkeypatch):
    calls = []
    original = occurrence._extract_polygons_by_layer

    def observed(cell, *, excluded_reference_indexes=()):
        calls.append((cell, tuple(excluded_reference_indexes)))
        return original(cell, excluded_reference_indexes=excluded_reference_indexes)

    monkeypatch.setattr(occurrence, "_extract_polygons_by_layer", observed)
    return calls


def test_same_cell_reused_and_exact_exclusion_tuple_distinguished(cells, extractions):
    parent, _, _ = cells
    with occurrence._occurrence_extraction_scope():
        first = occurrence._polygons_by_layer(parent)
        assert occurrence._polygons_by_layer(parent) is first
        assert set(first) == {(1, 0), (2, 0), (3, 0)}
        assert set(
            occurrence._polygons_by_layer(parent, excluded_reference_indexes=(0,))
        ) == {(2, 0), (3, 0)}
        assert set(
            occurrence._polygons_by_layer(parent, excluded_reference_indexes=(1,))
        ) == {(1, 0), (3, 0)}
        occurrence._polygons_by_layer(parent, excluded_reference_indexes=(0, 1))
        occurrence._polygons_by_layer(parent, excluded_reference_indexes=(1, 0))
        with pytest.raises(TypeError):
            first[(4, 0)] = ()
    assert [row[1] for row in extractions] == [(), (0,), (1,), (0, 1), (1, 0)]


def test_cell_identity_and_nested_scope_are_separate(cells, extractions):
    parent, _, _ = cells
    other = gdstk.Cell("PARENT")
    with occurrence._occurrence_extraction_scope():
        first = occurrence._polygons_by_layer(parent)
        assert not occurrence._polygons_by_layer(other)
        with occurrence._occurrence_extraction_scope():
            nested = occurrence._polygons_by_layer(parent)
            assert nested is not first
        assert occurrence._polygons_by_layer(parent) is first
    assert len(extractions) == 3


def test_changes_outside_scope_are_read_fresh(cells, extractions):
    parent, _, _ = cells
    with occurrence._occurrence_extraction_scope():
        assert (4, 0) not in occurrence._polygons_by_layer(parent)
    parent.add(gdstk.rectangle((6, 0), (7, 1), layer=4))
    with occurrence._occurrence_extraction_scope():
        assert (4, 0) in occurrence._polygons_by_layer(parent)
    occurrence._polygons_by_layer(parent)
    occurrence._polygons_by_layer(parent)
    assert len(extractions) == 4


def test_exception_restores_cache_context(cells, extractions):
    parent, _, _ = cells
    with pytest.raises(ValueError, match="scope failure"):
        with occurrence._occurrence_extraction_scope():
            occurrence._polygons_by_layer(parent)
            raise ValueError("scope failure")
    assert occurrence._EXTRACTION_CACHE.get() is None
    occurrence._polygons_by_layer(parent)
    assert len(extractions) == 2


def test_copied_context_does_not_share_cache_with_worker(cells, extractions):
    parent, _, _ = cells
    with occurrence._occurrence_extraction_scope():
        first = occurrence._polygons_by_layer(parent)
        context = copy_context()
        results = []

        def extract():
            results.append(occurrence._polygons_by_layer(parent))
            results.append(occurrence._polygons_by_layer(parent))

        worker = Thread(target=context.run, args=(extract,))
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert occurrence._polygons_by_layer(parent) is first
    assert len(results) == 2 and results[0] is not results[1]
    assert len(extractions) == 3


def test_selector_outputs_are_detached_and_cached_polygons_unchanged(
    cells, extractions
):
    parent, _, _ = cells
    entity = occurrence._entity_from_layer_record(
        {
            "semantic_id": "metal",
            "role": "conductor",
            "material_id": "PEC",
            "layer": 1,
            "datatype": 0,
            "z_um": 0,
            "thickness_um": 1,
        },
        materials={"PEC": {"kind": "conductor"}},
    )
    with occurrence._occurrence_extraction_scope():
        polygons = occurrence._polygons_by_layer(parent)
        before = {
            key: [p.points.tolist() for p in values] for key, values in polygons.items()
        }
        first = occurrence._occurrence_polygons_for_entity(
            entity, cell=parent, transform=(1, 0, 0, 1, 0, 0)
        )
        second = occurrence._occurrence_polygons_for_entity(
            entity, cell=parent, transform=(1, 0, 0, 1, 0, 0)
        )
        assert first == second
        assert json.loads(json.dumps(first))
        first[0]["exterior"][0][0] = 99
        assert second[0]["exterior"][0][0] != 99
        assert before == {
            key: [p.points.tolist() for p in values] for key, values in polygons.items()
        }
    assert len(extractions) == 1


def test_public_adapter_reads_fresh_library_per_call(tmp_path, extractions):
    from scgsim.geometry import build_gds_stack_geometry_input

    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("COUPON")
    cell.add(gdstk.rectangle((0, 0), (4, 4), layer=1))
    gds = tmp_path / "source.gds"
    library.write_gds(gds)
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "semantic_id": "substrate",
                        "layer": 1,
                        "datatype": 0,
                        "role": "substrate",
                        "material_id": "Si",
                        "z_um": -1,
                        "thickness_um": 1,
                    }
                ],
                "materials": {"Si": {"kind": "dielectric", "permittivity": 11.9}},
                "solution_regions": {},
            }
        )
    )
    first = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
    second = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
    assert first == second
    assert len(extractions) == 2
    assert extractions[0][0] is not extractions[1][0]
    assert occurrence._EXTRACTION_CACHE.get() is None
